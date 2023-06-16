# Rust builder
FROM registry.cn-hangzhou.aliyuncs.com/zt_gcr/cargo-chef:latest-rust-1.69 AS chef
WORKDIR /usr/src

FROM chef as planner
COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY benchmark benchmark
COPY router router
COPY launcher launcher
# set rust proxy to https://rsproxy.cn/
COPY third-pkgs/config ~/.cargo/config

RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

ARG GIT_SHA
ARG DOCKER_LABEL


### use curl
# RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
#     curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
#     unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
#     unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
#     rm -f $PROTOC_ZIP
###

# ========use file ===============
COPY third-pkgs/protoc-21.12-linux-x86_64.zip ./protoc-21.12-linux-x86_64.zip
RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    #curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP
# ===============================

COPY --from=planner /usr/src/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

COPY Cargo.toml Cargo.toml
COPY rust-toolchain.toml rust-toolchain.toml
COPY proto proto
COPY benchmark benchmark
COPY router router
COPY launcher launcher

# set rust proxy to https://rsproxy.cn/
COPY third-pkgs/config ~/.cargo/config
RUN cargo build --release

# Python builder
# Adapted from: https://github.com/pytorch/pytorch/blob/master/Dockerfile
FROM registry.cn-hangzhou.aliyuncs.com/zt_gcr/debian:bullseye-slim as pytorch-install

ARG PYTORCH_VERSION=2.0.0
ARG PYTHON_VERSION=3.9
ARG CUDA_VERSION=11.8
ARG MAMBA_VERSION=23.1.0-1
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch
# Automatically set by buildx
ARG TARGETPLATFORM

ENV PATH /opt/conda/bin:$PATH

RUN sed -i -E 's/(deb|security).debian.org/mirrors.huaweicloud.com/g' /etc/apt/sources.list && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        curl \
        git && \
        rm -rf /var/lib/apt/lists/*

# Install conda
# translating Docker's TARGETPLATFORM into mamba arches


# do not use linux/arm64 
# i will downliad mamba file for x86_64 and copy to build
COPY third-pkgs/Mambaforge-23.1.0-1-Linux-x86_64.sh mambaforge.sh
COPY third-pkgs/.condarc ~/.condarc

RUN chmod +x ./mambaforge.sh && \
    bash ./mambaforge.sh -b -p /opt/conda && \
    rm ./mambaforge.sh


# Install pytorch
# On arm64 we exit with an error code
RUN /opt/conda/bin/conda update -y conda &&  \
    /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -y "python=${PYTHON_VERSION}" pytorch==$PYTORCH_VERSION "pytorch-cuda=$(echo $CUDA_VERSION | cut -d'.' -f 1-2)" \
    /opt/conda/bin/conda clean -ya

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# CUDA kernels builder image
FROM pytorch-install as kernel-builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ninja-build \
        && rm -rf /var/lib/apt/lists/*

RUN /opt/conda/bin/conda install -c "nvidia/label/cuda-11.8.0"  cuda==11.8 && \
    /opt/conda/bin/conda clean -ya


# Build Flash Attention CUDA kernels
FROM kernel-builder as flash-att-builder

WORKDIR /usr/src

COPY server/Makefile-flash-att Makefile

# Build specific version of flash attention
RUN make build-flash-attention

# Build Transformers CUDA kernels
FROM kernel-builder as transformers-builder

WORKDIR /usr/src

COPY server/Makefile-transformers Makefile

# Build specific version of transformers
RUN BUILD_EXTENSIONS="True" make build-transformers

# Text Generation Inference base image
FROM registry.cn-hangzhou.aliyuncs.com/zt_gcr/cuda:11.8.0-base-ubuntu20.04 as base

# Conda env
ENV PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda

# Text Generation Inference base env
ENV HUGGINGFACE_HUB_CACHE=/data \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80

WORKDIR /usr/src

RUN sed -i "s/archive.ubuntu.com/mirrors.huaweicloud.com/g" /etc/apt/sources.list && \
    sed -i "s/security.ubuntu.com/mirrors.huaweicloud.com/g" /etc/apt/sources.list && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libssl-dev \
        ca-certificates \
        make \
        && rm -rf /var/lib/apt/lists/*

# Copy conda with PyTorch installed
COPY --from=pytorch-install /opt/conda /opt/conda

# Copy build artifacts from flash attention builder
COPY --from=flash-att-builder /usr/src/flash-attention/build/lib.linux-x86_64-cpython-39 /opt/conda/lib/python3.9/site-packages
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/layer_norm/build/lib.linux-x86_64-cpython-39 /opt/conda/lib/python3.9/site-packages
COPY --from=flash-att-builder /usr/src/flash-attention/csrc/rotary/build/lib.linux-x86_64-cpython-39 /opt/conda/lib/python3.9/site-packages

# Copy build artifacts from transformers builder
COPY --from=transformers-builder /usr/src/transformers /usr/src/transformers
COPY --from=transformers-builder /usr/src/transformers/build/lib.linux-x86_64-cpython-39/transformers /usr/src/transformers/src/transformers

# Install transformers dependencies
RUN cd /usr/src/transformers && pip install -e . --no-cache-dir && pip install einops --no-cache-dir

# Install server
COPY proto proto
COPY server server
COPY server/Makefile server/Makefile
RUN cd server && \
    make gen-server && \
    pip install -r requirements.txt && \
    pip install -r requirements-glm.txt && \
    pip install ".[bnb, accelerate]" --no-cache-dir

# Install benchmarker
COPY --from=builder /usr/src/target/release/text-generation-benchmark /usr/local/bin/text-generation-benchmark
# Install router
COPY --from=builder /usr/src/target/release/text-generation-router /usr/local/bin/text-generation-router
# Install launcher
COPY --from=builder /usr/src/target/release/text-generation-launcher /usr/local/bin/text-generation-launcher

# AWS Sagemaker compatbile image
FROM base as sagemaker

COPY sagemaker-entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

# Final image
FROM base

ENTRYPOINT ["text-generation-launcher"]
CMD ["--json-output"]
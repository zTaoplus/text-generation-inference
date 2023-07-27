import torch
import torch.nn as nn
import f16s4_gemm  # with CUDA kernels


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)
    
    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


class WQLinear(nn.Module):
    def __init__(self, qweight, qzeros, scales, w_bit, group_size, bias):
        super().__init__()
        
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")
        
        self.w_bit = w_bit
        self.group_size = group_size
        # quick sanity check (make sure aligment)
        
        self.register_buffer('qweight', qweight)
        self.register_buffer('qzeros', qzeros)
        self.register_buffer('scales', scales)
        if bias:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        
        self.out_features = qweight.shape[1] * (32 // 4)
        self.in_features = qweight.shape[0] 

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        out = f16s4_gemm.gemm_forward_cuda(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, 8)
        out = out + self.bias if self.bias is not None else out
        
        return out.reshape(out_shape)


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math
from torch import Tensor

CLIPMIN = 1e-5


class SimpleRMSNorm(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)



def activation_quant(x: Tensor,quant_rate=1.0):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): _description_

    Returns:
        _type_: _description_
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale

    x_quant =  x + (y - x).detach()
    return x_quant



def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
        disable_zero_point=False,
        is_weight_quant=False,
        **kwargs, 
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        assert 1 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        self.is_weight_quant = is_weight_quant
        self.shape = shape
        
        init_value = 4.             # inti value of learnable weight clipping
        if lwc:
            if group_size:
                dim1 = int(self.shape[0]*math.ceil(self.shape[1]/group_size))
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm symmetric quantization
            else:
                dim1 = self.shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)

        x = round_ste(x / scale)
        if round_zero_point is not None:
            x = x.add(round_zero_point)
        x = x.clamp(self.qmin, self.qmax)
        if round_zero_point is not None:
            x = x.sub(round_zero_point)
        x = x.mul(scale)
        if self.group_size:
            x = x.reshape(dim1, dim2)
        if self.deficiency > 0:
            x = x[:,:-self.deficiency]

        return x
    

    def forward(self, x: torch.Tensor,quant_rate=1.0):


        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)
        else:
            raise NotImplementedError()   
        
        # import ipdb;ipdb.set_trace()
        scale_dim = self.scale.shape[0]
        if self.group_size:
            scale_quant_dim_size =  clamp(math.ceil(self.scale.shape[0] * quant_rate),0,scale_dim)
        else:
            scale_quant_dim_size = scale_dim

        if quant_rate < 0.99:
            x_dim_size = x.shape[-1]
            quant_dim_size =  clamp(math.ceil(x.shape[-1] * quant_rate),0,x_dim_size)
            quant_x = self.fake_quant(x[...,:quant_dim_size], self.scale[:scale_quant_dim_size], self.round_zero_point[:scale_quant_dim_size])
            non_quant_x = x[..., quant_dim_size:]
            x = torch.cat((quant_x, non_quant_x), dim=-1)
        else:
            x = self.fake_quant(x, self.scale, self.round_zero_point)

        return x
    
    def quantize(self, x: torch.Tensor):
        return self.forward(x)
    
    def ready(self):
        return True

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1,self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
                x = torch.cat((x,pad_zeros),dim=1)
                x = x.reshape(-1,self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax =  x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor)*xmax
            xmin = self.sigmoid(self.lowbound_factor)*xmin
        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2**self.n_bits-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
        
    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point


# ============ CAT-Q：三值量化（LM 可学习调制 + ST 软化三值化）============


class _CatQScaleSigmoid(nn.Module):
    """sigmoid 参数化的可学习因子：sigmoid(bound) * alpha + beta"""

    def __init__(self, dim, alpha=2.0, beta=0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bound_factor = nn.Parameter(torch.zeros((dim, 1)))

    def forward(self):
        return torch.sigmoid(self.bound_factor) * self.alpha + self.beta


def _catq_factor_module(kind, dim):
    if kind == "sigmoid":
        return _CatQScaleSigmoid(dim, alpha=2.0)
    raise ValueError(f"Unsupported learnable_factor_act: {kind}")


class TernaryQuantizer(nn.Module):
    """CAT-Q 三值权重量化器（W -> {-1, 0, +1}）。

    - LM：3 个可学习因子 modulate mean(mu) / scale(alpha) / threshold(delta)
    - ST：训练侧用可微 tanh 软化（forward_soft），推理/weight_merge 侧用硬 round（forward）
    """

    def __init__(self, weight_quant_params, shape=None, is_weight_quant=True):
        super().__init__()
        del is_weight_quant
        self.group_size = weight_quant_params.get("group_size") or shape[-1]
        self.shift_mu = weight_quant_params.get("shift_mu", False)
        self.drop_quant_mu = weight_quant_params.get("drop_quant_mu", False)
        self.ter_scale_type = weight_quant_params.get("ter_scale_type", "absmean")
        self.learnable_scale = weight_quant_params.get("learnable_scale", False)
        self.learnable_mu = weight_quant_params.get("learnable_mu", False)
        self.learnable_round = weight_quant_params.get("learnable_round", False)
        self.init_round_thd = weight_quant_params.get("init_round_thd", 0.5)
        factor_kind = weight_quant_params.get("learnable_factor_act", "sigmoid")
        self.s0 = weight_quant_params.get("s0", 30.0)
        # ST 的 sharpness，训练时按 schedule 从 s_start 增大到 s0
        self.s_start = weight_quant_params.get("s_start", 2.0)
        self.register_buffer("cur_s", torch.tensor(self.s_start, dtype=torch.float32))

        dim = int(shape[0] * math.ceil(shape[1] / self.group_size))
        if self.learnable_scale:
            self.generate_scale_factor = _catq_factor_module(factor_kind, dim)
        if self.learnable_mu:
            if not self.shift_mu:
                raise ValueError("shift_mu must be True when learnable_mu is True")
            self.generate_mu_factor = _CatQScaleSigmoid(dim, alpha=2.0, beta=-1.0)
        if self.learnable_round:
            self.generate_round_factor = _catq_factor_module(factor_kind, dim)

    def set_cur_s(self, s):
        self.cur_s.fill_(float(s))

    def _stat(self, weight):
        grouped = weight.reshape(-1, self.group_size)
        if self.shift_mu:
            mean = grouped.mean(dim=-1, keepdim=True)
        else:
            mean = grouped.new_zeros((grouped.shape[0], 1))
        if self.ter_scale_type == "absmean":
            scale = (grouped - mean).abs().mean(dim=-1, keepdim=True) + 1e-6
        elif self.ter_scale_type == "variance":
            scale = grouped.std(dim=-1, keepdim=True, unbiased=False) + 1e-6
        else:
            raise ValueError(f"Unsupported ter_scale_type: {self.ter_scale_type}")
        if self.learnable_mu:
            mean = mean + self.generate_mu_factor() * scale
        if self.learnable_scale:
            scale = self.generate_scale_factor() * scale
        threshold = self.init_round_thd
        if self.learnable_round:
            threshold = threshold * self.generate_round_factor()
        return grouped, mean, scale, threshold

    def _soft_ternarize(self, z, threshold):
        s = self.cur_s
        return (torch.tanh(s * (z - threshold)) + torch.tanh(s * (z + threshold))) / (2.0 * torch.tanh(s))

    def _hard_ternarize(self, z, threshold):
        return torch.clamp(torch.round(z * 0.5 / threshold), -1.0, 1.0)

    def forward_soft(self, weight):
        """训练侧：tanh 软化三值化（可微）。"""
        original_shape = weight.shape
        grouped, mean, scale, threshold = self._stat(weight)
        z = (grouped - mean) / scale
        t = self._soft_ternarize(z, threshold)
        quantized = t * scale
        if self.shift_mu and not self.drop_quant_mu:
            quantized = quantized + mean
        return quantized.reshape(original_shape)

    def forward(self, weight, quant_rate=1.0):
        """推理/weight_merge 侧：硬三值化。"""
        del quant_rate
        original_shape = weight.shape
        grouped, mean, scale, threshold = self._stat(weight)
        z = (grouped - mean) / scale
        t = self._hard_ternarize(z, threshold)
        quantized = t * scale
        if self.shift_mu and not self.drop_quant_mu:
            quantized = quantized + mean
        return quantized.reshape(original_shape)
    


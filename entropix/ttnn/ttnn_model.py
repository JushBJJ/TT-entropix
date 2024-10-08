import ttnn
import torch

from entropix.ttnn.ttnn_weights import TTNNLayerWeights

def ttnn_rms_norm(x: ttnn.Tensor, w: ttnn.Tensor, eps: float=1e-6) -> ttnn.Tensor:
    return w * (x * ttnn.rsqrt(ttnn.mean(ttnn.pow(x, 2), dim=-1, keepdim=True) + eps))

def ttnn_feedforward(x: ttnn.Tensor, layer_weights: TTNNLayerWeights) -> ttnn.Tensor:
    return ttnn.linear(ttnn.silu(ttnn.linear(x, layer_weights.w1)) * ttnn.linear(x, layer_weights.w3), layer_weights.w2)

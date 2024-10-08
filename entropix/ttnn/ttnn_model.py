import ttnn
import torch

from entropix.ttnn.ttnn_weights import TTNNLayerWeights

def ttnn_rms_norm(x: ttnn.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.rms_norm(input_tensor=x, weight=w)

def ttnn_feedforward(x: ttnn.Tensor, layer_weights: TTNNLayerWeights) -> ttnn.Tensor:
    return ttnn.linear(ttnn.silu(ttnn.linear(x, layer_weights.w1)) * ttnn.linear(x, layer_weights.w3), layer_weights.w2)
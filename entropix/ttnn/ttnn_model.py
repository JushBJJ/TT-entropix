import ttnn
import torch

from typing import Tuple, Optional
from entropix.ttnn.ttnn_weights import TTNNLayerWeights
from entropix.config import LLAMA_1B_PARAMS

max_pos_embeddings = LLAMA_1B_PARAMS.max_position_embeddings
head_dim = LLAMA_1B_PARAMS.head_dim
rope_theta = LLAMA_1B_PARAMS.rope_theta
use_scaled_rope = LLAMA_1B_PARAMS.use_scaled_rope
start_pos = 0
seq_len = 2048

def ttnn_rms_norm(x: ttnn.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.rms_norm(input_tensor=x, weight=w)

def ttnn_feedforward(x: ttnn.Tensor, layer_weights: TTNNLayerWeights) -> ttnn.Tensor:
    return ttnn.linear(ttnn.silu(ttnn.linear(x, layer_weights.w1)) * ttnn.linear(x, layer_weights.w3), layer_weights.w2)

def ttnn_apply_rotary_emb(xq: ttnn.Tensor, xk: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, trans_mat: ttnn.Tensor, device: ttnn.Device) -> Tuple[torch.Tensor, torch.Tensor]:
    # Can't set dtype to float32 because GS doesn't support float32
    _xq = ttnn.experimental.rotary_embedding_llama(xq, cos, sin, trans_mat)
    _xk = ttnn.experimental.rotary_embedding_llama(xk, cos, sin, trans_mat)
    return _xq, _xk
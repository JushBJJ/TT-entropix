import ttnn
import math
import torch

from typing import Tuple, Optional
from entropix.ttnn.ttnn_weights import TTNNLayerWeights
from entropix.ttnn.utils import nearest_32
from entropix.config import LLAMA_1B_PARAMS

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
    # Also rotary_embedding_llama only supports bfloat16
    # see tt-metal/ttnn/operations/transformer/rotary_embedding_llama/device/rotary_embedding_llama_device_operation.cpp
    xq = ttnn.experimental.rotary_embedding(xq, cos, sin)
    xk = ttnn.experimental.rotary_embedding(xk, cos, sin)
    return xq, xk

def ttnn_attention(
    x: ttnn.Tensor, 
    layer_weights: TTNNLayerWeights, 
    model_params, 
    cur_pos: int, 
    layer_idx: int,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    trans_mat: ttnn.Tensor,
    kv_cache: ttnn.Tensor,
    device: ttnn.Device,
    attn_mask: Optional[ttnn.Tensor] = None
) -> ttnn.Tensor:
    # This implementation may be wrong (TODO: FIX)
    batch_size, seq_len, hidden_dim = x.shape
    
    head_dim = model_params.head_dim
    n_heads = model_params.n_local_heads
    n_kv_heads = model_params.n_local_kv_heads
    n_rep = n_heads // n_kv_heads  # For GQA support
    
    xq = ttnn.linear(x, layer_weights.wq).reshape((1, batch_size * n_heads, seq_len, head_dim))
    xk = ttnn.linear(x, layer_weights.wk).reshape((1, batch_size * n_kv_heads, seq_len, head_dim))
    xv = ttnn.linear(x, layer_weights.wv).reshape((batch_size, n_kv_heads, seq_len, head_dim))
    
    xq = ttnn.experimental.rotary_embedding(xq, cos, sin)
    xk = ttnn.experimental.rotary_embedding(xk, cos, sin)

    xq = xq.reshape((batch_size, n_heads, seq_len, head_dim))
    xk = xk.reshape((batch_size, n_kv_heads, seq_len, head_dim))

    ttnn.kv_cache.fill_cache_for_user_(kv_cache.k, xk, 0)
    ttnn.kv_cache.fill_cache_for_user_(kv_cache.v, xv, 0)

    xk = kv_cache.k
    xv = kv_cache.v

    xk = ttnn.repeat_interleave(xk, n_rep, 1)
    xv = ttnn.repeat_interleave(xv, n_rep, 1)

    scores = ttnn.matmul(xq, ttnn.transpose(xk, -2, -1))
    pre_scores = ttnn.div(scores, math.sqrt(head_dim))
    output = ttnn.transformer.scaled_dot_product_attention(
        xq, xk, xv,
        attn_mask=attn_mask,
        is_causal=False
    )
    out = ttnn.linear(output.reshape((batch_size, seq_len, n_heads * head_dim)), layer_weights.wo)
    return out, kv_cache, pre_scores
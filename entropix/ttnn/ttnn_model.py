import ttnn
import math
import torch

from typing import Tuple, Optional
from entropix.ttnn.ttnn_weights import TTNNLayerWeights, TTNNXfmrWeights
from entropix.ttnn.ttnn_kvcache import TTNN_KVCache
from entropix.ttnn.utils import nearest_32
from entropix.config import LLAMA_1B_PARAMS, ModelParams
from entropix.torch_stats import AttnStats

head_dim = LLAMA_1B_PARAMS.head_dim
rope_theta = LLAMA_1B_PARAMS.rope_theta
use_scaled_rope = LLAMA_1B_PARAMS.use_scaled_rope
start_pos = 0
seq_len = 2048

def ttnn_rms_norm(x: ttnn.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
    return ttnn.rms_norm(input_tensor=x, weight=w)

def ttnn_feedforward(x: ttnn.Tensor, layer_weights: TTNNLayerWeights) -> ttnn.Tensor:
    return ttnn.linear(ttnn.silu(ttnn.linear(x, layer_weights.w1)) * ttnn.linear(x, layer_weights.w3), layer_weights.w2)

def ttnn_attention(
    x: ttnn.Tensor, 
    layer_weights: TTNNLayerWeights, 
    model_params, 
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    kv_cache: TTNN_KVCache,
    attn_mask: Optional[ttnn.Tensor] = None
) -> ttnn.Tensor:
    # This implementation may be wrong (TODO: FIX)
    batch_size, seq_len, hidden_dim = x.shape
    
    head_dim = model_params.head_dim
    n_heads = model_params.n_local_heads
    n_kv_heads = model_params.n_local_kv_heads
    n_rep = n_heads // n_kv_heads
    
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

def ttnn_xfmr(
    xfmr_weights: TTNNXfmrWeights, 
    model_params: ModelParams, 
    tokens: ttnn.Tensor, 
    cos: ttnn.Tensor, 
    sin: ttnn.Tensor, 
    kvcache: TTNN_KVCache, 
    # device: ttnn.Device, # TODO: Add this arg in when AttnStats can be converted to ttnn
    attn_mask: Optional[ttnn.Tensor] = None
) -> Tuple[ttnn.Tensor, TTNN_KVCache, ttnn.Tensor, AttnStats]:
    h = ttnn.embedding(tokens, xfmr_weights.tok_embeddings, layout=ttnn.TILE_LAYOUT)
    attn_stats = AttnStats.new(
        bsz=tokens.shape[0],
        n_layers=model_params.n_layers,
        n_heads=model_params.n_local_heads
    )
    
    for i in range(model_params.n_layers):
        norm_x = ttnn_rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
        h_attn, kvcache, scores = ttnn_attention(
            norm_x,
            xfmr_weights.layer_weights[i],
            model_params,
            cos,
            sin,
            kvcache,
            attn_mask
        )
        
        # TTNN slicing is not so good yet so fallback to torch
        scores = ttnn.to_torch(scores)

        last_pos_scores = scores[:,:,-1,:]
        attn_stats = attn_stats.update(last_pos_scores, i)
        h = h + h_attn
        norm_x = ttnn_rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm)
        h_ffn = ttnn_feedforward(norm_x, xfmr_weights.layer_weights[i])
        h = h + h_ffn

    h = ttnn_rms_norm(h, xfmr_weights.norm)
    logits = ttnn.linear(h, xfmr_weights.output)
    
    return logits, kvcache, scores, attn_stats
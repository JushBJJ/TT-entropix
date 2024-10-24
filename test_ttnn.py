import ttnn
import torch

from entropix.config import LLAMA_1B_PARAMS
from entropix.torch_weights import XfmrWeights
from entropix.torch_model import (
    feed_forward, 
    rms_norm,
    apply_rotary_emb,
    attention
)
from entropix.torch_main import precompute_freqs_cis
from entropix.ttnn.ttnn_model import (
    ttnn_feedforward, 
    ttnn_rms_norm,
    ttnn_apply_rotary_emb,
    ttnn_attention
)
from entropix.ttnn.ttnn_weights import load_weights, convert_to_ttnn_xfmr_weights,TTNNXfmrWeights
from entropix.ttnn.llama_common import (
    compute_gather_cos_sin,
    get_rot_transformation_mat
)
from entropix.ttnn.ttnn_kvcache import TTNN_KVCache
from entropix.torch_kvcache import KVCache

head_dim = LLAMA_1B_PARAMS.head_dim
rope_theta = LLAMA_1B_PARAMS.rope_theta
use_scaled_rope = LLAMA_1B_PARAMS.use_scaled_rope
start_pos = 0
seq_len = 2048

def test_llama_rms(xfmr_weights: XfmrWeights, ttnn_xfmr_weights: TTNNXfmrWeights, device: ttnn.Device=None):
    x = torch.rand((1, 256, 2048), dtype=torch.bfloat16)
    ttnn_x = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)

    #for i in range(LLAMA_1B_PARAMS.n_layers):
    for i in range(1):
        print(f"Layer: {i}")
        
        # Test TTNN
        out_ttnn = ttnn_rms_norm(ttnn_x, ttnn_xfmr_weights.layer_weights[i].ffn_norm)
        out_ttnn = ttnn.to_torch(out_ttnn)

        # Test Golden
        out_golden = rms_norm(x, xfmr_weights.layer_weights[i].ffn_norm)

        print(f"TTNN: {out_ttnn}")
        print(f"Golden: {out_golden}")

def test_llama_ffw(xfmr_weights: XfmrWeights, ttnn_xfmr_weights: TTNNXfmrWeights, device: ttnn.Device=None):
    x = torch.rand((1, 256, 2048), dtype=torch.bfloat16)
    ttnn_x = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)

    for i in range(LLAMA_1B_PARAMS.n_layers):
        print(f"Layer: {i}")
        
        # Test TTNN
        out_ttnn = ttnn_feedforward(ttnn_x, ttnn_xfmr_weights.layer_weights[i])
        out_ttnn = ttnn.to_torch(out_ttnn)

        # Test Golden
        out_golden = feed_forward(x, xfmr_weights.layer_weights[i])

        print(f"TTNN: {out_ttnn}")
        print(f"Golden: {out_golden}")

def test_llama_apply_rotary_embedding(device: ttnn.Device = None):
    xq = torch.randn(1, 256, 32, 64, dtype=torch.float16)
    xk = torch.randn(1, 256, 32, 64, dtype=torch.float16)

    rope_theta = LLAMA_1B_PARAMS.rope_theta
    use_scaled_rope = LLAMA_1B_PARAMS.use_scaled_rope

    freq_cis = precompute_freqs_cis(64, 256, rope_theta, use_scaled_rope, dtype=torch.bfloat16)

    ttnn_xq = ttnn.from_torch(xq, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_xk = ttnn.from_torch(xk, device=device, layout=ttnn.TILE_LAYOUT)

    trans_mat = get_rot_transformation_mat(head_dim, device=device)
    cos, sin = compute_gather_cos_sin(head_dim, seq_len, torch.arange(0, 0 + 256), use_scaled_rope=use_scaled_rope, device=device)

    ttnn_q_out, ttnn_k_out = ttnn_apply_rotary_emb(ttnn_xq, ttnn_xk, cos, sin, trans_mat, device)
    q_out, k_out = apply_rotary_emb(xq, xk, freq_cis, dtype=torch.bfloat16)
    
    print(f"TTNN q_out: {ttnn_q_out}")
    print(f"Golden q_out: {q_out}")
    print(f"TTNN k_out: {ttnn_k_out}")
    print(f"Golden k_out: {k_out}")

def test_llama_attention(xfmr_weights: XfmrWeights, ttnn_xfmr_weights: TTNNXfmrWeights, device: ttnn.Device = None):
    model_params = LLAMA_1B_PARAMS
    x = torch.randn((1, 256, 2048), dtype=torch.bfloat16)
    ttnn_kv_cache = TTNN_KVCache(
        shape=(1, model_params.n_local_kv_heads, 256, model_params.head_dim), 
        device=device
    )
    cos, sin = compute_gather_cos_sin(head_dim, seq_len, torch.arange(0, 0 + 256), use_scaled_rope=use_scaled_rope, device=device)
    trans_mat = get_rot_transformation_mat(head_dim, device=device)
    freq_cis = precompute_freqs_cis(64, 256, rope_theta, use_scaled_rope, dtype=torch.bfloat16)
    kvcache = KVCache.new(model_params.n_layers, 1, 2048, model_params.n_local_kv_heads, model_params.head_dim)
    ttnn_x = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    
    ttnn_out, ttnn_kv_cache, ttnn_pre_scores = ttnn_attention(ttnn_x, ttnn_xfmr_weights.layer_weights[0], model_params, 0, 0, cos, sin, trans_mat, ttnn_kv_cache, device)
    out, kv_cache, pre_scores = attention(x, xfmr_weights.layer_weights[0], model_params, 0, 0, freq_cis, kvcache, attn_mask=None)

    print(f"TTNN: {ttnn_out}")
    print(f"Golden: {out}")

    # Measure difference between ttnn_out and out
    diff = torch.abs(ttnn.to_torch(ttnn_out) - out)
    print(f"Diff: {diff}")

device = ttnn.open_device(device_id=0)

try:
    xfmr_weights = load_weights()
    ttnn_xfmr_weights = convert_to_ttnn_xfmr_weights(xfmr_weights, device)
    test_llama_rms(xfmr_weights, ttnn_xfmr_weights, device=device)
    test_llama_ffw(xfmr_weights, ttnn_xfmr_weights, device=device)
    test_llama_apply_rotary_embedding(device=device)
    test_llama_attention(xfmr_weights, ttnn_xfmr_weights, device=device)
except Exception as e:
    raise e
finally:
    ttnn.close_device(device)
import torch
import ttnn

from entropix.torch_main import apply_scaling
from typing import Tuple

"""
(some) Snippets from: tt-metal/blob/main/models/demos/t3000/llama2_70b/tests/unit_tests/test_reshape_rotary_v2.py
"""

def precompute_freqs(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)

def compute_gather_cos_sin(dhead, end, position_ids, use_scaled_rope, device: ttnn.Device):
    cos, sin = precompute_freqs(dhead, end, use_scaled=use_scaled_rope)
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = torch.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    
    # Convert to TTNN
    cos = ttnn.from_torch(cos, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    sin = ttnn.from_torch(sin, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    return cos, sin

def get_rot_transformation_mat(dhead, device: ttnn.Device):
    # From tt-metal/models/demos/wormhole/mistral7b/tt/mistral_common.py
    # Override dhead to 32 for now, otherwise rotary_embedding_llama wont work
    dhead = 32
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    
    # Convert to TTNN
    rot_emb_matrix = ttnn.from_torch(rot_emb_matrix, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    return rot_emb_matrix
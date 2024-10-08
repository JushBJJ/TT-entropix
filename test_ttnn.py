import ttnn
import torch

from entropix.config import LLAMA_1B_PARAMS
from entropix.torch_weights import XfmrWeights
from entropix.torch_model import feed_forward, rms_norm
from entropix.ttnn.ttnn_model import ttnn_feedforward, ttnn_rms_norm
from entropix.ttnn.ttnn_weights import load_weights, convert_to_ttnn_xfmr_weights,TTNNXfmrWeights

def test_llama_rms(xfmr_weights: XfmrWeights, ttnn_xfmr_weights: TTNNXfmrWeights, device: ttnn.Device=None):
    x = torch.rand((1, 12, 2048), dtype=torch.bfloat16)
    ttnn_x = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)

    for i in range(LLAMA_1B_PARAMS.n_layers):
        print(f"Layer: {i}")
        
        # Test TTNN
        out_ttnn = ttnn_rms_norm(ttnn_x, ttnn_xfmr_weights.layer_weights[i].ffn_norm)
        out_ttnn = ttnn.to_torch(out_ttnn)

        # Test Golden
        out_golden = rms_norm(x, xfmr_weights.layer_weights[i].ffn_norm)

        print(f"TTNN: {out_ttnn}")
        print(f"Golden: {out_golden}")

def test_llama_ffw(xfmr_weights: XfmrWeights, ttnn_xfmr_weights: TTNNXfmrWeights, device: ttnn.Device=None):
    x = torch.rand((1, 12, 2048), dtype=torch.bfloat16)
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

device = ttnn.open_device(device_id=0)

try:
    xfmr_weights = load_weights()
    ttnn_xfmr_weights = convert_to_ttnn_xfmr_weights(xfmr_weights, device)
    test_llama_rms(xfmr_weights, ttnn_xfmr_weights, device=device)
    test_llama_ffw(xfmr_weights, ttnn_xfmr_weights, device=device)
except Exception as e:
    raise e
finally:
    ttnn.close_device(device)
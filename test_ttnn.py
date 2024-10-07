import ttnn
import torch
import numpy as np

from entropix.config import LLAMA_1B_PARAMS
from entropix.torch_weights import load_weights, LayerWeights
from entropix.torch_model import feed_forward

def test_llama_ffw():
    # TODO properly 
    device = ttnn.open_device(device_id=0)
    try:
        xfmr_weights = load_weights()
        x = torch.rand((1, 12, 2048), dtype=torch.bfloat16)
        ttnn_x = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)

        # convert ffn weights to ttnn and pad
        #for i in range(LLAMA_1B_PARAMS.n_layers):

        ttnn_layer_weights = LayerWeights(
            wq = ttnn.from_torch(xfmr_weights.layer_weights[0].wq, device=device, layout=ttnn.TILE_LAYOUT),
            wk = ttnn.from_torch(xfmr_weights.layer_weights[0].wk, device=device, layout=ttnn.TILE_LAYOUT),
            wv = ttnn.from_torch(xfmr_weights.layer_weights[0].wv, device=device, layout=ttnn.TILE_LAYOUT),
            wo = ttnn.from_torch(xfmr_weights.layer_weights[0].wo, device=device, layout=ttnn.TILE_LAYOUT),
            w1 = ttnn.from_torch(xfmr_weights.layer_weights[0].w1.T.contiguous(), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT),
            w2 = ttnn.from_torch(xfmr_weights.layer_weights[0].w2.T.contiguous(), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT),
            w3 = ttnn.from_torch(xfmr_weights.layer_weights[0].w3.T.contiguous(), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT),
            ffn_norm = ttnn.from_torch(xfmr_weights.layer_weights[0].ffn_norm, device=device, layout=ttnn.TILE_LAYOUT),
            attention_norm = ttnn.from_torch(xfmr_weights.layer_weights[0].attention_norm, device=device, layout=ttnn.TILE_LAYOUT)
        )
        
        # Test TTNN
        out_ttnn = ttnn.linear(ttnn.silu(ttnn.linear(ttnn_x, ttnn_layer_weights.w1)) * ttnn.linear(ttnn_x, ttnn_layer_weights.w3), ttnn_layer_weights.w2)
        out_ttnn = ttnn.to_torch(out_ttnn)

        # Test Golden
        out_golden = feed_forward(x, xfmr_weights.layer_weights[0])

        print(f"TTNN: {out_ttnn}")
        print(f"Golden: {out_golden}")
    except Exception as e:
        raise e
    finally:
        ttnn.close_device(device)
    """
    xfmr_weights = load_weights()
    model = llama_ffn(xfmr_weights.layer_weights[0])
    inputs = torch.rand((1, 12, 2048), dtype=torch.bfloat16)
    golden_output = model(inputs)
    forge_model = forge.compile(model, sample_inputs=[inputs])

    out = forge_model(inputs, xfmr_weights)
    print(out)
    """

test_llama_ffw()

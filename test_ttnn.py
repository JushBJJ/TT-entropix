# TODO move onto another file
# utils.py

import ttnn

from entropix.config import LLAMA_1B_PARAMS
from entropix.torch_weights import load_weights, LayerWeights
from entropix.torch_model import feed_forward as ffn

def test_llama_ffw():
    # TODO properly 
    device = ttnn.open_device(device_id=0)
    try:
        xfmr_weight = load_weights()
        ttnn_weights = [] 

        # convert ffn weights to ttnn and pad
        #for i in range(LLAMA_1B_PARAMS.n_layers):
        ttnn_weights.append(ttnn.from_torch(xfmr_weight.layer_weights[0].w1, device=device, layout=ttnn.TILE_LAYOUT))
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

import torch
import ttnn
import jax.numpy as jnp
import numpy as np

from entropix.torch_weights import load_weights, LayerWeights, XfmrWeights
from typing import List, NamedTuple
from pathlib import Path

class TTNNLayerWeights(NamedTuple):
    wq: ttnn.Tensor
    wk: ttnn.Tensor
    wv: ttnn.Tensor
    wo: ttnn.Tensor
    w1: ttnn.Tensor
    w2: ttnn.Tensor
    w3: ttnn.Tensor
    ffn_norm: ttnn.Tensor
    attention_norm: ttnn.Tensor

class TTNNXfmrWeights(NamedTuple):
    tok_embeddings: ttnn.Tensor
    norm: ttnn.Tensor
    output: ttnn.Tensor
    layer_weights: List[TTNNLayerWeights]

def xfmr_weight_to_ttnn(weight: torch.Tensor, device: ttnn.Device, transpose=True, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    if transpose:
        return ttnn.from_torch(weight.T.contiguous(), dtype=ttnn.bfloat16, device=device, layout=layout)
    return ttnn.from_torch(weight.contiguous(), dtype=ttnn.bfloat16, device=device, layout=layout)

def convert_to_ttnn_layer_weights(xfmr_layer_weights: LayerWeights, device: ttnn.Device) -> TTNNLayerWeights:
    print("Converting layer weights to ttnn...")
    return TTNNLayerWeights(
        wq = xfmr_weight_to_ttnn(xfmr_layer_weights.wq, device),
        wk = xfmr_weight_to_ttnn(xfmr_layer_weights.wk, device),
        wv = xfmr_weight_to_ttnn(xfmr_layer_weights.wv, device),
        wo = xfmr_weight_to_ttnn(xfmr_layer_weights.wo, device),
        w1 = xfmr_weight_to_ttnn(xfmr_layer_weights.w1, device),
        w2 = xfmr_weight_to_ttnn(xfmr_layer_weights.w2, device),
        w3 = xfmr_weight_to_ttnn(xfmr_layer_weights.w3, device),
        ffn_norm = xfmr_weight_to_ttnn(xfmr_layer_weights.ffn_norm, device, transpose=False),
        attention_norm = xfmr_weight_to_ttnn(xfmr_layer_weights.attention_norm, device, transpose=False)
    )

def convert_to_ttnn_xfmr_weights(xfmr_weights: XfmrWeights, device: ttnn.Device) -> TTNNXfmrWeights:
    return TTNNXfmrWeights(
        # TODO: Put tok_embeddings as ROW_MAJOR???
        tok_embeddings = xfmr_weight_to_ttnn(xfmr_weights.tok_embeddings, device, transpose=False, layout=ttnn.ROW_MAJOR_LAYOUT),
        norm = xfmr_weight_to_ttnn(xfmr_weights.norm, device, transpose=False),
        output = xfmr_weight_to_ttnn(xfmr_weights.output, device),
        layer_weights = [convert_to_ttnn_layer_weights(layer, device) for layer in xfmr_weights.layer_weights]
    )

def load_weights(ckpt_dir: Path = Path('weights/1B-Instruct'), n_layers: int = 16):
  w = {}
  layer_weights = []
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  with torch.inference_mode():
    for file in ckpt_dir.glob("*.npy"):
      name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
      print(f"Loading weight: {name}")
      np_weight = np.load(file, mmap_mode="r", allow_pickle=True)
      np_weight_bfloat16 = np_weight.view(jnp.bfloat16)
      np_weight_float32 = np_weight_bfloat16.astype(np.float32)
      weight = torch.from_numpy(np_weight_float32).to(torch.bfloat16)
      
      w[name] = weight.to(device)
    
      #! Skip comparing outputs for now
      # compare_outputs(torch_output=weight, jax_output=jax_weight)

    for i in range(n_layers):
      layer_weights.append(TTNNLayerWeights(
        wq=w[f'layers.{i}.attention.wq.weight'],
        wk=w[f'layers.{i}.attention.wk.weight'],
        wv=w[f'layers.{i}.attention.wv.weight'],
        wo=w[f'layers.{i}.attention.wo.weight'],
        w1=w[f'layers.{i}.feed_forward.w1.weight'],
        w2=w[f'layers.{i}.feed_forward.w2.weight'],
        w3=w[f'layers.{i}.feed_forward.w3.weight'],
        ffn_norm=w[f'layers.{i}.ffn_norm.weight'],
        attention_norm=w[f'layers.{i}.attention_norm.weight'],
      ))

    xfmr_weights = TTNNXfmrWeights(
      tok_embeddings=w['tok_embeddings.weight'],
      norm=w['norm.weight'],
      output=w['output.weight'],
      layer_weights=layer_weights
    )

    return xfmr_weights
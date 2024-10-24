import ttnn

class TTNN_KVCache:
    def __init__(self, shape: tuple, device: ttnn.Device):
        self.k = ttnn.zeros(shape, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        self.v = ttnn.zeros(shape, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
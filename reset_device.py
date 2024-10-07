import ttnn

device = ttnn.open_device(0)
ttnn.close_device(device)
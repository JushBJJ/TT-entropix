import ttnn

device = ttnn.open_device(device_id=0)
ttnn.close_device(device)
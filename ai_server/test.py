import torch
print("torch", torch.__version__)
print("cuda built-in:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
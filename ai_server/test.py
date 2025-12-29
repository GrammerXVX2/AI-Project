import torch

print("torch", torch.__version__)
# Use getattr and ignore type checker on torch.version
print("cuda built-in:", getattr(torch.version, "cuda", None))  # type: ignore[attr-defined]
print("cuda available:", torch.cuda.is_available())
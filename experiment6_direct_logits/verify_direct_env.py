import torch
import transformers

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("torch CUDA build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("visible GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name} | {p.total_memory / (1024**3):.1f} GiB")

import torch


def check_cuda_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_cached(0)
    a = torch.cuda.memory_allocated(0)
    f = c-a  # free inside cache
    print(f"Total: {t}\nCached: {c}\nAllocated: {a}\nFree: {f}\n\n")

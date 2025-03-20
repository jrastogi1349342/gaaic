import torch

def get_mem_used(): 
    free, total = torch.cuda.mem_get_info("cuda")
    mem_used_MB = (total - free) / 1024 ** 2
    return mem_used_MB

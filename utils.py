import torch
from torchviz import make_dot

def get_mem_used(): 
    free, total = torch.cuda.mem_get_info("cuda")
    mem_used_MB = (total - free) / 1024 ** 2
    return mem_used_MB

def count_nan_in_weights(model):
    nan_count = 0
    for param in model.parameters():
        if torch.isnan(param).any():  # Check if there are any NaNs in the parameter
            nan_count += torch.isnan(param).sum().item()
    
    return nan_count

def count_inf_in_weights(model):
    inf_count = 0
    for param in model.parameters():
        if torch.isinf(param).any():  # Check if there are any Infs in the parameter
            inf_count += torch.isinf(param).sum().item()
    
    return inf_count

def display_comp_graph(output, file_name): 
    dot = make_dot(output)
    dot.render(file_name, format="png")
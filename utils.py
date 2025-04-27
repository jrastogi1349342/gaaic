import torch
from torchviz import make_dot
import matplotlib.pyplot as plt

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

def plot_learning_curve(avg_rewards, freq): 
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(avg_rewards) + 1), avg_rewards, label='Deterministic Episode Rewards')
    plt.xlabel(f'Episode (x{freq})')
    plt.ylabel('Total Reward')
    plt.title('Average Reward for Deterministic Rollouts (No Exploration)')
    plt.grid(True)
    plt.show()
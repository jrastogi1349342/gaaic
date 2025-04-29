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

def plot_learning_curve(avg_rewards, freq, file_name): 
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(avg_rewards) + 1), avg_rewards, label='Deterministic Episode Rewards')
    plt.xlabel(f'Episode (x{freq})')
    plt.ylabel('Total Reward')
    plt.title('Average Reward for Deterministic Rollouts (No Exploration)')
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()

def plot_per_step(arr, freq, file_name, name): 
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(arr) + 1), arr, label=name)
    plt.xlabel(f'Episode (x{freq})')
    plt.ylabel('Loss')
    plt.title(f'{name} over Timesteps')
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()



def plot_actor_loss(actor_losses, freq, file_name): 
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(actor_losses) + 1), actor_losses, label='Actor Losses')
    plt.xlabel(f'Episode (x{freq})')
    plt.ylabel('Loss')
    plt.title('Actor Loss over Timesteps')
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()


def plot_critic_loss(critic_losses, freq, file_name): 
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(critic_losses) + 1), critic_losses, label='Critic Losses')
    plt.xlabel(f'Episode (x{freq})')
    plt.ylabel('Loss')
    plt.title('Critic Loss over Timesteps')
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()


def plot_alpha_loss(alpha_losses, freq, file_name): 
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(alpha_losses) + 1), alpha_losses, label='Alpha Losses')
    plt.xlabel(f'Episode (x{freq})')
    plt.ylabel('Loss')
    plt.title('Alpha Loss over Timesteps')
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()


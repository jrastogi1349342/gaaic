import torch
from torchvision.ops import box_iou
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage
from dataloader import denormalize_batch

def calc_rewards(orig, perturbed, orig_results, perturbed_results, step_idx, done=None, target=None, device="cuda"): 
    rewards = np.array([0.0])
    dones = np.array([False])

    # This is to keep batching for the rollouts
    if done == True: 
        dones[0] = True
        return rewards, dones

    orig_probs = orig_results.probs
    perturbed_probs = perturbed_results.probs

    same_class = True if orig_probs.top1 == perturbed_probs.top1 else False
    conf_diff = perturbed_probs.top1conf - orig_probs.top1conf # negative is the direction I want, if the classes are the same

    if same_class:
        rewards[0] = -5 - 2.5 * conf_diff.item()
    else:  
        dones[0] = True
        rewards[0] = 10

    rewards[0] -= 0.075 * step_idx

    return rewards, dones

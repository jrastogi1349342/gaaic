import torch
from torchvision.ops import box_iou
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage
from dataloader import denormalize_batch

# Empty: Erase that bounding box
# Untargeted: Change class of bounding box to something else, doesn't matter what that is
# Targeted: Change class of bounding box to specific target class
def calc_rewards(orig, perturbed, orig_results, perturbed_results, step_idx, done=None, target=None, device="cuda"): 
    rewards = np.array([0.0])
    dones = np.array([False])

    # This is to keep batching for the rollouts
    if done == True: 
        # print("Early return")
        dones[0] = True
        return rewards, dones

    # TODO plot images to verify ranges and that yolo worked properly

    # to_pil = ToPILImage()
    # for i, r in enumerate(orig):
    #     img = to_pil(r)
    #     img.show()

    # for i, r in enumerate(perturbed):
    #     img = to_pil(r)
    #     img.show()

    # for i, r in enumerate(orig_results):
    #     im_bgr = r.plot()  # BGR-order numpy array
    #     r.show()

    # for i, r in enumerate(perturbed_result):
    #     im_bgr = r.plot()  # BGR-order numpy array
    #     r.show()

    # print(orig_results.shape)

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

    # print(perturbed.shape, orig.shape)

    # l2_norms = np.linalg.norm((perturbed - orig).reshape(len(perturbed), -1), ord=2, axis=1)
    # print("L2 norms:", l2_norms)
    # print("Rewards without L2 norm", rewards)
    # rewards -= l2_norms
    # dones_torch = np.array(dones)

    # print(rewards_torch, dones_torch)

    return rewards, dones

# boxes_A = torch.tensor([
#     [50, 80, 180, 220],
#     [100, 150, 200, 250],
#     [90, 130, 190, 240], 
#     [-20, 50, -90, 180]
# ], device="cuda")  # Bounding boxes in image A

# boxes_B = torch.tensor([
#     [120, 140, 210, 260],
#     [50, 80, 180, 220],
#     [200, 200, 300, 350]
#     # ,
#     # [300, 350, 400, 450]
# ], device="cuda")  # Bounding boxes in image B

# perfect_matches, imperfect_matches, unused_orig, unused_perturbed = greedy_match_bboxes(boxes_A, boxes_B)
# print("Perfect matches:", perfect_matches)
# print("Imperfect matches:", imperfect_matches)
# print("Unused orig:", unused_orig)
# print("Unused perturbed:", unused_perturbed)

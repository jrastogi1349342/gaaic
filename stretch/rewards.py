import torch
from torchvision.ops import box_iou
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage
from dataloader import denormalize_batch

def greedy_match_bboxes(orig_xyxy, perturbed_xyxy, min_iou=0.95): 
    # M x N
    ious = box_iou(orig_xyxy, perturbed_xyxy)

    # print(ious)

    perfect_matches = {}
    imperfect_matches = {}
    used_orig = set()
    used_perturbed = set()

    sorted_orig_ind = torch.argsort(torch.max(ious, dim=1)[0], descending=True)

    for i in sorted_orig_ind: 
        sorted_perturbed_ind = torch.argsort(ious[i], descending=True)

        for j in sorted_perturbed_ind: 
            j = j.item()
            iou = ious[i.item(), j].item()

            if j not in used_perturbed and iou > min_iou: 
                perfect_matches[i.item()] = j
                used_orig.add(i.item())
                used_perturbed.add(j)
                break

    for i in sorted_orig_ind: 
        if i.item() in perfect_matches: 
            continue

        sorted_perturbed_ind = torch.argsort(ious[i], descending=True)
        # print(f"i: {i}:\t{sorted_perturbed_ind}")
        i = i.item()

        best_imperfect_iou = 0
        best_imperfect_match = -1

        for j in sorted_perturbed_ind: 
            j = j.item()

            # print(f"j: {j}\tUsed j: {used_perturbed}\tj in used: {j in used_perturbed}")

            if j in used_perturbed: 
                continue

            iou = ious[i, j].item()

            if 0 < iou and iou < min_iou and iou > best_imperfect_iou: 
                best_imperfect_iou = iou
                best_imperfect_match = j

            # print(f"{best_imperfect_iou}\t{best_imperfect_match}")
                
        if best_imperfect_match != -1: 
            imperfect_matches[i] = (best_imperfect_match, best_imperfect_iou)
            used_orig.add(i)
            used_perturbed.add(best_imperfect_match)

    
    unused_orig = set([i for i in range(len(orig_xyxy))])
    unused_perturbed = set([i for i in range(len(perturbed_xyxy))])
    
    unused_orig.difference_update(used_orig)
    unused_perturbed.difference_update(used_perturbed)

    return perfect_matches, imperfect_matches, unused_orig, unused_perturbed

def associate(orig_boxes, perturbed_boxes): 
    same_det = 0
    same_spot_diff_cls = 0
    diff_spot_same_cls = 0
    diff_spot_diff_cls = 0
    removed = 0 # removed from orig
    added = 0 # added to perturbation

    orig_xyxy = orig_boxes.xyxy
    perturbed_xyxy = perturbed_boxes.xyxy

    if len(orig_xyxy) != 0 and len(perturbed_xyxy) == 0: 
        removed = len(orig_xyxy)
    
    elif len(orig_xyxy) == 0 and len(perturbed_xyxy) != 0: 
        added = len(perturbed_xyxy)

    elif len(orig_xyxy) != 0 and len(perturbed_xyxy) != 0:
        perfect_matches, imperfect_matches, unused_orig, unused_perturbed = greedy_match_bboxes(orig_xyxy, perturbed_xyxy)

        for orig_idx, perturbed_idx in perfect_matches.items(): 
            orig_cls = orig_boxes[orig_idx].cls
            perturbed_cls = perturbed_boxes[perturbed_idx].cls

            if torch.equal(orig_cls, perturbed_cls): 
                same_det += 1
            else: 
                same_spot_diff_cls += 1

        for orig_idx, (perturbed_idx, iou) in imperfect_matches.items(): 
            orig_cls = orig_boxes[orig_idx].cls
            perturbed_cls = perturbed_boxes[perturbed_idx].cls

            if torch.equal(orig_cls, perturbed_cls): 
                diff_spot_same_cls += iou
            else: 
                diff_spot_diff_cls += iou

        removed = len(unused_orig)
        added = len(unused_perturbed)

    return same_det, same_spot_diff_cls, diff_spot_same_cls, diff_spot_diff_cls, removed, added


# Empty: Erase that bounding box
# Untargeted: Change class of bounding box to something else, doesn't matter what that is
# Targeted: Change class of bounding box to specific target class
def calc_rewards(orig, perturbed, obj_detector, goal, target=None, device="cuda"): 
    assert goal in {"empty", "untargeted", "targeted"}, f"Goal {goal} not found"

    rewards = [0] * len(orig)
    dones = [False] * len(orig)

    # TODO plot images to verify ranges and that yolo worked properly

    # to_pil = ToPILImage()
    # for i, r in enumerate(orig):
    #     img = to_pil(r)
    #     img.show()

    # for i, r in enumerate(perturbed):
    #     img = to_pil(r)
    #     img.show()

    with torch.no_grad(): 
        orig_results = obj_detector(orig)
        perturbed_results = obj_detector(perturbed)

    # for i, r in enumerate(orig_results):
    #     im_bgr = r.plot()  # BGR-order numpy array
    #     r.show()

    # for i, r in enumerate(perturbed_result):
    #     im_bgr = r.plot()  # BGR-order numpy array
    #     r.show()

    for i in range(len(orig)): 
        orig_result = orig_results[i].boxes
        perturbed_result = perturbed_results[i].boxes

        if len(perturbed_result) == 0: 
            dones[i] = True

        same_det, same_spot_diff_cls, diff_spot_same_cls, diff_spot_diff_cls, removed, added = associate(orig_result, perturbed_result)

        # TODO normalize by number of detected objects 
        if goal == "empty": 
            # TODO rewamp this reward system, prof said to bring rewards down
            rewards[i] = -50 * same_det + -25 * same_spot_diff_cls + -15 * diff_spot_same_cls + -15 * diff_spot_diff_cls + 50 * removed + -50 * added
            sum = same_det + same_spot_diff_cls + diff_spot_same_cls + diff_spot_diff_cls + removed + added
            if sum != 0: 
                rewards[i] /= sum
            else: 
                rewards[i] -= 200 

            pass

        # TODO implement
        elif goal == "untargeted": 

            pass
        else: 
            assert target is not None, f"Target should be dictionary mapping from original class pred to desired class pred"

            pass

    l2_norms = np.linalg.norm((perturbed - orig).reshape(len(perturbed), -1), ord=2, axis=1)
    # print("L2 norms:", l2_norms)
    # print("Rewards without L2 norm", rewards)
    rewards_torch = np.array(rewards) - l2_norms
    dones_torch = np.array(dones)

    return rewards_torch, dones_torch

def calc_rewards(orig, perturbed, orig_results, perturbed_results, goal, done=None, target=None, device="cuda"): 
    assert goal in {"empty", "untargeted", "targeted"}, f"Goal {goal} not found"

    rewards = np.array([0.0])
    dones = np.array([False])

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

    orig_result = orig_results.boxes
    perturbed_result = perturbed_results.boxes

    if len(perturbed_result) == 0: 
        dones[0] = True

    same_det, same_spot_diff_cls, diff_spot_same_cls, diff_spot_diff_cls, removed, added = associate(orig_result, perturbed_result)

    # TODO normalize by number of detected objects 
    if goal == "empty": 
        # TODO rewamp this reward system, prof said to bring rewards down
        rewards[0] = -50 * same_det + -25 * same_spot_diff_cls + -15 * diff_spot_same_cls + -15 * diff_spot_diff_cls + 50 * removed + -50 * added
        sum = same_det + same_spot_diff_cls + diff_spot_same_cls + diff_spot_diff_cls + removed + added
        if sum != 0: 
            rewards[0] /= sum
        else: 
            rewards[0] -= 200 

        pass

    # TODO implement
    elif goal == "untargeted": 

        pass
    else: 
        assert target is not None, f"Target should be dictionary mapping from original class pred to desired class pred"

        pass

    # print(perturbed.shape, orig.shape)

    l2_norms = np.linalg.norm((perturbed - orig).reshape(len(perturbed), -1), ord=2, axis=1)
    # print("L2 norms:", l2_norms)
    # print("Rewards without L2 norm", rewards)
    rewards -= l2_norms
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

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
import os
from ultralytics import YOLO
from stable_baselines3.common.utils import polyak_update
from tqdm import trange
import time
import lpips

from stable_baselines3.common.vec_env import DummyVecEnv


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader import train_dataloader, test_dataloader, val_dataloader, denormalize_batch, renormalize_batch
from environment import DataloaderEnv

from utils import *
from models import *

# To run: python3 main/train.py

parser = argparse.ArgumentParser(description='Command Line Args for Image Classifier')
parser.add_argument("--save_lc", action="store_false", help="Save all files")
args = parser.parse_args()


def make_env_fn(dataset, idx, latent_dim):
    def _init():
        return DataloaderEnv(dataset, idx, latent_dim, batch_size=1)
    return _init

def make_vec_env(dataset, num_envs, latent_dim):
    return DummyVecEnv([make_env_fn(dataset, idx, latent_dim) for idx in range(num_envs)])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gamma = 0.5
latent_dim = 32
batch_size = 1 # must be 1, use multiple environments for parallel episodes
training_batch_size = 128
num_train_envs = 128
num_timesteps = 80
gradient_update_freq = 128
train_data = train_dataloader(batch_size=batch_size, num_workers=0)
img_classifier = YOLO("yolo11n-cls.pt").to(device).eval()

train_envs = make_vec_env(train_data, num_train_envs, latent_dim)

num_test_envs = 20
test_data = test_dataloader(batch_size=batch_size, num_workers=0)
eval_envs = make_vec_env(test_data, num_test_envs, latent_dim)
test_freq = 1 # every x timesteps in training

time_save = time.time()
file_path = f"main_results//{time_save}"
if not os.path.exists(file_path):
    os.makedirs(file_path)

lpips_model = lpips.LPIPS(net='alex', spatial=True, verbose=False).to(device)

laplacian_kernel = torch.tensor([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

laplacian_kernel = laplacian_kernel.repeat(3, 1, 1, 1)  # [3,1,3,3]

cls_hp = 4e2
perc_hp = 5e1
gate_sparsity_hp = 1e3
large_perturb_hp = 2e1
small_penalty_hp = 1e2
brightness_hp = 1e5
l1_hp = 1e-3
gate_area_hp = 2e-4
orthog_hp = 1e2
high_freq_hp = 1e1
gate_binary_hp = 1e1
l2_hp = 1e-1
smoothness_hp = 0e0
l2_latent_hp = 2e0
div_latent_hp = 2e1
div_img_hp = 5e2

encoder = Encoder(latent_dim=latent_dim, device=device)

model = ZarrSAC(
    policy=CustomSACPolicy,
    env=train_envs,
    buffer_size=50_000, 
    policy_kwargs=dict(
        encoder=encoder,
        latent_dim=latent_dim,
        batch_size=batch_size * num_train_envs,
        device=device
    ),
    verbose=1,
)


def rollout(
    envs: DummyVecEnv, 
    policy: CustomSACPolicy,
    gamma: float,
    max_steps: int = 50, 
): 
    envs.reset()

    total_rewards = np.zeros((envs.num_envs))
    step_num = 0
    curr_gamma = 1

    while True:
        obs_batch = np.stack([env.batch for env in envs.envs]).squeeze(1)  # (num_envs, C, H, W)
        obs_tensor = torch.from_numpy(obs_batch).to(policy.device) 

        with torch.no_grad():
            _, actions, _ = policy.pred_upsampled_action(obs_tensor, deterministic=True)
        
        actions_npy = actions.cpu().numpy()

        perturbed_normalized_clamp, orig_results, perturbed_results = apply_action_no_grad(img_classifier, obs_tensor, actions)

        for i, env in enumerate(envs.envs):
            if step_num == 0: 
                env.set_results(perturbed_normalized_clamp[i], orig_results[i], perturbed_results[i], False)
            else: 
                env.set_results(perturbed_normalized_clamp[i], orig_results[i], perturbed_results[i], dones[i])

        _, rewards, dones, _ = envs.step(actions_npy)
        total_rewards += curr_gamma * rewards
        curr_gamma *= gamma

        step_num += 1

        if step_num > max_steps or all(dones): 
            break
        
    return total_rewards.mean()

def train_model(
    train_envs: DummyVecEnv,
    test_envs: DummyVecEnv, 
    policy: CustomSACPolicy,
    replay_buffer, 
    total_timesteps: int = 20,
    batch_size: int = 32,
    gradient_update_freq: int = 1,
    gamma: float = 0.95,
    tau: float = 0.005, 
    test_freq: float = 5
):
    train_envs.reset() 
    avg_rewards = []
    max_rollout_found = -np.inf
    actor_losses = []
    critic_losses = []
    alpha_losses = []
    classification_losses = []
    upsampled_l1_norms = []
    # upsampled_l2_norms = []
    # delta_l2_norms = []
    # smoothness_vals = []
    perceptual_losses = []
    # latent_diversity_losses = []
    # upsampled_diversity_losses = []
    brightness_losses = []
    # gate_binary_losses = []
    gate_sparsity_losses = []
    large_perturb_losses = []
    small_penalty_losses = []
    gate_area_losses = []
    orthogonality_losses = []
    high_freq_losses = []

    for i in trange(total_timesteps):
        obs_batch = np.stack([env.batch for env in train_envs.envs]).squeeze(1)  # (num_envs, C, H, W)
        obs_tensor = torch.from_numpy(obs_batch).to(policy.device) 

        with torch.no_grad():
            latent_deltas, actions, _ = policy.pred_upsampled_action(obs_tensor, deterministic=False)
        
        latent_deltas = latent_deltas.cpu().numpy()
        actions_npy = actions.cpu().numpy()

        perturbed_normalized_clamp, orig_results, perturbed_results = apply_action_no_grad(img_classifier, obs_tensor, actions)

        for j, env in enumerate(train_envs.envs):
            env.set_results(perturbed_normalized_clamp[j], orig_results[j], perturbed_results[j])

        next_obs, rewards, dones, _ = train_envs.step(actions_npy)

        replay_buffer.add(obs_batch, next_obs, latent_deltas, rewards, dones)

        for idx, done in enumerate(dones):
            if done.item():
                train_envs.env_method("reset", indices=idx)

        if replay_buffer.length() >= 10_000:
            for _ in range(gradient_update_freq):
                batch = replay_buffer.sample(batch_size)

                policy.critic.optimizer.zero_grad()
                policy.actor.optimizer.zero_grad()
                policy.alpha_optimizer.zero_grad()

                with torch.no_grad():
                    latent_obs = policy.feature_encoder(batch.observations)
                    latent_next_obs = policy.feature_encoder(batch.next_observations)

                # downsampled_obs = policy.actor.downsampled_enc(latent_obs)
                downsampled_next_obs = policy.actor.downsampled_enc(latent_next_obs)

                with torch.no_grad():
                    next_action_deltas, next_log_probs = policy.predict_action_with_prob(latent_next_obs, deterministic=False)
                    q_next_a, q_next_b = policy.critic_target(torch.cat([downsampled_next_obs, next_action_deltas], dim=1))
                    q_next = torch.min(q_next_a, q_next_b)

                    # TODO check if I need to detach alpha here, since in torch.no_grad()
                    target_q = batch.rewards.unsqueeze(1) + gamma * (1 - batch.dones.unsqueeze(1)) * (q_next - policy.alpha.detach() * next_log_probs)

                # Actor update
                downsampled_obs, new_actions_upsampled, gate_mask, new_action_deltas, log_probs = policy.predict_action_with_prob_upsampling(latent_obs, deterministic=False)

                # Assume the prediction from the classifier on the original image is the true result, even if that's not true
                orig_results, perturbed_results, orig_denormalized, perturbed_denormalized = apply_action_grad(img_classifier, batch.observations, new_actions_upsampled)
                
                # Detached tensor with the probability of the true classification, batched
                formatted_probs = results_to_tensor(orig_results, perturbed_results)
                # Formatted_probs but connected to computation graph for perturbed_denormalized
                perturbed_outputs_grads = ResultConnector.apply(perturbed_denormalized, formatted_probs)


                # In [0, 1]
                classification_loss = perturbed_outputs_grads.mean()
                # classification_loss = perturbed_results.gather(1, batched_true_classes.view(-1, 1)).mean()

                q_new_action_a, q_new_action_b = policy.critic(torch.cat([downsampled_obs, new_action_deltas], dim=1))
                # Note: these norms are not on [0, 1] images
                # l2_norm_loss = torch.norm(new_actions_upsampled, p=2, dim=(1, 2, 3)).mean()
                # smoothness_loss = torch.abs(torch.diff(new_actions_upsampled, dim=2)).mean() + \
                #                   torch.abs(torch.diff(new_actions_upsampled, dim=1)).mean() 
                l1_norm_loss = torch.norm(new_actions_upsampled, p=1, dim=(1, 2, 3)).mean()
                # l2_norm_loss_deltas = torch.norm(new_action_deltas, p=2, dim=1).mean()

                # Harder selection
                # gate_binary_loss = torch.mean(gate_mask * (1 - gate_mask)) # not needed with differentiable top-k
                gate_sparsity_loss = torch.mean(gate_mask) # L1

                change_magnitude = torch.abs(new_actions_upsampled)  # [B, 3, H, W]
                # soft_topk_mask = differentiable_topk_mask(change_magnitude, k_percent=0.01, temperature=0.01)

                # above_thresh = (change_magnitude > 0.3).float()

                large_perturb_loss = -torch.mean(change_magnitude * gate_mask)  # encourage top k pixels
                small_penalty_loss = torch.mean(change_magnitude * (1 - gate_mask)) # discourage non top k pixels

                orig_gated_rescaled = 2 * (orig_denormalized * gate_mask) - 1
                perturbed_gated_rescaled = 2 * (perturbed_denormalized * gate_mask) - 1

                lpips_map = lpips_model.forward(orig_gated_rescaled, perturbed_gated_rescaled)
                perceptual_weight = 1.0 - change_magnitude
                perceptual_loss = (lpips_map * perceptual_weight).mean()

                # deltas_flat = new_action_deltas.view(batch_size, -1)
                # delta_centered = deltas_flat - deltas_flat.mean(dim=0, keepdim=True)
                # deltas_norm = F.normalize(delta_centered, p=2, dim=1)  # [B, N]
                # delta_cos_sims = torch.matmul(deltas_norm, deltas_norm.T)  # [B, B]
                # latent_diversity_loss = delta_cos_sims[~torch.eye(batch_size, dtype=torch.bool, device=new_action_deltas.device)].mean()

                # flat_upsampled = new_actions_upsampled.view(batch_size, -1)
                # normed_upsampled = F.normalize(flat_upsampled, p=2, dim=1)
                # cos_sim_upsampled = (normed_upsampled @ normed_upsampled.T)
                # decoder_diversity_loss = cos_sim_upsampled[~torch.eye(batch_size, dtype=torch.bool, device=new_actions_upsampled.device)].mean()

                brightness_loss = F.mse_loss(brightness(orig_denormalized), brightness(perturbed_denormalized))

                target_area = 0.01 * 224 * 224
                area = gate_mask.sum(dim=(1, 2, 3))  # per-sample
                gate_area_loss = ((area - target_area) ** 2).mean()

                input_flat = orig_denormalized.view(batch_size, -1)
                perturb_flat = perturbed_denormalized.view(batch_size, -1)

                # TODO try entropy based regularization/consistency after augmentations (self-supervised)/discriminator
                # epsilon = 1e-6  # small value for numerical stability
                # entropy_loss = -torch.sum(gate_mask * torch.log(gate_mask + epsilon), dim=(2,3))

                orthogonality_loss = F.cosine_similarity(perturb_flat, input_flat, dim=1).mean()

                high_freq = F.conv2d(perturbed_denormalized, laplacian_kernel, padding=1, groups=3)
                high_freq_loss = -high_freq.abs().mean()

                # L2, smoothness, L1, classification weights
                # 1e-2, 1e-3, 1e-2 for Learned_main_1745897869.9799478.zip
                # 1e-1, 1e-3, 5e-4 for Learned_main_1745940359.0014925.zip
                # 1e-2, 0, 1e-4 for Learned_main_1745976187.284214.zip, with smooth top-k k=50, temp=0.2
                # 1e-2, 0, 1e-4 for Learned_main_1745978720.9735777.zip, with smooth top-k k=50000, temp=0.75
                # 1e-2, 1e-5, 1e-5 for Learned_main_1746032822.896086.zip, with smooth top-k k=50000, temp=0.9
                # 1e-2, 0, 1e-5, 100 for Learned_main_1746154653.8339.zip: too much noise but perfect classifications
                # 1e-2, 1e-5, 1e-2, 200 for Learned_main_1746198230.5772636.zip: too much noise but perfect classifications
                # 1e-2, 1e-4, 5e-2, 300 for Learned_main_1746245278.3612838.zip: invisible noise for each step but worse classifications
                # 1e-2, 1e-4, 3e-2, 500 for Learned_main_1746248347.6195297.zip: semi visible noise for each step, not much worse classifications
                # 1e-2, 1e-4, 4e-2, 1000 for Learned_main_1746288507.925695.zip: semi visible noise for each step, decent classifications
                # 1e-2, 1e-0, 1e-5, 100 for Learned_main_1747093838.8063166.zip: looks like shader, decent classifications, good numerical results, loss increases (good b/c learning something)
                # 1e-2, 1e-0, 2e-5, 200 for Learned_main_1747102585.3809047.zip: looks like shader, decent classifications, good numerical results, loss increases (good b/c learning something)
                # Last 2 both give ~ same results with eval (~37k L1, ~97 L2)
                # Still overfitting on reconstruction loss b/c learning simple shading


                # Learned_main_1747159734.155619.zip
                # Learned_main_1747171764.2120852.zip
                # Learned_main_1747179504.425404.zip
                # Learned_main_1747224190.7167678.zip: 5e1 class, 1e-1 l2 upsampled, 2e-3 l1_norm_upsampled, 2e0 * l2_latent, 5e0 perceptual, 2e1 latent_diversity, 5e2 decoder_diversity, 1e4 brightness

                # Learned_main_1747232618.8114283.zip: 5e0 class, 5e0 perceptual, 1e0 gate binary, 1e0 gate sparsity, 1e0 large peturb, 1e0 small penalty
                # Learned_main_1747240757.9703887.zip: 5e0 class, 5e0 perceptual, 1e1 gate binary, 2e1 gate sparsity, 5e1 large peturb, 1e1 small penalty
                # Learned_main_1747255898.576219.zip: 5e0 class, 5e0 perceptual, 1e1 gate binary, 2e1 gate sparsity, 5e1 large peturb, 1e1 small penalty, 1e3 brightness
                # Learned_main_1747521000.1190495.zip: 2e-3 l1 action, 1e2 class, 5e1 perceptual, 1e2 gate sparsity (L1), 2e1 large peturb, 1e2 small penalty, 1e4 brightness, saliency w/o normalization
                # Learned_main_1747535493.1333928.zip: 1e-3 l1 action, 1e2 class, 5e1 perceptual, 1e3 gate sparsity (L1), 2e1 large peturb, 1e2 small penalty, 1e4 brightness, 1e-1 gate area saliency w normalization
                # 67315.75259115885 L1, 173.98226381467654 L2, 31.434739941118742 steps
                # TODO consider using variance regularization to bound sampled action/perceptual loss (mse on latent embeddings)
                actor_loss = (policy.alpha.detach() * log_probs - torch.min(q_new_action_a, q_new_action_b)).mean() + \
                             cls_hp * classification_loss + \
                             perc_hp * perceptual_loss + \
                             gate_sparsity_hp * gate_sparsity_loss + \
                             large_perturb_hp * large_perturb_loss + \
                             small_penalty_hp * small_penalty_loss + \
                             brightness_hp * brightness_loss + \
                             l1_hp * l1_norm_loss + \
                             gate_area_hp * gate_area_loss + \
                             orthog_hp * orthogonality_loss + \
                             high_freq_hp * high_freq_loss
                            #  gate_binary_hp * gate_binary_loss + \
                            #  l2_hp * l2_norm_loss + \
                            #  smoothness_hp * smoothness_loss + \
                            #  l2_latent_hp * l2_norm_loss_deltas + \
                            #  div_latent_hp * latent_diversity_loss + \
                            #  div_img_hp * decoder_diversity_loss + \
                actor_losses.append(actor_loss.item())
                classification_losses.append(classification_loss.item())
                # upsampled_l2_norms.append(l2_norm_loss.item())
                upsampled_l1_norms.append(l1_norm_loss.item())
                # smoothness_vals.append(smoothness_loss.item())
                # delta_l2_norms.append(l2_norm_loss_deltas.item())
                perceptual_losses.append(perceptual_loss.item())
                # latent_diversity_losses.append(latent_diversity_loss.item())
                # upsampled_diversity_losses.append(decoder_diversity_loss.item())
                brightness_losses.append(brightness_loss.item())
                # gate_binary_losses.append(gate_binary_loss.item())
                gate_sparsity_losses.append(gate_sparsity_loss.item())
                large_perturb_losses.append(large_perturb_loss.item())
                small_penalty_losses.append(small_penalty_loss.item())
                gate_area_losses.append(gate_area_loss.item())
                orthogonality_losses.append(orthogonality_loss.item())
                high_freq_losses.append(high_freq_loss.item())

                # if i == 9: 
                #     display_comp_graph(classification_loss, "perturbed_probs_comp_graph")
                
                actor_loss = actor_loss.to(policy.device)

                # Critic update
                current_q_a, current_q_b = policy.critic(torch.cat([downsampled_obs, batch.actions], dim=1))
                critic_loss = F.mse_loss(current_q_a, target_q) + F.mse_loss(current_q_b, target_q)
                critic_losses.append(critic_loss.item())

                critic_loss = critic_loss.to(policy.device)

                # Alpha update
                alpha_loss = -(policy.log_alpha * (log_probs.detach() + policy.target_entropy)).mean()
                alpha_losses.append(alpha_loss.item())

                alpha_loss = alpha_loss.to(policy.device)

                critic_loss.backward(retain_graph=True)
                actor_loss.backward()
                alpha_loss.backward()

                # if i % 40 == 0: 
                #     print(f"Mem summary at iter {i}: \n{torch.cuda.memory_summary()}\n\n")

                policy.critic.optimizer.step()
                policy.actor.optimizer.step()
                policy.alpha_optimizer.step()

                # Soft update target
                polyak_update(policy.critic.parameters(), policy.critic_target.parameters(), tau)

        if i % test_freq == 0 and replay_buffer.length() >= 10_000: 
            rollout_val = rollout(test_envs, policy, gamma)
            if rollout_val > max_rollout_found: 
                model.save(f"main_results//{time_save}//middle.zip")
                max_rollout_found = rollout_val
            avg_rewards.append(rollout_val)

    if args.save_lc: 
        plot_learning_curve(avg_rewards, test_freq, f"main_results//{time_save}//LC.png")
        plot_per_step(actor_losses, 1, f"main_results//{time_save}//actor.png", "Actor Loss")
        plot_per_step(critic_losses, 1, f"main_results//{time_save}//critic.png", "Critic Loss")
        plot_per_step(alpha_losses, 1, f"main_results//{time_save}//alpha.png", "Alpha Loss")
        plot_per_step(upsampled_l1_norms, 1, f"main_results//{time_save}//l1.png", f"Upsampled L1 Loss, Weight {l1_hp}")
        # plot_per_step(upsampled_l2_norms, 1, f"main_results//{time_save}//l2.png", f"Upsampled L2 Loss, Weight {l2_hp}")
        # plot_per_step(smoothness_vals, 1, f"main_results//{time_save}//smoothness.png", f"Smoothness Loss, Weight {smoothness_hp}")
        plot_per_step(classification_losses, 1, f"main_results//{time_save}//cls.png", f"Classification Loss, Weight {cls_hp}")
        # plot_per_step(delta_l2_norms, 1, f"main_results//{time_save}//latent_l2.png", f"Latent Action L2 Loss, Weight {l2_latent_hp}")
        plot_per_step(perceptual_losses, 1, f"main_results//{time_save}//perceptual.png", f"LPIPS Perceptual Loss, Weight {perc_hp}")
        # plot_per_step(latent_diversity_losses, 1, f"main_results//{time_save}//latent_diversity.png", f"Latent Diversity Loss, Weight {div_latent_hp}")
        # plot_per_step(upsampled_diversity_losses, 1, f"main_results//{time_save}//upsampled_diversity.png", f"Decoder Diversity Loss, Weight {div_img_hp}")
        plot_per_step(brightness_losses, 1, f"main_results//{time_save}//brightness.png", f"Brightness Loss, Weight {brightness_hp}")
        # plot_per_step(gate_binary_losses, 1, f"main_results//{time_save}//gate_binary.png", f"Gate Binary Loss, Weight {gate_binary_hp}")
        plot_per_step(gate_sparsity_losses, 1, f"main_results//{time_save}//gate_sparsity.png", f"Gate Sparsity Loss, Weight {gate_sparsity_hp}")
        plot_per_step(large_perturb_losses, 1, f"main_results//{time_save}//large_perturbs.png", f"Large Perturbation Loss, Weight {large_perturb_hp}")
        plot_per_step(small_penalty_losses, 1, f"main_results//{time_save}//small_penalty.png", f"Small Penalty Loss, Weight {small_penalty_hp}")
        plot_per_step(gate_area_losses, 1, f"main_results//{time_save}//gate_area.png", f"Gate Area Loss, Weight {gate_area_hp}")
        plot_per_step(orthogonality_losses, 1, f"main_results//{time_save}//orthog.png", f"Orthogonality Loss, Weight {orthog_hp}")
        plot_per_step(high_freq_losses, 1, f"main_results//{time_save}//high_freq.png", f"High Frequency Loss, Weight {high_freq_hp}")


train_model(model.env, eval_envs, model.policy, model.replay_buffer, total_timesteps=num_timesteps, batch_size=training_batch_size, gradient_update_freq=gradient_update_freq, gamma=gamma, test_freq=test_freq)
model.save(f"main_results//{time_save}//end.zip")

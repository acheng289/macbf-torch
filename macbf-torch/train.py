import core
import config
import scene
from scene import Scene, SceneDataset
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from tqdm import tqdm
from datetime import datetime


# Define training directories containing scene csvs
training_scenes = [
    "../data/three_agents_obs_track_easy_forest_0",
    "../data/three_agents_obs_track_easy_forest_1",
    "../data/three_agents_obs_track_easy_forest_2",
    "../data/three_agents_obs_track_easy_forest_five_random",
    "../data/three_agents_obs_track_easy_forest_five_random_small",
    "../data/three_agents_obs_track_easy_forest_three_random_small",
]

dataset = SceneDataset(training_scenes)

# Instantiate networks
action_net = core.NetworkAction()
cbf_net = core.NetworkCBF()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
action_net.to(device)
cbf_net.to(device)

dataloader_num_workers = 0 # Start with 0, increase if I/O is slow
if device == torch.device("cuda"):
    dataloader_num_workers = min(os.cpu_count() // 2, 4) # Heuristic for num_workers
                       
train_loader = DataLoader(
    dataset=dataset,
    batch_size=1, # Use batch size one so that we train on one scene at a time
    shuffle=True,
    num_workers=dataloader_num_workers,
    pin_memory=True if device == torch.device("cuda") else False
)



# Optimizers for Action and CBF networks as per original MACBF
optimizer_h = optim.Adam(cbf_net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
optimizer_a = optim.Adam(action_net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)


# Logging and Metrics
loss_lists = [] # Accumulates component losses over batches for display
dist_errors_np = [] # Dist errors for the learned controller
safety_ratios_epoch = [] # Safety ratios for learned controller

# Create directory to save models
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
run_save_dir = os.path.join('../models', f"{current_time_str}_run")
os.makedirs(run_save_dir, exist_ok=True)
print(f"Models for this run will be saved in: {run_save_dir}")

print("Starting training...")
start_time_overall = time.time()

# Main training loop
for train_step in range(config.TRAIN_STEPS):
    print(f"--- Train Step: {train_step + 1}/{config.TRAIN_STEPS} ---")
    # Grab scene from dataloader
    for batch_idx, (current_scene_name, current_scene) in enumerate(train_loader):
        print(f"  Processing Scene: {current_scene_name[0]} {batch_idx + 1}/{len(train_loader)} in Train Step {train_step + 1}") # Added print
        # Remove batch dimension and put on device
        current_scene = current_scene.squeeze(0) # (TxNxS)
        # Get timesteps
        t_scene = current_scene.shape[0]
        current_scene = current_scene.to(device)

        # Extract start state
        current_state = current_scene[0, :, :].clone() # (NxS)

        # Zero gradients
        optimizer_h.zero_grad()
        optimizer_a.zero_grad()

        # Calculate total accumulation steps for averaging of gradients for the scene
        total_accumulation_steps_for_scene = (t_scene - 1) * config.INNER_LOOPS

        # Training over timesteps
        for time_step in range(1, t_scene):
            # Get reference state
            reference_state = current_scene[time_step, :, :] # (NxS)

            # Accumulation steps where the system is ran
            acc_steps_pbar_desc = f"Scene {batch_idx+1}, TS {time_step}/{t_scene-1}"
            for acc_step in tqdm(range(config.INNER_LOOPS), desc=acc_steps_pbar_desc, leave=False):
                # Optimizations to minimize computation graph
                # Detach current state to avoid backprop through the entire trajectory
                current_state_for_loss = current_state.detach().clone().requires_grad_(True)
                reference_state_for_loss = reference_state

                # Usual forward pass
                current_state_diff_for_loss = current_state_for_loss.unsqueeze(1) - current_state_for_loss.unsqueeze(0)
                h_for_loss, mask_for_loss = cbf_net(current_state_diff_for_loss)
                u_out_for_loss = action_net(current_state_for_loss, reference_state_for_loss)

                # Compute losses
                loss_dang_val, loss_safe_val = core.loss_barrier_pytorch(h_for_loss, current_state_for_loss)
                loss_dang_deriv_val, loss_safe_deriv_val, loss_medium_deriv_val = core.loss_derivatives_pytorch(current_state_for_loss, u_out_for_loss, h_for_loss, cbf_net)
                loss_action_val = core.loss_actions_pytorch(current_state_for_loss, u_out_for_loss, reference_state_for_loss)

                # Get barrier function values
                # current_state_diff = current_state.unsqueeze(1) - current_state.unsqueeze(0)
                # h, mask = cbf_net(current_state_diff)

                # # Get control input
                # u_out = action_net(current_state, reference_state)

                # # Compute losses
                # loss_dang_val, loss_safe_val = core.loss_barrier_pytorch(h, current_state)
                # loss_dang_deriv_val, loss_safe_deriv_val, loss_medium_deriv_val = core.loss_derivatives_pytorch(current_state, u_out, h, cbf_net)
                # loss_action_val = core.loss_actions_pytorch(current_state, u_out, reference_state)

                loss_list_current_step = [
                    loss_dang_val, loss_safe_val, 3 * loss_dang_deriv_val,
                    loss_safe_deriv_val, 2 * loss_medium_deriv_val, 0.5 * loss_action_val
                ]

                step_loss = sum(loss_list_current_step)

                # In macbf TF, they explicitly build the weight decay loss, but this is handled implicitly by optimizers
                total_loss_current_step = 10 * step_loss

                # Appending losses for logging
                loss_lists.append([l.item() for l in loss_list_current_step])

                # Accumulate gradients
                # total_loss_current_step.backward()
                # Scale loss by the total number of accumulation steps in the scene
                averaged_loss_for_backward = total_loss_current_step / total_accumulation_steps_for_scene
                averaged_loss_for_backward.backward()

                # Simulate movement
                dsdt = u_out_for_loss
                current_state = current_state + dsdt * config.TIME_STEP

                # Logging safety ratios
                with torch.no_grad():
                    current_state_cpu = current_state.cpu()
                    dang_mask = core.compute_dangerous_mask_pytorch(current_state_cpu, config.DIST_MIN_THRES)
                    dang_mask_np = dang_mask.numpy()
                    safety_ratio_step = 1 - np.mean(dang_mask_np, axis=1)
                    safety_ratio_step = np.mean(safety_ratio_step == 1)
                    safety_ratios_epoch.append(safety_ratio_step)

                # Optimized Safety Ratio Calculation (on GPU)
                # with torch.no_grad():
                #     # current_state is on GPU
                #     dang_mask = core.compute_dangerous_mask_pytorch(current_state, config.DIST_MIN_THRES) # GPU computation
                #     num_agents = current_state.shape[0]
                #     if num_agents > 1:
                #         # compute_dangerous_mask_pytorch returns True if agent i is in danger with agent j,
                #         # and its diagonal is False (no self-danger).
                #         per_agent_danger_from_others = torch.sum(dang_mask, dim=1) # Sum over columns for each row agent
                #         is_agent_safe = (per_agent_danger_from_others == 0).float() # Agent is safe if sum is 0
                #         safety_ratio_step = torch.mean(is_agent_safe).item() # .item() moves to CPU
                #     else:
                #         safety_ratio_step = 1.0
                #     safety_ratios_epoch.append(safety_ratio_step)
        # Performing optimization step, alternating between CBF and Action
        if np.mod(train_step // 10, 2) == 0:
            optimizer_h.step()
            print(f"  Optimized CBF Network at step {train_step}")
        else:
            optimizer_a.step()
            print(f"  Optimized Action Network at step {train_step}")
        
        # Calculate final distance error
        with torch.no_grad():
            final_distance_error = np.mean(
                np.linalg.norm(
                    current_state.cpu().numpy()[:, :3] - reference_state.cpu().numpy()[:, :3]
                )
            )
            dist_errors_np.append(final_distance_error)
        
        if np.mod(train_step, config.DISPLAY_STEPS) == 0:
            print('Step: {}, Time: {:.1f}, Loss: {}, Dist: {:.3f}, Safety Rate: {:.3f}'.format(
                    train_step, time.time() - start_time_overall, np.mean(loss_lists, axis=0),
                    np.mean(dist_errors_np), np.mean(safety_ratios_epoch)))
            start_time_overall = time.time()
            (loss_lists, dist_errors_np, safety_ratios_epoch) = [], [], []
        
        if np.mod(train_step, config.SAVE_STEPS) == 0 or train_step + 1 == config.TRAIN_STEPS:
            # Save model state dicts
            model_save_path = os.path.join(run_save_dir, f'model_iter_{train_step}.pth')
            torch.save({
                'epoch': train_step,
                'action_net_state_dict': action_net.state_dict(),
                'cbf_net_state_dict': cbf_net.state_dict(),
                'optimizer_h_state_dict': optimizer_h.state_dict(),
                'optimizer_a_state_dict': optimizer_a.state_dict(),
            }, model_save_path)
            print(f"Model saved at iteration {train_step}")
            print(f"Saved model state dicts to: {model_save_path}")

print("Training complete!")

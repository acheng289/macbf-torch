import torch
import numpy as np
import os
import argparse
from tqdm import tqdm

import core
import config
from scene import Scene, SceneDataset # Assuming Scene can now give its name
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained MACBF-Torch model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file.')
    parser.add_argument('--eval_dirs', nargs='+', type=str, required=True, help='Directory containing evaluation scenes.')
    parser.add_argument('--output_dir', type=str, default='evaluated_scenes', help='Base directory to save the output scenes.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use (e.g., "0", "1"). Use "-1" for CPU.')
    parser.add_argument('--show_plots', action='store_true', help='Show trajectory plots after generating each scene.')
    parser.add_argument('--save_plots', action='store_true', help='Save trajectory plots.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Setup device
    if args.gpu == "-1" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")
    print(f"Using device: {device}")

    # Instantiate networks
    action_net = core.NetworkAction()
    # cbf_net = core.NetworkCBF() # CBF net might not be directly used

    # Load checkpoint
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist.")
        return
    
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    action_net.load_state_dict(checkpoint['action_net_state_dict'])
    # cbf_net.load_state_dict(checkpoint['cbf_net_state_dict'])

    action_net.to(device)
    # cbf_net.to(device)

    action_net.eval()
    # cbf_net.eval()

    # --- Process a single evaluation directory ---
    # if not os.path.isdir(args.eval_dir):
    #     print(f"Error: Evaluation directory {args.eval_dir} does not exist or is not a directory.")
    #     return

    # Create a Scene object directly from list of eval_dirs
    try:
        # SceneDataset expects a list of paths
        eval_dataset = SceneDataset(args.eval_dirs)
        if len(eval_dataset) == 0:
            print(f"No valid scene data loaded from {args.eval_dirs}. Exiting.")
            return
    except RuntimeError as e:
        print(f"Error initializing SceneDataset for {args.eval_dirs}: {e}")
        return

    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=0
    )
    
    overall_avg_safety_rates = [] 
    overall_final_distances_to_ref = []

    # For now, I've set up the script to just run once.
    for scene_idx, (scene_name, original_scene_data_batch) in enumerate(eval_loader):
        original_scene_data = original_scene_data_batch.squeeze(0).to(device)
        num_timesteps_original = original_scene_data.shape[0]
        num_agents_original = original_scene_data.shape[1]

        print(f"\n--- Evaluating Scene from {args.eval_dirs} Scene Name: {scene_name[0]} (Original T={num_timesteps_original}, N={num_agents_original}) ---")

        # Create a subdirectory within args.output_dir for this specific evaluation run
        current_eval_output_base_dir = os.path.join(args.output_dir, f"{scene_name[0]}_evaluated")
        os.makedirs(current_eval_output_base_dir, exist_ok=True)
        print(f"Output for {scene_name[0]} will be saved in: {current_eval_output_base_dir}")

        current_state_eval = original_scene_data[0, :, :].clone()
        simulated_trajectory_list = [current_state_eval.clone().cpu().numpy()]
        scene_safety_ratios = []

        for t_ref_idx in tqdm(range(1, num_timesteps_original), desc="  Simulating to reference waypoints"):
            reference_state_eval = original_scene_data[t_ref_idx, :, :].clone()
            for _ in range(config.INNER_LOOPS_EVAL):
                with torch.no_grad():
                    action = action_net(current_state_eval, reference_state_eval)

                # Simulate movement and append next state to trajectory list 
                dsdt = action
                current_state_eval = current_state_eval + dsdt * config.TIME_STEP_EVAL
                simulated_trajectory_list.append(current_state_eval.clone().cpu().numpy())

                with torch.no_grad():
                    dang_mask = core.compute_dangerous_mask_pytorch(current_state_eval, config.DIST_MIN_CHECK)
                    if num_agents_original > 1:
                        per_agent_danger = torch.sum(dang_mask, dim=1)
                        is_agent_safe = (per_agent_danger == 0).float()
                        safety_ratio_step = torch.mean(is_agent_safe).item()
                    else:
                        safety_ratio_step = 1.0
                    scene_safety_ratios.append(safety_ratio_step)

                with torch.no_grad():
                    dist_to_ref = torch.linalg.norm(current_state_eval[:, :3] - reference_state_eval[:, :3], dim=1)
                    # Move to next reference state if all agents are close enough
                    if torch.mean(dist_to_ref) < config.DIST_TOLERATE:
                        break
        
        simulated_states_np = np.array(simulated_trajectory_list)
        print(f"  Generated simulated trajectory of shape: {simulated_states_np.shape}")

        if scene_safety_ratios:
            avg_scene_safety = np.mean(scene_safety_ratios)
            overall_avg_safety_rates.append(avg_scene_safety) # For this one scene
            print(f"  Average Safety Rate for this scene (simulated): {avg_scene_safety:.3f}")
        
        with torch.no_grad():
            final_ref_state = original_scene_data[-1, :, :].clone()
            final_sim_state = torch.tensor(simulated_states_np[-1], device=device, dtype=torch.float32)
            final_dist = torch.mean(torch.linalg.norm(final_sim_state[:,:3] - final_ref_state[:,:3], dim=1)).item()
            overall_final_distances_to_ref.append(final_dist) 
            print(f"  Distance to final reference state: {final_dist:.3f}")

        # Create a new Scene object from the simulated trajectory
        output_scene_obj = Scene(states=simulated_states_np)

        # Save the new scene as CSVs into the specific subdirectory
        output_scene_obj.save_scene_as_csvs(output_dir=current_eval_output_base_dir)

        if args.save_plots or args.show_plots:
            plot_filename = None
            if args.save_plots:
                plot_filename = os.path.join(current_eval_output_base_dir, f"simulated_trajectory_plot.png")
            
            print(f"  Visualizing simulated scene from {scene_name[0]}...")
            output_scene_obj.visualize_scene(output_path=plot_filename, show_plot=args.show_plots)


    print("\n--- Evaluation Summary (for this run) ---")
    if overall_avg_safety_rates: 
        print(f"Overall Average Safety Rate: {np.mean(overall_avg_safety_rates):.3f}")
    if overall_final_distances_to_ref: 
        print(f"Overall Average Distance to Final Reference: {np.mean(overall_final_distances_to_ref):.3f}")
    print("Evaluation complete for this directory.")

if __name__ == '__main__':
    main()

# python3 eval.py --model_path ../models/20250603_152049_run/model_iter_0.pth \ 
    # --eval_dir ../data/three_agents_obs_track_easy_forest_0/ \ 
    # --output_dir ../eval \ 
    # --show_plots --save_plots
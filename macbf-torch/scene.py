import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
"""
This file contains definitions of the Scene and SceneDataset classes which will be useful during training and evaluation
"""

class Scene:
    """
    Class that describes a scene, which can be further broken down into the states, or flight paths taken by an arbitrary number of agents. Takes in the scene directory which
    contains csvs of the flight paths of multiple agents and transforms them into numpy arrays that can be manipulated during training or evaluation. Supports visualization, saving
    of visualization, creating of new scenes and saving as csvs.
    """
    def __init__(self, scene_path=None, states=None, position_threshold=1e-8):
        self.scene_path = scene_path
        self.states = states
        self.position_threshold = position_threshold
        self.sorted_agent_ids = None
        self.agent_paths = {}
        self.scene_name = None

        if self.states is not None:
            print(f"States were provided, initializing scene from provided states of shape {self.states.shape}")
        else:
            print(f"Initializing scene using states in {self.scene_path}")
            if self.scene_path is not None:
                if not os.path.isdir(scene_path):
                    raise FileNotFoundError(f"Scene path {scene_path} does not exist!")

                self.scene_name = os.path.basename(os.path.normpath(self.scene_path))

                for file in os.listdir(scene_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(scene_path, file)
                        print(f"Found {file_path}, processing...")

                        # Get agent ID by removing file extension
                        agent_id = os.path.splitext(file)[0]
                        self._process_agent_csv(file_path, agent_id)

                # Make sure all paths are of the same length
                self._pad_agent_paths()
            else:
                raise ValueError("Provide either a 'scene_path' or 'states' for Scene initialization")
    
    def _process_agent_csv(self, file_path, agent_id):
        """
        Read agent csv, remove hovering states and stores it.
        """
        df = pd.read_csv(file_path)

        # Check that expected columns exist
        expected_columns = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'q_x', 'q_y', 'q_z', 'q_w']
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"CSV file {file_path} is missing expected columns! Check that states are 10")
        
        processed_states = []

        data = df[expected_columns].values

        if len(data) == 0:
            print(f"Warning: {file_path} is empty. Skipping...")
            self.agent_paths[agent_id] = np.array([])
            return
        
        processed_states.append(data[0])

        for i in range (1, len(data)):
            # Extract and compare positional data. If the ith + 1 state and ith state are similar we remove the ith state
            prev_pos = data[i-1, :3]
            current_pos = data[i, :3]

            distance = np.linalg.norm(current_pos - prev_pos)

            if distance > self.position_threshold:
                processed_states.append(data[i])
        
        self.agent_paths[agent_id] = np.array(processed_states)
        print(f"Processed {agent_id}, kept {len(self.agent_paths[agent_id])} states after removing hover states from original {len(data)}")

    def _pad_agent_paths(self):
        """
        Pads individual agent paths to maximum length and combines them to yield a TxNxS array
        T is timestep, N is the number of agents and S is the state space
        """
        if not self.agent_paths:
            self.states = np.array([])
            print("No agent paths to pad.")
            return

        max_timesteps = 0
        for agent_id, path in self.agent_paths.items():
            if len(path) > max_timesteps:
                max_timesteps = len(path)

        if max_timesteps == 0:
            self.states = np.array([])
            print("All agent paths are empty. No states to form.")
            return
        
        num_agents = len(self.agent_paths)
        state_dim = 10

        # Initialize states array with zeroes
        self.states = np.zeros((max_timesteps, num_agents, state_dim))

        # Ensure consistent agent order, so 0th is NX01 and so on
        sorted_agent_ids = sorted(self.agent_paths.keys())
        self.sorted_agent_ids = sorted_agent_ids
        for i, agent_id in enumerate(sorted_agent_ids):
            path = self.agent_paths[agent_id]

            if len(path) == 0:
                print(f"Warning: Agent {agent_id} has an empty path after processing. It will be padded entirely.")
                # If path is empty, we can just fill with zeros or a neutral state if one exists
                # For now, it will remain zeros from the initialization
                continue

            # Fill in states
            self.states[:len(path), i, :] = path

            # Pad by repeating the last entry
            if len(path) < max_timesteps:
                last_entry = path[-1]
                self.states[len(path):max_timesteps, i, :] = last_entry
                print(f"Padded agent {agent_id} from {len(path)} to {max_timesteps} timesteps.")
        
        print(f"Final scene states shape: {self.states.shape} (TxNxS)")

    def save_scene_as_csvs(self, output_dir="new_scene_data"):
        """
        Saves the processed and padded scene data back into individual CSV files
        in a new directory, mimicking the original input format.
        """
        if self.states is None or self.states.size == 0:
            print("No states to save. Scene is empty or not initialized.")
            return

        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving processed scene to {output_dir}")

        T, N, S = self.states.shape
        headers = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'q_x', 'q_y', 'q_z', 'q_w']
        if self.sorted_agent_ids is not None:
            for i, agent_id in enumerate(self.sorted_agent_ids):
                agent_data = self.states[:, i, :]
                df = pd.DataFrame(agent_data, columns=headers)
                # Save as "NX01.csv" and so on
                filename = f"{agent_id}.csv"
                output_file_path = os.path.join(output_dir, filename)
                df.to_csv(output_file_path, index=False)
                print(f"Saved {output_file_path}")
        else: 
            for i in range(N):
                agent_data = self.states[:, i, :]
                df = pd.DataFrame(agent_data, columns=headers)
                # Assuming original filenames were like NX01.csv, NX02.csv
                filename = f"NX{(i+1):02d}.csv" 
                output_file_path = os.path.join(output_dir, filename)
                df.to_csv(output_file_path, index=False)
                print(f"Saved {output_file_path}")

    def visualize_scene(self, output_path=None, show_plot=True):
        """
        Visualizes the 3D flight paths of all agents in the scene.
        Optionally saves the plot to a specified output path.
        """
        if self.states is None or self.states.size == 0:
            print("No states to visualize. Scene is empty or not initialized.")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        T, N, S = self.states.shape

        for i in range(N):
            # Extract position data for the i-th agent (first 3 columns of the state vector)
            agent_positions = self.states[:, i, :3]
            ax.plot(agent_positions[:, 0], agent_positions[:, 1], agent_positions[:, 2], label=f'Agent {i+1}')
            ax.scatter(agent_positions[0, 0], agent_positions[0, 1], agent_positions[0, 2], marker='o', s=50, color='green', label=f'Agent {i+1} Start' if i == 0 else "")
            ax.scatter(agent_positions[-1, 0], agent_positions[-1, 1], agent_positions[-1, 2], marker='x', s=50, color='red', label=f'Agent {i+1} End' if i == 0 else "")


        ax.set_xlabel('pos_x')
        ax.set_ylabel('pos_y')
        ax.set_zlabel('pos_z')
        ax.set_title('Multi-Agent Flight Paths')
        ax.legend()
        ax.grid(True)

        if output_path:
            plt.savefig(output_path)
            print(f"Scene visualization saved to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig) # Close the figure if not showing to free up memory
    
    def get_timestamps(self):
        if self.states is not None:
            return self.states.shape[0]
        else:
            raise ValueError(f"States is not initialized yet!")
    
    def get_state(self, timestep):
        if self.states is not None:
            return self.states[timestep]
        else:
            raise ValueError(f"States is not initialized yet!")
    
    def get_states(self):
        if self.states is not None:
            return self.states
        else:
            raise ValueError(f"States is not initialized yet!")
    
    def get_scene_name(self):
        if self.scene_name is not None:
            return self.scene_name
        else:
            raise ValueError(f"Scene name is not initialized yet!")
        
class SceneDataset(Dataset):
    def __init__(self, scene_paths):
        """
        Initialize the dataset by loading and processing multiple scenes.
        Each scene would be 1 training example
        Args:
            scene_paths: A list of directories, each containing CSVs for one scene
        """
        self.scenes = []
        for path in scene_paths:
            # Load each scene using Scene class
            scene = Scene(path)

            # Get full state trajectory (T, N, S)
            full_states_np = scene.get_states()

            # Convert to Tensor
            full_states_tensor = torch.tensor(full_states_np, dtype=torch.float32)

            # Get scene name
            scene_name = scene.get_scene_name()

            self.scenes.append((scene_name, full_states_tensor))
        
        self.num_scenes = len(self.scenes)
    
    # Necessary overriding of Dataset functions
    def __len__(self):
        """
        Return total number of scenes.
        """
        return self.num_scenes
    
    def __getitem__(self, idx):
        """
        Retrieves trajectory data for a single scene. Now also includes scene_name
        """
        return self.scenes[idx]

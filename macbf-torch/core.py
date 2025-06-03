import torch
import torch.nn as nn
import torch.nn.functional as F
import config


def quadrotor_controller_pytorch(s, s_ref):
    """
    Dummy controller. 
    u_ref = controller(s, s_ref)
    dsdt = dynamics(s, u)
    s_next = s + dsdt * TIME
    The sim controller just updates the next state to be the goal state.
    """
    # We try simple subtraction first, concern is quaternion. But if just numbers maybe it works out.
    return (s_ref - s) / config.TIME_STEP

def compute_safe_mask_pytorch(s, r):
    """
    Identify agents outside the safe radius (or self-connection).
    Args:
        s (torch.Tensor): The current state of N agents (N, S).
        r (float): The safe radius.

    Returns:
        torch.Tensor: A boolean mask (N, N) where True indicates
                      either outside the safe radius or a self-connection.
    """
    N, S_dim = s.shape

    # Calculate state difference between agents
    s_diff = s.unsqueeze(1) - s.unsqueeze(0) # (N, N, S)

    # Create an identity matrix (N, N) for self-connection indication.
    eye_indicator = torch.eye(N, device=s.device, dtype=s.dtype).unsqueeze(-1) # (N, N, 1)

    # Concatenate the identity indicator to the last dimension of s_diff.
    s_diff_with_eye = torch.cat([s_diff, eye_indicator], dim=-1)

    # Extract positional difference (first 3 dimensions) and the eye indicator.
    z_diff = s_diff_with_eye[:, :, :3] # Shape: (N, N, 3)
    eye = s_diff_with_eye[:, :, -1:]  # Shape: (N, N, 1)

    # Calculate Euclidean norm (distance) for positional difference.
    dist_3d = torch.linalg.norm(z_diff + 1e-4, dim=-1, keepdim=True) # Shape: (N, N, 1)

    # Logic: Mask is True if (distance > safe_radius) OR (it's a self-connection)
    mask = torch.logical_or(dist_3d > r, eye == 1) # (N, N, 1) boolean

    return mask.squeeze(-1) # Shape: (N, N) boolean

def compute_dangerous_mask_pytorch(s, r):
    """
    Identify agents within the dangerous radius
    Args:
        s (torch.Tensor): The current state of N agents (N, S).
        r (float): The danger radius.

    Returns:
        torch.Tensor: A boolean mask (N, N) where True indicates an agent in the dangerous radius
    """
    N, S_dim = s.shape

    # Calculate state difference between agents
    s_diff = s.unsqueeze(1) - s.unsqueeze(0) # (N, N, S)

    # Create an identity matrix (N, N) for self-connection indication.
    eye_indicator = torch.eye(N, device=s.device, dtype=s.dtype).unsqueeze(-1) # (N, N, 1)

    # Concatenate the identity indicator to the last dimension of s_diff.
    s_diff_with_eye = torch.cat([s_diff, eye_indicator], dim=-1)

    # Extract positional difference (first 3 dimensions) and the eye indicator.
    z_diff = s_diff_with_eye[:, :, :3] # Shape: (N, N, 3)
    eye = s_diff_with_eye[:, :, -1:]  # Shape: (N, N, 1)

    # Calculate Euclidean norm (distance) for positional difference.
    dist_3d = torch.linalg.norm(z_diff + 1e-4, dim=-1, keepdim=True) # Shape: (N, N, 1)

    # Logic: Mask is True if (distance < safe_radius) AND (it's a NOT self-connection)
    mask = torch.logical_and(dist_3d < r, eye == 0) # (N, N, 1) boolean

    return mask.squeeze(-1) # (N, N) boolean

class NetworkAction(nn.Module):
    """
    Controller as a neural network, rewritten in PyTorch. We overlook the top_k variable for now
    since our simulation will be ran on 2 agents.
    Args:
        state_dim (int): Dimension of the state vector (e.g., 10).
        ref_state_dim (int): Dimension of the reference state vector (e.g., 10).
        output_action_dim (int): Dimension of the control action (e.g., 2 or 3). In our case, 10
        obs_radius (float): The observation radius.
    """
    def __init__(self, state_dim=10, ref_state_dim=10, output_action_dim=10, obs_radius=config.OBS_RADIUS):
        super(NetworkAction, self).__init__()
        self.obs_radius = obs_radius
        self.state_dim = state_dim
        self.ref_state_dim = ref_state_dim
        self.output_action_dim = output_action_dim

        # The input channel has an additional identity field
        self.conv1_in_channels = state_dim + 1

        # 1D convolutions
        self.conv1 = nn.Conv1d(in_channels=self.conv1_in_channels, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)

        # Fully connected layers
        self.fc1_in_features = 128 + state_dim
        self.fc1 = nn.Linear(in_features=self.fc1_in_features, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=output_action_dim)
    
    def forward(self, s, s_ref):
        """
        The forward pass of the model. Yields control input u.
        Args:
            s (N, S_dim): The current state of N agents.
            s_ref (N, S_dim): The reference location, velocity and acceleration.
        Returns:
            u (N, output_action_dim): The control action.
        """
        # x: (N, N, S_dim) representing s_i - s_j (state difference across agents)
        #  print(f"Input dimensions: {s.shape}")
        x = s.unsqueeze(1) - s.unsqueeze(0)
        # print(f"x dimensions after subtraction: {x.shape}")
        
        # Add identity matrix as means of self-identification. unsqueeze adds extra dimenion
        identity_matrix = torch.eye(x.shape[0], device=s.device, dtype=s.dtype).unsqueeze(2)
        x = torch.cat([x, identity_matrix], dim=2)
        # print(f"x dimensions after concatenation with identity: {x.shape}")

        # Define observation mask
        dist = torch.linalg.norm(x[:, :, :3], dim=2, keepdim=True)
        mask = (dist < self.obs_radius).float()
        # print(f"mask dimensions: {mask.shape}")

        # Permute because Conv1D expects (batch_size, channels, sequence_length)
        # permute from (N, N, features) to (N, features, N)
        x_conv_in = x.permute(0, 2, 1)
        # print(f"x_conv input shape: {x_conv_in.shape}")

        x = F.relu(self.conv1(x_conv_in)) # x becomes (N, 64, N)
        x = F.relu(self.conv2(x)) # x: (N, 128, N)
        # print(f"x_conv output shape: {x.shape}")

        # Permute the mask to be (N, 1, N) to be multiplied with x
        mask_conv_out = mask.permute(0, 2, 1)

        # Element-wise multiplication
        x_masked = x * mask_conv_out
        # print(f"x shape after masking: {x_masked.shape}")
        
        # Maxpooling along the "K" dimension
        x = torch.max(x_masked, dim=2)[0]
        # Should be (N, 128)
        # print(f"x shape after pooling: {x.shape}")

        # Concatentate with s_ref
        x = torch.cat([x, s - s_ref], dim=1)
        # print(f"x shape after concatenation with state differences: {x.shape}")

        # Put x through fully connected layers
        x = F.relu(self.fc1(x)) # x: (N, 64)
        x = F.relu(self.fc2(x)) # x: (N, 128)
        x = F.relu(self.fc3(x)) # x: (N, 64)
        x = self.fc4(x) # x: (N, output_action_dim) - no activation on final layer
        # print(x)

        # Reference controller action
        # u_ref = controller(s, s_ref)
        # dsdt = dynamics(s, u)
        # s_next = s + dsdt * TIME
        # But for us, we have no dynamics
        u_ref = quadrotor_controller_pytorch(s, s_ref)
        u = x + u_ref
        return u
    
class NetworkCBF(nn.Module):
    """
    Control barrier function as a neural network, written in PyTorch.
    Calculates a scalar CBF value 'h' for each agent with respect to its neighbors
    Similarly, we ignore the notion of nearest agents for now
    Args:
        state_dim (int): Dimension of the state vector (e.g., 10).
        ref_state_dim (int): Dimension of the reference state vector (e.g., 10).
        obs_radius (float): The observation radius.
    """
    def __init__(self, state_diff_dim=10, dangerous_radius=config.DIST_MIN_THRES, obs_radius=config.OBS_RADIUS):
        super(NetworkCBF, self).__init__()

        self.dangerous_radius = dangerous_radius
        self.obs_radius = obs_radius

        # State differences(10), identity(1), signed distance(1)
        self.conv1_in_channels = state_diff_dim + 1 + 1

        # Need to permute input later to work with Conv1D
        self.conv1 = nn.Conv1d(in_channels=self.conv1_in_channels, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x, r=config.DIST_MIN_THRES):
        """
        Args:
            x: (N, N, 10) state difference of N agents
            r: radius of dangerous zone
        Returns:
            h: (N, N, 1) CBF of N agents with neighboring agents
            mask: (N, N, 1) Mask of agents within observation radius
        """
        # Make r a tensor for easier manipulation
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=x.dtype, device=x.device)
        
        # Calculate Euclidean distance
        d_norm = torch.sqrt(torch.sum(torch.square(x[:, :, :3]) + 1e-4, dim=2, keepdim=True)) # (N, N, 1)

        # Create identity matrix
        N_agents = x.shape[0]
        identity_matrix = torch.eye(N_agents, device=x.device, dtype=x.dtype).unsqueeze(2) # (N, N, 1)

        # Signed distance to dangerous zone
        signed_dist_to_r = d_norm - r

        # Concatenate features
        x = torch.cat([x, identity_matrix, signed_dist_to_r], dim=2) # (N, N, 12)

        # Calculate 2D distance for observation mask
        dist_2d = torch.sqrt(torch.sum(torch.square(x[:, :, :2]) + 1e-4, dim=2, keepdim=True)) # (N, N, 1)

        # Create mask based on observation radius
        mask = (dist_2d <= self.obs_radius).float() # (N, N, 1)

        # Permuate for Conv1D
        x_conv_in = x.permute(0, 2, 1) # (N, 12, N)

        x = F.relu(self.conv1(x_conv_in)) # (N, 64, N)
        x = F.relu(self.conv2(x)) # (N, 128, N)
        x = F.relu(self.conv3(x)) # (N, 64, N)
        h_raw = self.conv4(x) # (N, 1, N)

        # Permute back
        h_raw = h_raw.permute(0, 2, 1)

        # Apply mask to CBF output
        h = h_raw * mask

        return h, mask
    
def loss_actions_pytorch(s, u, s_ref):
    """
    Loss function for the action neural network
    Args:
        s: (N, S) Current state of N agents
        s_ref: (N, S) Reference state of N agents
        u: (N, S) Control input from network
    """
    # Calculate reference control
    u_ref = quadrotor_controller_pytorch(s, s_ref)

    # Calculate deviation
    error = u - u_ref

    # Huber loss per element
    loss_val_per_element = torch.where(torch.abs(error) < 1.0, # If absolute error is less than 1.0
                                   torch.abs(error),       # Use absolute error
                                   error**2)               # Else, use squared error
    
    # Sum over the action dimensions (S_dim) to get loss per agent
    loss_per_agent_dim = loss_val_per_element.sum(dim=-1) # (N,)
    
    # Compute safe mask considering all drones (N, N) boolean representing safe or self
    raw_safe_mask = compute_safe_mask_pytorch(s, config.DIST_SAFE)

    # Calculates, for each agent, the proportion of all other agents (and itself) that it is safe with
    safe_proportion_per_agent = raw_safe_mask.float().mean(dim=1) # (N,)

    # We say that an agent is truly safe if it is safe with all other agents
    final_binary_safe_mask = (safe_proportion_per_agent == 1.0).float() # (N,)

    # We only consider the agents deviation from the original control if it is safe
    masked_loss = loss_per_agent_dim * final_binary_safe_mask # (N,)

    sum_masked_loss = masked_loss.sum()

    sum_safe_mask_count = final_binary_safe_mask.sum()

    loss = sum_masked_loss / (1e-4 + sum_safe_mask_count)

    return loss

def loss_barrier_pytorch(h, s, eps=[5e-2, 1e-3]):
    """
    Loss function for the control barrier function
    Args:
        h: (N, N, 1) Output for the CBF neural network
        s: (N, 10) Current state of N agents
        eps: Margin factors
    Returns:
        loss_dang: Barrier loss for dangerous states
        loss_safe: Barrier loss for safe states
        Accuracies are omitted for now
    """
    h_flat = h.flatten()

    # Compute and flatten danger mask
    dang_mask = compute_dangerous_mask_pytorch(s, r=config.DIST_MIN_THRES)
    dang_mask_flat = dang_mask.flatten()

    # Compute and flatten safe mask
    safe_mask = compute_safe_mask_pytorch(s, r=config.DIST_SAFE)
    safe_mask_flat = safe_mask.flatten()

    # Boolean indexing
    dang_h = h_flat[dang_mask_flat]
    safe_h = h_flat[safe_mask_flat]

    num_dang = dang_h.numel()
    num_safe = safe_h.numel()

    # Penalize h > 0 or (h > -eps[0]) when in dangerous region
    loss_dang = torch.relu(dang_h + eps[0]).sum() / (1e-5 + num_dang)
    
    # Penalize h < 0 or (h < eps[1]) when in safe region
    loss_safe = torch.relu(-safe_h + eps[1]).sum() / (1e-5 + num_safe)

    return loss_dang, loss_safe

def loss_derivatives_pytorch(s, u, h, model_CBF, eps=[8e-2, 0, 3e-2]):
    """
    Build the loss function for the derivatives of the CBF.

    Args:
        s (torch.Tensor): The current state of N agents (N, S_dim).
        u (torch.Tensor): The control action / desired dsdt (N, S_dim).
        h (torch.Tensor): The current control barrier function value (N, N, 1).
        eps (list/tuple): [eps0, eps1, eps2] The margin factors for dangerous, safe, and medium states.

    Returns:
        loss_dang_deriv (torch.Tensor): The derivative loss of dangerous states (scalar).
        loss_safe_deriv (torch.Tensor): The derivative loss of safe states (scalar).
        loss_medium_deriv (torch.Tensor): The derivative loss of medium states (scalar).
    """
    # u is already dsdt
    dsdt = u # (N, S_dim)

    # Predict the next state
    s_next = s + dsdt * config.TIME_STEP # (N, S_dim)

    # Calculate state differences for the next state
    x_next = s_next.unsqueeze(1) - s_next.unsqueeze(0) # (N, N, S_dim)

    # Predict the CBF value at the next state using the CBF network, model must be instantiated
    h_next, mask_next = model_CBF(x_next, config.DIST_MIN_THRES) # (N, N, 1)

    # Calculate the core CBF derivative term
    # deriv = h_next - h + config.TIME_STEP * config.ALPHA_CBF * h
    deriv = h_next - h + config.TIME_STEP * config.ALPHA_CBF * h # (N, N, 1)

    # Flatten the derivative term for boolean masking
    deriv_flat = deriv.flatten() # (N * N,)

    # Compute masks for different safety regions
    dang_mask = compute_dangerous_mask_pytorch(s, r=config.DIST_MIN_THRES) # (N, N) boolean
    dang_mask_flat = dang_mask.flatten() # (N * N,) boolean

    safe_mask = compute_safe_mask_pytorch(s, r=config.DIST_SAFE) # (N, N) boolean
    safe_mask_flat = safe_mask.flatten() # (N * N,) boolean

    # Medium mask is simply "not dangerous AND not safe"
    medium_mask_flat = torch.logical_not(torch.logical_or(dang_mask_flat, safe_mask_flat)) # (N * N,) boolean

    # Extract derivative values for each region using boolean indexing
    dang_deriv = deriv_flat[dang_mask_flat] # (num_dang,)
    safe_deriv = deriv_flat[safe_mask_flat] # (num_safe,)
    medium_deriv = deriv_flat[medium_mask_flat] # (num_medium,)

    # Get counts for normalization
    num_dang = dang_deriv.numel()
    num_safe = safe_deriv.numel()
    num_medium = medium_deriv.numel()

    # Calculate losses for each region
    # Penalize if deriv < eps (i.e., torch.relu(-deriv + eps) > 0)
    loss_dang_deriv = torch.relu(-dang_deriv + eps[0]).sum() / (1e-5 + num_dang)
    loss_safe_deriv = torch.relu(-safe_deriv + eps[1]).sum() / (1e-5 + num_safe)
    loss_medium_deriv = torch.relu(-medium_deriv + eps[2]).sum() / (1e-5 + num_medium)

    return loss_dang_deriv, loss_safe_deriv, loss_medium_deriv
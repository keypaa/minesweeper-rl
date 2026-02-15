import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MinesweeperAgent(nn.Module):
    def __init__(self, height, width, hidden_size=4096, device=None):
        super().__init__()
        self.height = height
        self.width = width
        input_size = height * width
        
        # Use provided device or auto-detect
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # MUCH LARGER network to maximize GPU utilization
        self.net = nn.Sequential(
            # First conv block - 256 channels
            nn.Conv2d(1, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Second conv block - 512 channels
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Third conv block - 512 channels
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
            # Large FC layers
            nn.Linear(512 * input_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * input_size)
        )
        self.net.to(self.device)
        # Create matching target network
        self.target_net = nn.Sequential(
            # First conv block - 256 channels
            nn.Conv2d(1, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Second conv block - 512 channels
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Third conv block - 512 channels
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
            # Large FC layers
            nn.Linear(512 * input_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * input_size)
        )
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.to(self.device)
        
        # Normalization for inputs (-3 to 9) -> roughly [-1, 1]
        self.register_buffer('obs_mean', torch.tensor(0.0))
        self.register_buffer('obs_std', torch.tensor(5.0))

    def forward(self, obs, use_target=False):
        obs = obs.float().to(self.device)
        # Normalize inputs
        obs = (obs - self.obs_mean) / self.obs_std
        if obs.dim() == 2:  # single (9,9)
            obs = obs.unsqueeze(0).unsqueeze(0)  # (1,1,9,9)
        elif obs.dim() == 3:  # batch (batch,9,9)
            obs = obs.unsqueeze(1)  # (batch,1,9,9)
        network = self.target_net if use_target else self.net
        return network(obs)

    def get_action(self, obs, epsilon=0.1, mask_invalid=True):
        if np.random.rand() < epsilon:
            # Explore only valid actions
            if mask_invalid:
                valid_actions = self._get_valid_actions(obs)
                if len(valid_actions) > 0:
                    return np.random.choice(valid_actions)
            return np.random.randint(2 * self.height * self.width)
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self(obs_tensor)
            if mask_invalid:
                # Mask invalid actions (already revealed or flagged cells)
                mask = self._get_action_mask(obs)
                mask_tensor = torch.from_numpy(mask).float().to(self.device)
                q_values = q_values - (1 - mask_tensor) * 1e9  # Large penalty for invalid
            return q_values.argmax().item()
    
    def _get_valid_actions(self, obs):
        """Get list of valid actions (unrevealed, unflagged cells for reveal)"""
        valid = []
        for i in range(self.height):
            for j in range(self.width):
                # Cell is hidden (-2) or flagged (-3)
                if obs[i, j] == -2:  # Can reveal
                    valid.append(i * self.width + j)
                # Can always try to flag/unflag (action >= height*width)
                valid.append(self.height * self.width + i * self.width + j)
        return valid if valid else list(range(2 * self.height * self.width))
    
    def _get_action_mask(self, obs):
        """Create mask for valid actions (1=valid, 0=invalid)"""
        mask = np.zeros(2 * self.height * self.width)
        for i in range(self.height):
            for j in range(self.width):
                idx = i * self.width + j
                # Can reveal if hidden
                if obs[i, j] == -2:
                    mask[idx] = 1
                # Can flag/unflag unrevealed cells
                if obs[i, j] <= -2:
                    mask[self.height * self.width + idx] = 1
        # Ensure at least one action is valid
        if mask.sum() == 0:
            mask[:] = 1
        return mask

    def get_actions_batch(self, obs_batch, epsilon, mask_invalid=True):
        """Vectorized batch action selection for GPU efficiency"""
        batch_size = obs_batch.shape[0]
        
        # Random exploration for some samples
        explore_mask = np.random.rand(batch_size) < epsilon
        actions = np.zeros(batch_size, dtype=np.int64)
        
        # Handle exploration with valid actions
        if mask_invalid and explore_mask.any():
            for i in np.where(explore_mask)[0]:
                valid_actions = self._get_valid_actions(obs_batch[i])
                actions[i] = np.random.choice(valid_actions) if len(valid_actions) > 0 else np.random.randint(2 * self.height * self.width)
        else:
            actions[explore_mask] = np.random.randint(0, 2 * self.height * self.width, size=explore_mask.sum())
        
        # Vectorized exploitation for remaining samples
        if not explore_mask.all():
            exploit_indices = np.where(~explore_mask)[0]
            if len(exploit_indices) > 0:
                obs_tensor = torch.from_numpy(obs_batch[exploit_indices]).float().to(self.device)
                with torch.no_grad():
                    q_values = self(obs_tensor)  # Already batched
                    
                    if mask_invalid:
                        # Vectorized action masking
                        masks = np.array([self._get_action_mask(obs_batch[i]) for i in exploit_indices])
                        mask_tensor = torch.from_numpy(masks).float().to(self.device)
                        q_values = q_values - (1 - mask_tensor) * 1e9
                    
                    actions[exploit_indices] = q_values.argmax(dim=1).cpu().numpy()
        
        return actions


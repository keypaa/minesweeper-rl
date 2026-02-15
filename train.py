import torch
import torch.optim as optim
from minesweeper_env import MinesweeperEnv
from agent import MinesweeperAgent
from gui import MinesweeperGUI
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import logging
from datetime import datetime
from collections import deque
import random
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import time
import os
import argparse

# Use newer torch.amp API if available, fallback to older torch.cuda.amp
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = torch.cuda.is_available()
    AMP_DEVICE = 'cuda'
except (ImportError, AttributeError):
    try:
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = torch.cuda.is_available()
        AMP_DEVICE = None  # Old API doesn't need device parameter
    except ImportError:
        AMP_AVAILABLE = False
        AMP_DEVICE = None

def get_device_config(device_name):
    """Get optimal configuration based on device"""
    if device_name == 'cuda' and torch.cuda.is_available():
        return {
            'device': torch.device('cuda'),
            'num_envs': 128,
            'batch_size': 2048,
            'updates_per_step': 4,
            'replay_buffer_size': 100000,
            'use_amp': AMP_AVAILABLE,
            'use_compile': hasattr(torch, 'compile'),
            'non_blocking': True
        }
    else:
        # CPU configuration with more conservative settings
        return {
            'device': torch.device('cpu'),
            'num_envs': 8,  # Much fewer parallel environments for CPU
            'batch_size': 128,  # Smaller batch size for CPU memory
            'updates_per_step': 1,  # Single update per step on CPU
            'replay_buffer_size': 50000,  # Smaller replay buffer
            'use_amp': False,  # No mixed precision on CPU
            'use_compile': False,  # torch.compile can be slower on CPU
            'non_blocking': False
        }

def get_timestamp():
    """Get formatted timestamp for logging"""
    return datetime.now().strftime('%H:%M:%S')

def train(device_name='auto', num_episodes=1000):
    """Train the Minesweeper agent
    
    Args:
        device_name: 'auto', 'cuda', or 'cpu'. Auto will use CUDA if available.
        num_episodes: Number of episodes to train for
    """
    start_time = time.time()
    
    # Device configuration
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = get_device_config(device_name)
    device = config['device']
    
    print(f"[{get_timestamp()}] Training on device: {device}")
    print(f"[{get_timestamp()}] Configuration: {config['num_envs']} envs, batch size {config['batch_size']}")
    
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Tracking metrics for plotting
    episode_history = []
    win_rate_history = []
    reward_history = []
    time_history = []
    
    # Create environments with device-appropriate settings
    num_envs = config['num_envs']
    env = AsyncVectorEnv([lambda: MinesweeperEnv() for _ in range(num_envs)])
    single_env = MinesweeperEnv()
    agent = MinesweeperAgent(single_env.height, single_env.width, 512, device=device)
    agent.to(device)
    
    # Compile model for faster execution (PyTorch 2.0+) - only on GPU
    if config['use_compile'] and hasattr(torch, 'compile'):
        print(f"[{get_timestamp()}] Compiling model with torch.compile for faster execution...")
        agent.net = torch.compile(agent.net, mode='max-autotune')
        agent.target_net = torch.compile(agent.target_net, mode='max-autotune')
    
    optimizer = optim.AdamW(agent.parameters(), lr=0.0003, weight_decay=1e-5)
    gui = MinesweeperGUI(single_env, agent)

    # Device-specific batch size and buffer
    replay_buffer = deque(maxlen=config['replay_buffer_size'])
    batch_size = config['batch_size']
    warmup_steps = min(2000, config['replay_buffer_size'] // 2)
    updates_per_step = config['updates_per_step']
    
    # Mixed precision training for faster computation (GPU only)
    if config['use_amp']:
        scaler = GradScaler(AMP_DEVICE) if AMP_DEVICE else GradScaler()
        print(f"[{get_timestamp()}] Using mixed precision training (AMP) for faster GPU computation")
    else:
        scaler = None

    obs, _ = env.reset()
    total_rewards = np.zeros(num_envs)
    step_counts = np.zeros(num_envs, dtype=int)
    episode_count = 0
    total_wins = 0
    global_step = 0
    update_count = 0
    
    print(f"[{get_timestamp()}] Training with {num_envs} parallel environments")
    print(f"[{get_timestamp()}] Batch size: {batch_size}, Updates per step: {updates_per_step}")
    print(f"[{get_timestamp()}] Replay buffer: {len(replay_buffer)}/{replay_buffer.maxlen}")
    print(f"[{get_timestamp()}] Device: {device}")
    print(f"[{get_timestamp()}] Model parameters: {sum(p.numel() for p in agent.parameters()):,}")
    
    # Pre-allocate GPU memory to reduce allocation overhead (GPU only)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"[{get_timestamp()}] Pre-allocating GPU memory...")
        with torch.no_grad():
            dummy_obs = torch.randn(batch_size, 9, 9).to(device)
            _ = agent(dummy_obs)
            del dummy_obs
        torch.cuda.synchronize()
        print(f"[{get_timestamp()}] GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
        print(f"[{get_timestamp()}] GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.0f} MB")
    
    while episode_count < num_episodes:
        epsilon = max(0.05, 0.9 - episode_count * 0.002)
        actions = agent.get_actions_batch(obs, epsilon)
        next_obs, rewards, dones_step, _, _ = env.step(actions)
        global_step += num_envs
        
        for i in range(num_envs):
            replay_buffer.append((obs[i], actions[i], rewards[i], next_obs[i], dones_step[i]))
            total_rewards[i] += rewards[i]
            step_counts[i] += 1
            if step_counts[i] >= 50 or dones_step[i]:
                episode_count += 1
                if total_rewards[i] > 5:  # Consider it a win if positive reward
                    total_wins += 1
                win_rate = total_wins / episode_count
                avg_reward = total_rewards[i] / step_counts[i] if step_counts[i] > 0 else 0
                
                # Track metrics for plotting
                episode_history.append(episode_count)
                win_rate_history.append(win_rate)
                reward_history.append(total_rewards[i])
                elapsed_time = time.time() - start_time
                time_history.append(elapsed_time)
                
                # Print less frequently to reduce I/O overhead
                if episode_count % 50 == 0 or total_rewards[i] > 5:
                    eps_per_sec = episode_count / elapsed_time if elapsed_time > 0 else 0
                    print(f"[{get_timestamp()}] Episode {episode_count}: Reward {total_rewards[i]:.1f}, Steps {step_counts[i]}, Win Rate {win_rate:.3f}, Epsilon {epsilon:.3f} ({eps_per_sec:.1f} eps/s)")
                
                # Update plots every 50 episodes
                if episode_count % 50 == 0 and episode_count > 0:
                    plot_training_progress(episode_history, win_rate_history, reward_history, time_history)
                
                logger.info(f"Episode {episode_count}: Reward {total_rewards[i]:.1f}, Steps {step_counts[i]}, Avg Reward {avg_reward:.2f}, Win Rate {win_rate:.3f}")
                # reset for next
                total_rewards[i] = 0
                step_counts[i] = 0
        
        # Train with device-appropriate number of updates per step
        if len(replay_buffer) >= warmup_steps:
            for _ in range(updates_per_step):
                batch = random.sample(replay_buffer, batch_size)
                
                # Efficient tensor conversion (use non_blocking for GPU)
                non_blocking = config['non_blocking']
                obs_batch = torch.stack([torch.tensor(b[0], dtype=torch.float32) for b in batch]).to(device, non_blocking=non_blocking)
                action_batch = torch.tensor([b[1] for b in batch], dtype=torch.long).to(device, non_blocking=non_blocking)
                reward_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(device, non_blocking=non_blocking)
                next_obs_batch = torch.stack([torch.tensor(b[3], dtype=torch.float32) for b in batch]).to(device, non_blocking=non_blocking)
                done_batch = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(device, non_blocking=non_blocking)
                
                optimizer.zero_grad()
                
                if config['use_amp']:
                    # Mixed precision training for faster computation (GPU)
                    autocast_ctx = autocast(AMP_DEVICE) if AMP_DEVICE else autocast()
                    with autocast_ctx:
                        # Double DQN: use main network to select action, target network to evaluate
                        q_values = agent(obs_batch)
                        with torch.no_grad():
                            next_q_values = agent(next_obs_batch)
                            best_actions = next_q_values.argmax(dim=1)
                            target_next_q = agent(next_obs_batch, use_target=True)
                            targets = reward_batch + 0.99 * target_next_q.gather(1, best_actions.unsqueeze(1)).squeeze() * (1 - done_batch)
                        
                        current_q = q_values.gather(1, action_batch.unsqueeze(1)).squeeze()
                        loss = ((current_q - targets) ** 2).mean()
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard training (CPU or GPU without AMP)
                    q_values = agent(obs_batch)
                    with torch.no_grad():
                        next_q_values = agent(next_obs_batch)
                        best_actions = next_q_values.argmax(dim=1)
                        target_next_q = agent(next_obs_batch, use_target=True)
                        targets = reward_batch + 0.99 * target_next_q.gather(1, best_actions.unsqueeze(1)).squeeze() * (1 - done_batch)
                    
                    current_q = q_values.gather(1, action_batch.unsqueeze(1)).squeeze()
                    loss = ((current_q - targets) ** 2).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()
                
                update_count += 1
        
        # Update target network periodically
        if update_count > 0 and update_count % 500 == 0:
            agent.target_net.load_state_dict(agent.net.state_dict())
            if update_count % 2000 == 0:
                elapsed = time.time() - start_time
                print(f"[{get_timestamp()}] [Step {global_step}] Updated target network. Win rate: {win_rate:.3f}, Buffer: {len(replay_buffer)}, Time: {elapsed/60:.1f}m")
        
        # AsyncVectorEnv automatically resets done environments
        obs = next_obs
        
        if episode_count % 500 == 0 and episode_count > 0:
            print(f"[{get_timestamp()}] Visualizing episode {episode_count}...")
            gui.run_episode()
    
    # Final plot
    plot_training_progress(episode_history, win_rate_history, reward_history, time_history)
    
    elapsed = time.time() - start_time
    print(f"\n[{get_timestamp()}] Training complete! Final win rate: {win_rate:.3f}")
    print(f"[{get_timestamp()}] Total updates: {update_count}, Total steps: {global_step}")
    print(f"[{get_timestamp()}] Total training time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"[{get_timestamp()}] Average speed: {episode_count/elapsed:.1f} episodes/second")

def plot_training_progress(episodes, win_rates, rewards, times):
    """Plot and save training progress graphs"""
    if len(episodes) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Minesweeper RL Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Win Rate over Episodes
    axes[0, 0].plot(episodes, win_rates, 'b-', linewidth=2, alpha=0.7)
    # Add moving average (window=50)
    if len(win_rates) >= 50:
        win_rate_ma = np.convolve(win_rates, np.ones(50)/50, mode='valid')
        axes[0, 0].plot(episodes[49:], win_rate_ma, 'r-', linewidth=2, label='MA(50)')
        axes[0, 0].legend()
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Win Rate', fontsize=12)
    axes[0, 0].set_title('Win Rate Progress', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, max(win_rates) * 1.1 if win_rates else 1)
    
    # Plot 2: Episode Rewards
    axes[0, 1].scatter(episodes, rewards, alpha=0.3, s=10, c='green')
    # Add moving average
    if len(rewards) >= 50:
        reward_ma = np.convolve(rewards, np.ones(50)/50, mode='valid')
        axes[0, 1].plot(episodes[49:], reward_ma, 'r-', linewidth=2, label='MA(50)')
        axes[0, 1].legend()
    axes[0, 1].set_xlabel('Episode', fontsize=12)
    axes[0, 1].set_ylabel('Episode Reward', fontsize=12)
    axes[0, 1].set_title('Reward Progress', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Win Threshold')
    
    # Plot 3: Win Rate over Time
    time_minutes = [t/60 for t in times]
    axes[1, 0].plot(time_minutes, win_rates, 'purple', linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Time (minutes)', fontsize=12)
    axes[1, 0].set_ylabel('Win Rate', fontsize=12)
    axes[1, 0].set_title('Win Rate vs Training Time', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Training Speed (episodes per minute)
    if len(time_minutes) > 1:
        eps_per_min = [episodes[i] / time_minutes[i] if time_minutes[i] > 0 else 0 
                       for i in range(len(episodes))]
        axes[1, 1].plot(episodes, eps_per_min, 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Episode', fontsize=12)
        axes[1, 1].set_ylabel('Episodes/Minute', fontsize=12)
        axes[1, 1].set_title('Training Speed', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f'plots/training_progress_{timestamp}.png'
    plt.savefig(plot_filename, dpi=100, bbox_inches='tight')
    plt.savefig('plots/training_progress_latest.png', dpi=100, bbox_inches='tight')  # Always update latest
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Minesweeper RL Agent')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='Device to train on (auto: use CUDA if available, otherwise CPU)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train for (default: 1000)')
    args = parser.parse_args()
    
    train(device_name=args.device, num_episodes=args.episodes)
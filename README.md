# Minesweeper Reinforcement Learning

Train an AI agent to play Minesweeper using Deep Q-Learning (DQN). This project demonstrates reinforcement learning techniques applied to the classic Minesweeper game, with support for both CPU and GPU training.

## 🎯 Features

- **Deep Q-Network (DQN)** with target network and experience replay
- **Double DQN** for more stable training
- **Parallel environment training** for faster data collection
- **Support for both CPU and GPU** with automatic optimization
- **Real-time visualization** with Pygame GUI
- **Training progress plots** with win rate, rewards, and speed metrics
- **Mixed precision training** (GPU only) for faster computation
- **Configurable hyperparameters** via command line

## 📋 Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- Pygame
- Matplotlib
- NumPy

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/keypaa/minesweeper-rl.git
cd minesweeper-rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Basic Training (Auto-detect GPU/CPU)

```bash
python train.py
```

This will automatically use CUDA if available, otherwise fallback to CPU.

### Specify Device

Train on GPU:
```bash
python train.py --device cuda
```

Train on CPU:
```bash
python train.py --device cpu
```

### Custom Number of Episodes

```bash
python train.py --episodes 2000
```

### All Options

```bash
python train.py --device cuda --episodes 5000
```

## ⚙️ Configuration

The training automatically adjusts settings based on the device:

### GPU Configuration (CUDA)
- **Parallel Environments**: 128
- **Batch Size**: 2048
- **Updates per Step**: 4
- **Replay Buffer**: 100,000
- **Mixed Precision**: Enabled
- **Model Compilation**: Enabled (PyTorch 2.0+)

### CPU Configuration
- **Parallel Environments**: 8
- **Batch Size**: 128
- **Updates per Step**: 1
- **Replay Buffer**: 50,000
- **Mixed Precision**: Disabled
- **Model Compilation**: Disabled

## 📊 Training Outputs

### Logs
Training logs are saved to `training_log_YYYYMMDD_HHMMSS.txt` with detailed episode information.

### Plots
Training progress plots are saved to the `plots/` directory, showing:
- Win rate over episodes (with 50-episode moving average)
- Episode rewards (with moving average)
- Win rate vs training time
- Training speed (episodes per minute)

### Visualization
Every 500 episodes, the agent plays a game with the GUI to visualize its current strategy.

## 🏗️ Project Structure

```
.
├── train.py              # Main training script
├── agent.py              # DQN agent implementation
├── minesweeper_env.py    # Gymnasium environment for Minesweeper
├── gui.py                # Pygame visualization
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## 🧠 How It Works

### Environment
The Minesweeper environment is implemented as a Gymnasium-compatible environment:
- **State**: 9x9 grid with cell values (-3: flagged, -2: hidden, -1: mine, 0-8: adjacent mine count)
- **Actions**: 162 actions (81 reveal + 81 flag/unflag)
- **Rewards**: 
  - +10 for winning
  - -5 for hitting a mine
  - +1.0 for revealing a 0-cell
  - +0.2 for revealing numbered cells
  - +0.1 bonus per cell revealed by cascading
  - -1 for invalid moves

### Agent Architecture
- **Convolutional layers**: Extract spatial features from the board
- **Fully connected layers**: Map features to Q-values for each action
- **Target network**: Stabilizes training by providing consistent targets
- **Experience replay**: Breaks correlation between consecutive samples

### Training Algorithm
1. **Collect experiences** from parallel environments
2. **Store in replay buffer** (50K-100K capacity)
3. **Sample random batches** for training (128-2048 samples)
4. **Update policy network** using Double DQN
5. **Periodically sync target network** every 500 updates
6. **Epsilon-greedy exploration** (decays from 0.9 to 0.05)

## 📈 Expected Performance

With GPU training (1000 episodes):
- **Training Time**: ~11-15 minutes
- **Win Rate**: 20-25%
- **Training Speed**: ~1.5 episodes/second

With CPU training (1000 episodes):
- **Training Time**: ~1-2 hours (varies by CPU)
- **Win Rate**: Similar final performance
- **Training Speed**: ~0.2-0.5 episodes/second

## 🔧 Customization

You can modify hyperparameters in `train.py`:
- Learning rate, batch size, buffer size
- Network architecture in `agent.py`
- Reward structure in `minesweeper_env.py`
- Epsilon decay schedule

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with PyTorch and Gymnasium
- Inspired by DeepMind's DQN paper
- Minesweeper game mechanics based on the classic Windows game

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy Training! 🎮🤖**
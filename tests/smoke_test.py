import numpy as np
import torch

from minesweeper_env import MinesweeperEnv
from agent import MinesweeperAgent


def test_env_and_agent_shapes():
    env = MinesweeperEnv()
    obs, _ = env.reset()
    assert obs.shape == (env.height, env.width)

    agent = MinesweeperAgent(env.height, env.width, device=torch.device('cpu'), model='small')
    # Single action
    a = agent.get_action(obs, epsilon=0.0)
    assert 0 <= a < 2 * env.width * env.height

    # Batch actions
    batch = np.stack([obs, obs], axis=0)
    acts = agent.get_actions_batch(batch, epsilon=0.0)
    assert acts.shape[0] == 2


if __name__ == "__main__":
    test_env_and_agent_shapes()
    print("Smoke test passed")

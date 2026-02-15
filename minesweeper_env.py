import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MinesweeperEnv(gym.Env):
    def __init__(self, width=9, height=9, mines=10):
        super().__init__()
        self.width = width
        self.height = height
        self.mines = mines
        self.action_space = spaces.Discrete(2 * width * height)  # 0 to width*height-1: reveal, width*height to 2*width*height-1: flag
        self.observation_space = spaces.Box(low=-3, high=9, shape=(height, width), dtype=np.int32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.revealed = np.zeros((self.height, self.width), dtype=bool)
        self.flags = np.zeros((self.height, self.width), dtype=bool)
        self.game_over = False
        self.won = False
        # Place mines
        positions = np.random.choice(self.width * self.height, self.mines, replace=False)
        for pos in positions:
            x, y = pos // self.width, pos % self.width
            self.board[x, y] = -1  # Mine
        # Calculate numbers
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i, j] != -1:
                    count = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.height and 0 <= nj < self.width and self.board[ni, nj] == -1:
                                count += 1
                    self.board[i, j] = count
        return self._get_obs(), {}

    def step(self, action):
        is_flag = action >= self.width * self.height
        pos = action % (self.width * self.height)
        x, y = pos // self.width, pos % self.width
        if is_flag:
            # Toggle flag
            if self.revealed[x, y]:
                return self._get_obs(), -2, False, False, {}  # Can't flag revealed
            self.flags[x, y] = not self.flags[x, y]
            return self._get_obs(), 0.1, False, False, {}  # Small reward for flagging
        else:
            # Reveal
            if self.revealed[x, y] or self.flags[x, y]:
                return self._get_obs(), -1, False, False, {}  # Invalid move (light penalty)
            self.revealed[x, y] = True
            if self.board[x, y] == -1:
                self.game_over = True
                return self._get_obs(), -5, True, False, {}  # Hit mine
            # Reward based on cell value (revealing 0s is good)
            cells_before = np.sum(self.revealed)
            reward = 1.0 if self.board[x, y] == 0 else 0.2
            if self.board[x, y] == 0:
                self._reveal_empty(x, y)
                # Bonus for revealing multiple cells
                cells_revealed = np.sum(self.revealed) - cells_before
                reward += cells_revealed * 0.1
            if self._check_win():
                self.won = True
                return self._get_obs(), 10, True, False, {}  # Win
            return self._get_obs(), reward, False, False, {}  # Continue with positive reward

    def _check_win(self):
        # Win if all non-mine cells revealed
        safe_revealed = np.sum(self.revealed & (self.board != -1))
        total_safe = self.width * self.height - self.mines
        return safe_revealed == total_safe

    def _reveal_empty(self, x, y):
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = x + di, y + dj
                if 0 <= ni < self.height and 0 <= nj < self.width and not self.revealed[ni, nj]:
                    self.revealed[ni, nj] = True
                    if self.board[ni, nj] == 0:
                        self._reveal_empty(ni, nj)

    def _get_obs(self):
        obs = np.full((self.height, self.width), -2, dtype=np.int32)
        obs[self.flags & ~self.revealed] = -3  # Flagged hidden
        obs[self.revealed] = self.board[self.revealed]
        return obs

    def render(self, mode='human'):
        # Simple text render for now
        for i in range(self.height):
            row = []
            for j in range(self.width):
                if self.flags[i, j]:
                    row.append('F')
                elif not self.revealed[i, j]:
                    row.append('.')
                elif self.board[i, j] == -1:
                    row.append('*')
                else:
                    row.append(str(self.board[i, j]))
            print(' '.join(row))
        print()
import os
import pygame
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch



class MinesweeperGUI:
    def __init__(self, env, agent, headless=True):
        """Create GUI.

        Args:
            env: MinesweeperEnv instance
            agent: MinesweeperAgent instance
            headless: if True, use SDL dummy driver (no visible window). If False, open a visible window.
        """
        self.env = env
        self.agent = agent
        self.headless = bool(headless)

        # If headless requested, set SDL to dummy before initializing pygame
        if self.headless:
            os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        else:
            # Ensure dummy driver is removed when a visible window is requested
            if os.environ.get('SDL_VIDEODRIVER', '').lower() == 'dummy':
                try:
                    del os.environ['SDL_VIDEODRIVER']
                except Exception:
                    pass

        # If pygame is already initialized (for example earlier headless GUI),
        # quit and reinitialize so a visible driver can be selected.
        if not self.headless and pygame.get_init():
            try:
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass

        # Debug info for driver selection
        try:
            print(f"[GUI] SDL_VIDEODRIVER env before init: {os.environ.get('SDL_VIDEODRIVER')}")
        except Exception:
            pass
        pygame.init()
        self.cell_size = 40
        self.screen = pygame.display.set_mode((env.width * self.cell_size, env.height * self.cell_size + 200))
        pygame.display.set_caption("Minesweeper RL")
        # Print which pygame video driver is active (useful for debugging on Windows)
        try:
            driver = pygame.display.get_driver()
            print(f"[GUI] pygame display driver: {driver}")
        except Exception:
            pass
        self.font = pygame.font.SysFont(None, 24)
        self.rewards = []
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        plt.ion()
        self.line1, = self.ax1.plot([], [], 'r-', label='Total Reward')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Reward')
        self.ax1.legend()
        self.line2, = self.ax2.plot([], [], 'b-', label='Win Rate')
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Win Rate')
        self.ax2.legend()
        self.win_count = 0
        self.win_rates = []

    def draw_board(self):
        # Clear background to a light color so text is visible
        self.screen.fill((230, 230, 230))
        for i in range(self.env.height):
            for j in range(self.env.width):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                # Fill depending on cell state
                if self.env.revealed[i, j]:
                    # Revealed cell background
                    pygame.draw.rect(self.screen, (245, 245, 245), rect)
                else:
                    # Hidden cell background
                    pygame.draw.rect(self.screen, (200, 200, 200), rect)

                # Draw cell border
                pygame.draw.rect(self.screen, (120, 120, 120), rect, 1)

                # Determine display text and color
                if self.env.flags[i, j] and not self.env.revealed[i, j]:
                    text = self.font.render('F', True, (200, 30, 30))
                elif not self.env.revealed[i, j]:
                    # Use a subtle dot for hidden cells
                    text = self.font.render('.', True, (80, 80, 80))
                elif self.env.board[i, j] == -1:
                    text = self.font.render('*', True, (200, 30, 30))
                else:
                    # Use darker color for numbers
                    text = self.font.render(str(self.env.board[i, j]), True, (10, 10, 10))

                # Center the text within the cell
                text_rect = text.get_rect()
                text_pos = (j * self.cell_size + (self.cell_size - text_rect.width) // 2,
                            i * self.cell_size + (self.cell_size - text_rect.height) // 2)
                self.screen.blit(text, text_pos)

    def update_graph(self, frame):
        self.line1.set_data(range(len(self.rewards)), self.rewards)
        self.line2.set_data(range(len(self.win_rates)), self.win_rates)
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        plt.tight_layout()

    def run_episode(self):
        obs, _ = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            action = self.agent.get_action(obs)
            obs, reward, done, _, _ = self.env.step(action)
            total_reward += reward
            self.draw_board()
            # Removed LLM reasoning for speed
            # Process events so the window remains responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # If window closed by user, break out of the episode loop
                    done = True
            pygame.display.flip()
            # Small delay to make visualization visible but not too slow
            pygame.time.wait(30)
        self.rewards.append(total_reward)
        if total_reward > 0:
            self.win_count += 1
        self.win_rates.append(self.win_count / len(self.rewards))
        if len(self.rewards) > 100:
            self.rewards.pop(0)
            self.win_rates.pop(0)
        self.update_graph(0)
        # Only call plt.pause when running headless (interactive not available with Agg)
        try:
            if self.headless:
                plt.pause(0.01)
            else:
                # When running with a visible pygame window, avoid interactive pause (backend may be Agg)
                # Save a snapshot of the training progress without blocking
                plt.savefig('training_progress_play.png')
        except Exception:
            # Matplotlib may raise if backend is non-interactive; ignore and continue
            pass

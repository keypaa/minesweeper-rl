import pygame
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
import os

os.environ['SDL_VIDEODRIVER'] = 'dummy'

class MinesweeperGUI:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        pygame.init()
        self.cell_size = 40
        self.screen = pygame.display.set_mode((env.width * self.cell_size, env.height * self.cell_size + 200))
        pygame.display.set_caption("Minesweeper RL")
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
        for i in range(self.env.height):
            for j in range(self.env.width):
                rect = pygame.Rect(j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
                if self.env.flags[i, j]:
                    text = self.font.render('F', True, (255, 0, 0))
                elif not self.env.revealed[i, j]:
                    text = self.font.render('.', True, (0, 0, 0))
                elif self.env.board[i, j] == -1:
                    text = self.font.render('*', True, (255, 0, 0))
                else:
                    text = self.font.render(str(self.env.board[i, j]), True, (0, 0, 0))
                self.screen.blit(text, (j * self.cell_size + 10, i * self.cell_size + 10))

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
            pygame.display.flip()
            pygame.time.wait(10)
        self.rewards.append(total_reward)
        if total_reward > 0:
            self.win_count += 1
        self.win_rates.append(self.win_count / len(self.rewards))
        if len(self.rewards) > 100:
            self.rewards.pop(0)
            self.win_rates.pop(0)
        self.update_graph(0)
        plt.pause(0.01)
        plt.savefig('training_progress.png')
#!/usr/bin/env python
"""Utility to load a saved model and play N visual episodes in a fresh process.

Usage examples:
  python play.py --model model_v1.pt --games 5 --device cpu --headless False
"""
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to saved model file')
    parser.add_argument('--games', type=int, default=5, help='Number of episodes to play')
    parser.add_argument('--device', type=str, default='cpu', help='Device to load model on (cpu or cuda)')
    parser.add_argument('--headless', type=str, default='False', help='Run headless (True/False)')
    parser.add_argument('--mines', type=int, default=10)
    parser.add_argument('--width', type=int, default=9)
    parser.add_argument('--height', type=int, default=9)
    args = parser.parse_args()

    # Determine headless setting
    headless = str(args.headless).lower() in ('1', 'true', 'yes')

    # If a visible window is requested, ensure SDL_VIDEODRIVER is not forcing dummy
    if not headless:
        if 'SDL_VIDEODRIVER' in os.environ:
            try:
                del os.environ['SDL_VIDEODRIVER']
            except Exception:
                pass

    # Import after adjusting environment so pygame picks correct driver
    from agent import MinesweeperAgent
    from minesweeper_env import MinesweeperEnv
    from gui import MinesweeperGUI
    import torch

    device = torch.device(args.device)
    agent = MinesweeperAgent.load_from_file(args.model, device=device)
    env = MinesweeperEnv(width=args.width, height=args.height, mines=args.mines)
    gui = MinesweeperGUI(env, agent, headless=headless)

    for i in range(args.games):
        print(f"Playing episode {i+1}/{args.games}...")
        gui.run_episode()

if __name__ == '__main__':
    main()

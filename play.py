#!/usr/bin/env python
"""Utility to load a saved model and play N visual episodes in a fresh process.

Usage examples:
  python play.py --model model_v1.pt --games 5 --device cpu --headless False
"""
import os
import argparse
import time
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to saved model file')
    parser.add_argument('--games', type=int, default=5, help='Number of episodes to play')
    parser.add_argument('--device', type=str, default='cpu', help='Device to load model on (cpu or cuda)')
    parser.add_argument('--headless', type=str, default='False', help='Run headless (True/False)')
    parser.add_argument('--mines', type=int, default=10)
    parser.add_argument('--width', type=int, default=9)
    parser.add_argument('--height', type=int, default=9)
    parser.add_argument('--max-steps', type=int, default=1000, help='Max steps per episode')
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

    # Create GUI only if rendering requested
    gui = None
    if not headless:
        gui = MinesweeperGUI(env, agent, headless=False)

    results = []
    wins = 0
    start = time.time()
    for i in range(args.games):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        if gui:
            gui.env = env  # ensure GUI references current env

        while not done and steps < args.max_steps:
            action = agent.get_action(obs, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            steps += 1

            if gui:
                gui.draw_board()
                for event in __import__('pygame').event.get():
                    if event.type == __import__('pygame').QUIT:
                        done = True
                __import__('pygame').display.flip()
                __import__('pygame').time.wait(30)

        is_win = bool(getattr(env, 'won', False))
        if is_win:
            wins += 1
        results.append({'episode': i + 1, 'reward': total_reward, 'steps': steps, 'win': int(is_win)})
        timed_out = (steps >= args.max_steps and not is_win)
        extra = ' (timed out)' if timed_out else ''
        print(f"Episode {i+1}/{args.games}: reward={total_reward:.2f}, steps={steps}, win={is_win}{extra}")

    duration = time.time() - start
    rewards = [r['reward'] for r in results]
    steps_list = [r['steps'] for r in results]
    print('\nPlayback summary:')
    print(f'  Episodes: {args.games}')
    print(f'  Wins: {wins} ({wins/args.games:.3%})')
    print(f'  Avg reward: {np.mean(rewards):.3f}, median: {np.median(rewards):.3f}, std: {np.std(rewards):.3f}')
    print(f'  Avg steps: {np.mean(steps_list):.1f}, median steps: {np.median(steps_list):.1f}')
    print(f'  Duration: {duration:.2f}s, episodes/sec: {args.games/duration:.2f}')

if __name__ == '__main__':
    main()

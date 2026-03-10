#!/usr/bin/env python3
"""Evaluate a saved Minesweeper agent.

Runs N greedy episodes and prints summary statistics. Optionally saves per-episode
results to CSV and can render episodes with the GUI.

Example:
  python eval.py --model model_v1.pt --games 100 --device cpu --seed 42 --save-csv results.csv
  python eval.py --model model_v1.pt --games 5 --device cpu --render True
"""
import argparse
import os
import csv
import time
import random
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to saved model file')
    parser.add_argument('--games', type=int, default=100, help='Number of episodes to run')
    parser.add_argument('--device', type=str, default='cpu', help='Device to load model on (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed')
    parser.add_argument('--width', type=int, default=9, help='Board width')
    parser.add_argument('--height', type=int, default=9, help='Board height')
    parser.add_argument('--mines', type=int, default=10, help='Number of mines')
    parser.add_argument('--save-csv', type=str, default=None, help='Optional path to save per-episode results')
    parser.add_argument('--render', type=str, default='False', help='Render episodes (True/False)')
    parser.add_argument('--eval-eps', type=float, default=0.0, help='Epsilon for evaluation (exploration rate)')
    parser.add_argument('--dump-dir', type=str, default=None, help='Directory to dump final-board text files for timed-out or failed episodes')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max steps per episode (safety)')
    args = parser.parse_args()

    render = str(args.render).lower() in ('1', 'true', 'yes')

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # If rendering requested, ensure SDL_VIDEODRIVER isn't forcing dummy
    if render and 'SDL_VIDEODRIVER' in os.environ and os.environ['SDL_VIDEODRIVER'] == 'dummy':
        try:
            del os.environ['SDL_VIDEODRIVER']
        except Exception:
            pass

    from agent import MinesweeperAgent
    from minesweeper_env import MinesweeperEnv
    # GUI imported only if rendering requested (it sets SDL state)
    gui = None
    if render:
        from gui import MinesweeperGUI

    device = torch.device(args.device)
    print(f"Loading model from {args.model} to device {device}")
    agent = MinesweeperAgent.load_from_file(args.model, device=device)

    env = MinesweeperEnv(width=args.width, height=args.height, mines=args.mines)

    results = []
    start = time.time()
    wins = 0
    timeouts = 0
    for i in range(args.games):
        seed = None
        if args.seed is not None:
            seed = args.seed + i
        obs, _ = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        steps = 0
        if render:
            play_gui = MinesweeperGUI(env, agent, headless=False)

        while not done and steps < args.max_steps:
            # Epsilon-greedy action for evaluation (default 0)
            action = agent.get_action(obs, epsilon=args.eval_eps)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            steps += 1
            if render:
                play_gui.draw_board()
                for event in __import__('pygame').event.get():
                    if event.type == __import__('pygame').QUIT:
                        done = True
                __import__('pygame').display.flip()
                __import__('pygame').time.wait(30)

        is_win = bool(getattr(env, 'won', False))
        if is_win:
            wins += 1

        timed_out = (steps >= args.max_steps and not is_win)
        if timed_out:
            timeouts += 1

        results.append({'episode': i + 1, 'reward': total_reward, 'steps': steps, 'win': int(is_win), 'timed_out': int(timed_out)})
        # Optionally dump final board for timed-out or failed episodes
        if args.dump_dir and (timed_out or not is_win):
            os.makedirs(args.dump_dir, exist_ok=True)
            fname = os.path.join(args.dump_dir, f'episode_{i+1}_final.txt')
            with open(fname, 'w') as f:
                # Write a simple textual board: F=flag, .=hidden, *=mine, 0-8 numbers
                obs_state = env._get_obs()
                for r in range(env.height):
                    row = []
                    for c in range(env.width):
                        v = obs_state[r, c]
                        if v == -3:
                            row.append('F')
                        elif v == -2:
                            row.append('.')
                        elif v == -1:
                            row.append('*')
                        else:
                            row.append(str(int(v)))
                    f.write(' '.join(row) + '\n')
        if (i + 1) % max(1, args.games // 10) == 0:
            print(f"Completed {i+1}/{args.games}")

    duration = time.time() - start
    rewards = [r['reward'] for r in results]
    steps_list = [r['steps'] for r in results]

    print('\nEvaluation summary')
    print(f'  Episodes: {args.games}')
    print(f'  Wins: {wins} ({wins/args.games:.3%})')
    print(f'  Timeouts (hit max-steps without terminal): {timeouts}')
    print(f'  Avg reward: {np.mean(rewards):.3f}  Median: {np.median(rewards):.3f}  Std: {np.std(rewards):.3f}')
    print(f'  Avg steps: {np.mean(steps_list):.1f}  Median steps: {np.median(steps_list):.1f}')
    print(f'  Duration: {duration:.2f}s  Episodes/sec: {args.games/duration:.2f}')

    if args.save_csv:
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['episode', 'reward', 'steps', 'win'])
            writer.writeheader()
            writer.writerows(results)
        print(f'Saved per-episode results to {args.save_csv}')


if __name__ == '__main__':
    main()

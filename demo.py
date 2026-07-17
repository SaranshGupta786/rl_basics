"""Run GridWorld path planning and RL demos."""

from __future__ import annotations

import argparse
from pathlib import Path

from gridworld import GridWorld
from path_planning import a_star, actions_from_path, bfs
from rl_algorithms import extract_greedy_path, q_learning, value_iteration
from visualize import plot_grid, plot_learning_curve


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demo_path_planning(show_plot: bool, output_dir: Path | None) -> None:
    env = GridWorld.classic()
    print_header("Path Planning")
    print("Map:")
    print(env.render())

    bfs_result = bfs(env)
    if bfs_result:
        print(f"\nBFS path length: {len(bfs_result.path) - 1} steps")
        print(f"Expanded nodes: {bfs_result.expanded}")
        print(env.render(bfs_result.path))

    astar_result = a_star(env)
    if astar_result:
        print(f"\nA* path length: {len(astar_result.path) - 1} steps")
        print(f"Expanded nodes: {astar_result.expanded}")
        actions = actions_from_path(astar_result.path)
        print("Actions:", ", ".join(action.name.lower() for action in actions))

    if show_plot and astar_result:
        plot_grid(
            env,
            path=astar_result.path,
            title="A* shortest path",
            save_path=(output_dir / "astar_path.png") if output_dir else None,
        )


def demo_value_iteration(show_plot: bool, output_dir: Path | None) -> None:
    env = GridWorld.classic()
    print_header("Value Iteration")
    result = value_iteration(env)
    print(f"Converged in {result.iterations} iterations")

    path = extract_greedy_path(env, result.policy)
    print(f"Greedy policy path length: {len(path) - 1} steps")
    print(env.render(path))

    top_states = sorted(result.values.items(), key=lambda item: item[1], reverse=True)[:5]
    print("\nTop state values:")
    for state, value in top_states:
        print(f"  {state}: {value:.4f}")

    if show_plot:
        plot_grid(
            env,
            path=path,
            values=result.values,
            title="Value iteration",
            save_path=(output_dir / "value_iteration.png") if output_dir else None,
        )


def demo_q_learning(show_plot: bool, output_dir: Path | None) -> None:
    env = GridWorld.classic()
    print_header("Q-learning")
    result = q_learning(env, episodes=400, alpha=0.5, epsilon=0.2, seed=42)

    path = extract_greedy_path(env, result.policy)
    print(f"Learned policy path length: {len(path) - 1} steps")
    print(env.render(path))
    print(f"Last 10 episode rewards: {[round(r, 3) for r in result.episode_rewards[-10:]]}")

    if show_plot:
        plot_grid(
            env,
            path=path,
            title="Q-learning policy",
            save_path=(output_dir / "qlearning_policy.png") if output_dir else None,
        )
        plot_learning_curve(
            result.episode_rewards,
            save_path=(output_dir / "qlearning_curve.png") if output_dir else None,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="RL basics: GridWorld demos")
    parser.add_argument(
        "demo",
        nargs="?",
        choices=("planning", "value", "qlearning", "all"),
        default="all",
        help="Which demo to run",
    )
    parser.add_argument("--plot", action="store_true", help="Show or save matplotlib visualizations")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots (used in Docker/headless mode)",
    )
    args = parser.parse_args()

    if args.demo in ("planning", "all"):
        demo_path_planning(args.plot, args.output_dir)
    if args.demo in ("value", "all"):
        demo_value_iteration(args.plot, args.output_dir)
    if args.demo in ("qlearning", "all"):
        demo_q_learning(args.plot, args.output_dir)


if __name__ == "__main__":
    main()

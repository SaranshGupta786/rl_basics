"""RL algorithms on GridWorld: value iteration and Q-learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gridworld import Action, GridWorld


@dataclass
class ValueIterationResult:
    values: dict[tuple[int, int], float]
    policy: dict[tuple[int, int], Action]
    iterations: int


@dataclass
class QLearningResult:
    q_table: dict[tuple[int, int], np.ndarray]
    policy: dict[tuple[int, int], Action]
    episode_rewards: list[float]


def value_iteration(
    env: GridWorld,
    *,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_iterations: int = 1000,
) -> ValueIterationResult:
    """Tabular value iteration for the deterministic GridWorld MDP."""
    states = env.in_bounds_states()
    values = {state: 0.0 for state in states}
    policy: dict[tuple[int, int], Action] = {}

    for iteration in range(1, max_iterations + 1):
        delta = 0.0
        for state in states:
            if env.is_terminal(state):
                continue

            best_action = None
            best_value = float("-inf")
            for action in Action:
                next_state, reward, _ = env.transition(state, action)
                value = reward + gamma * values[next_state]
                if value > best_value:
                    best_value = value
                    best_action = action

            delta = max(delta, abs(best_value - values[state]))
            values[state] = best_value
            policy[state] = best_action

        if delta < theta:
            return ValueIterationResult(values=values, policy=policy, iterations=iteration)

    return ValueIterationResult(values=values, policy=policy, iterations=max_iterations)


def extract_greedy_path(env: GridWorld, policy: dict[tuple[int, int], Action]) -> list[tuple[int, int]]:
    """Roll out a deterministic policy until the goal or a loop is detected."""
    path = [env.start_state]
    visited = {env.start_state}
    state = env.start_state

    while not env.is_terminal(state):
        action = policy.get(state)
        if action is None:
            break
        next_state, _, done = env.transition(state, action)
        if next_state in visited:
            break
        path.append(next_state)
        visited.add(next_state)
        state = next_state
        if done:
            break

    return path


def q_learning(
    env: GridWorld,
    *,
    episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 0.99,
    epsilon: float = 0.2,
    seed: int | None = 42,
) -> QLearningResult:
    """Q-learning (off-policy TD control) with epsilon-greedy exploration."""
    rng = np.random.default_rng(seed)
    states = env.in_bounds_states()
    q_table = {state: np.zeros(len(Action), dtype=float) for state in states}
    episode_rewards: list[float] = []

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0

        while True:
            if env.is_terminal(state):
                break

            if rng.random() < epsilon:
                action = Action(rng.integers(0, len(Action)))
            else:
                action = Action(int(np.argmax(q_table[state])))

            step_result = env.step(action)
            next_state = step_result.state
            reward = step_result.reward
            done = step_result.done
            best_next = np.max(q_table[next_state])
            q_table[state][action] += alpha * (reward + gamma * best_next - q_table[state][action])
            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)

    policy = {
        state: Action(int(np.argmax(q_values)))
        for state, q_values in q_table.items()
        if not env.is_terminal(state)
    }
    return QLearningResult(q_table=q_table, policy=policy, episode_rewards=episode_rewards)

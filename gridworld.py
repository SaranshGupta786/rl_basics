"""GridWorld environment for RL basics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np


class Cell(IntEnum):
    EMPTY = 0
    WALL = 1
    START = 2
    GOAL = 3


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTION_NAMES = ("up", "down", "left", "right")
ACTION_DELTAS = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


@dataclass(frozen=True)
class StepResult:
    state: tuple[int, int]
    reward: float
    done: bool
    info: dict


class GridWorld:
    """Simple grid MDP: move on empty cells, avoid walls, reach the goal."""

    def __init__(
        self,
        layout: list[str] | np.ndarray,
        *,
        step_cost: float = -0.04,
        goal_reward: float = 1.0,
        wall_penalty: float = -0.5,
        slip_prob: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.grid = self._parse_layout(layout)
        self.height, self.width = self.grid.shape
        self.step_cost = step_cost
        self.goal_reward = goal_reward
        self.wall_penalty = wall_penalty
        self.slip_prob = slip_prob
        self.rng = np.random.default_rng(seed)

        start = np.argwhere(self.grid == Cell.START)
        goal = np.argwhere(self.grid == Cell.GOAL)
        if len(start) != 1 or len(goal) != 1:
            raise ValueError("Layout must contain exactly one start (S) and one goal (G).")

        self.start_state = tuple(start[0])
        self.goal_state = tuple(goal[0])
        self.state = self.start_state

    @staticmethod
    def _parse_layout(layout: list[str] | np.ndarray) -> np.ndarray:
        symbol_map = {
            ".": Cell.EMPTY,
            "#": Cell.WALL,
            "S": Cell.START,
            "G": Cell.GOAL,
        }
        rows = [list(row) for row in layout]
        grid = np.array([[symbol_map[cell] for cell in row] for row in rows], dtype=np.int8)
        return grid

    @classmethod
    def classic(cls, **kwargs) -> GridWorld:
        layout = [
            "S.......",
            "........",
            "..###...",
            "..#.....",
            "..#..###",
            "........",
            ".....G..",
        ]
        return cls(layout, **kwargs)

    def reset(self) -> tuple[int, int]:
        self.state = self.start_state
        return self.state

    def is_valid(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_wall(self, row: int, col: int) -> bool:
        return self.grid[row, col] == Cell.WALL

    def is_terminal(self, state: tuple[int, int]) -> bool:
        return state == self.goal_state

    def in_bounds_states(self) -> list[tuple[int, int]]:
        states: list[tuple[int, int]] = []
        for row in range(self.height):
            for col in range(self.width):
                if not self.is_wall(row, col):
                    states.append((row, col))
        return states

    def legal_actions(self, state: tuple[int, int] | None = None) -> list[Action]:
        state = self.state if state is None else state
        if self.is_terminal(state):
            return []
        return list(Action)

    def _apply_action(self, state: tuple[int, int], action: Action) -> tuple[int, int]:
        dr, dc = ACTION_DELTAS[action]
        row, col = state
        next_row, next_col = row + dr, col + dc

        if not self.is_valid(next_row, next_col) or self.is_wall(next_row, next_col):
            return state
        return next_row, next_col

    def _sample_action(self, action: Action) -> Action:
        if self.slip_prob <= 0.0:
            return action

        perpendicular = {
            Action.UP: [Action.LEFT, Action.RIGHT],
            Action.DOWN: [Action.LEFT, Action.RIGHT],
            Action.LEFT: [Action.UP, Action.DOWN],
            Action.RIGHT: [Action.UP, Action.DOWN],
        }[action]

        roll = self.rng.random()
        if roll < self.slip_prob:
            return perpendicular[0]
        if roll < 2 * self.slip_prob:
            return perpendicular[1]
        return action

    def step(self, action: Action | int) -> StepResult:
        if self.is_terminal(self.state):
            return StepResult(self.state, 0.0, True, {"message": "Episode already finished."})

        action = Action(action)
        actual_action = self._sample_action(action)
        next_state = self._apply_action(self.state, actual_action)

        if next_state == self.state and actual_action in self.legal_actions(self.state):
            reward = self.wall_penalty
        elif next_state == self.goal_state:
            reward = self.goal_reward
        else:
            reward = self.step_cost

        done = next_state == self.goal_state
        self.state = next_state
        return StepResult(next_state, reward, done, {"requested_action": action, "actual_action": actual_action})

    def transition(self, state: tuple[int, int], action: Action) -> tuple[tuple[int, int], float, bool]:
        """Deterministic transition model for planning algorithms."""
        if self.is_terminal(state):
            return state, 0.0, True

        next_state = self._apply_action(state, action)
        if next_state == state:
            reward = self.wall_penalty
        elif next_state == self.goal_state:
            reward = self.goal_reward
        else:
            reward = self.step_cost
        return next_state, reward, next_state == self.goal_state

    def render(self, path: Iterable[tuple[int, int]] | None = None) -> str:
        path_set = set(path or [])
        symbols = {
            Cell.EMPTY: ".",
            Cell.WALL: "#",
            Cell.START: "S",
            Cell.GOAL: "G",
        }
        lines: list[str] = []
        for row in range(self.height):
            row_chars: list[str] = []
            for col in range(self.width):
                state = (row, col)
                if state in path_set and state not in (self.start_state, self.goal_state):
                    row_chars.append("*")
                elif state == self.state:
                    row_chars.append("A")
                else:
                    row_chars.append(symbols[Cell(self.grid[row, col])])
            lines.append("".join(row_chars))
        return "\n".join(lines)

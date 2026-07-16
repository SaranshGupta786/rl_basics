"""Classical path planning on GridWorld: BFS and A*."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq

from gridworld import Action, ACTION_DELTAS, GridWorld


@dataclass(frozen=True)
class PlanResult:
    path: list[tuple[int, int]]
    cost: float
    expanded: int


def reconstruct_path(came_from: dict[tuple[int, int], tuple[int, int] | None], goal: tuple[int, int]) -> list[tuple[int, int]]:
    path = [goal]
    current = goal
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def bfs(env: GridWorld) -> PlanResult | None:
    """Shortest path in number of steps (unweighted graph)."""
    start = env.start_state
    goal = env.goal_state

    queue: deque[tuple[int, int]] = deque([start])
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    expanded = 0

    while queue:
        state = queue.popleft()
        expanded += 1
        if state == goal:
            path = reconstruct_path(came_from, goal)
            return PlanResult(path=path, cost=len(path) - 1, expanded=expanded)

        row, col = state
        for action in Action:
            dr, dc = ACTION_DELTAS[action]
            nxt = (row + dr, col + dc)
            if not env.is_valid(*nxt) or env.is_wall(*nxt) or nxt in came_from:
                continue
            came_from[nxt] = state
            queue.append(nxt)

    return None


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(env: GridWorld) -> PlanResult | None:
    """Optimal path minimizing step cost with admissible Manhattan heuristic."""
    start = env.start_state
    goal = env.goal_state

    open_set: list[tuple[float, int, tuple[int, int]]] = [(manhattan(start, goal), 0, start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score = {start: 0.0}
    expanded = 0
    counter = 0

    while open_set:
        _, _, state = heapq.heappop(open_set)
        expanded += 1

        if state == goal:
            path = reconstruct_path(came_from, goal)
            return PlanResult(path=path, cost=g_score[goal], expanded=expanded)

        row, col = state
        for action in Action:
            dr, dc = ACTION_DELTAS[action]
            nxt = (row + dr, col + dc)
            if not env.is_valid(*nxt) or env.is_wall(*nxt):
                continue

            move_cost = 1.0
            tentative_g = g_score[state] + move_cost
            if nxt not in g_score or tentative_g < g_score[nxt]:
                came_from[nxt] = state
                g_score[nxt] = tentative_g
                counter += 1
                f_score = tentative_g + manhattan(nxt, goal)
                heapq.heappush(open_set, (f_score, counter, nxt))

    return None


def actions_from_path(path: list[tuple[int, int]]) -> list[Action]:
    """Convert a coordinate path into action sequence."""
    actions: list[Action] = []
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        dr, dc = r2 - r1, c2 - c1
        for action, delta in ACTION_DELTAS.items():
            if delta == (dr, dc):
                actions.append(action)
                break
        else:
            raise ValueError(f"Non-adjacent steps in path: {(r1, c1)} -> {(r2, c2)}")
    return actions

"""Windy Desert environment used throughout Chapter 3."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from mdp import Environment, State

STEP_COST = -0.01
QUICKSAND_PENALTY = -1.0
GOAL_REWARD = +1.0


def create_windy_desert_env(
    layout: Optional[List[str]] = None,
    p_intended: float = 0.70,
    p_perp: float = 0.15,
    gamma: float = 1.0,
) -> Environment:
    """Build the stochastic Windy Desert grid."""
    if not np.isclose(p_intended + 2 * p_perp, 1.0):
        raise ValueError("Need p_intended + 2*p_perp == 1.0 for a valid transition model.")

    if layout is None:
        layout = [
            "S D D D D D D D",
            "D D D Q Q Q D D",
            "D D D D D D D D",
            "D Q D D D D Q D",
            "D D D D D D D D",
            "D D D D D D D O",
        ]
        layout = ["".join(row.split()) for row in layout]

    H, W = len(layout), len(layout[0])

    states: list[State] = []
    start_id = None
    goal_id = None
    quicksand_ids: set[int] = set()

    for i in range(H * W):
        r, c = divmod(i, W)
        ch = layout[r][c]
        if ch == "S":
            start_id = i
            terminal = False
        elif ch == "O":
            goal_id = i
            terminal = True
        elif ch == "Q":
            quicksand_ids.add(i)
            terminal = True
        else:
            terminal = False
        states.append(State(id=i, name=f"S{i}", terminal=terminal))

    if start_id is None or goal_id is None:
        raise ValueError("Layout must contain one 'S' (start) and one 'O' (goal).")

    env = Environment(states, actions=[0, 1, 2, 3], gamma=gamma)
    env.grid_height, env.grid_width = H, W
    env.start_state = states[start_id]
    env.goal_state = states[goal_id]
    env.quicksand_ids = quicksand_ids
    env.p_intended = p_intended
    env.p_perp = p_perp

    def move(r: int, c: int, a: int) -> tuple[int, int]:
        nr, nc = r, c
        if a == 0:
            nr = max(0, r - 1)
        elif a == 1:
            nc = min(W - 1, c + 1)
        elif a == 2:
            nr = min(H - 1, r + 1)
        elif a == 3:
            nc = max(0, c - 1)
        return nr, nc

    perp_actions = {
        0: [3, 1],
        1: [0, 2],
        2: [3, 1],
        3: [0, 2],
    }

    for s in states:
        if s.terminal:
            continue
        r, c = divmod(s.id, W)
        for a in env.actions:
            candidates = [(a, p_intended)] + [(pa, p_perp) for pa in perp_actions[a]]
            next_prob: dict[int, float] = {}
            for a2, p in candidates:
                nr, nc = move(r, c, a2)
                nid = nr * W + nc
                next_prob[nid] = next_prob.get(nid, 0.0) + p

            next_states_probs = {states[nid]: prob for nid, prob in next_prob.items()}
            env.set_transition_probs(s, a, next_states_probs)

            for nxt_state in next_states_probs:
                if nxt_state.id == goal_id:
                    rew = GOAL_REWARD
                elif nxt_state.id in quicksand_ids:
                    rew = QUICKSAND_PENALTY
                else:
                    rew = STEP_COST
                env.set_reward(s, a, nxt_state, rew)

    return env


def apply_reward_shaping(
    env: Environment,
    step_cost: float = -1.0,
    hole_penalty: float = -100.0,
    goal_reward: float = +10.0,
) -> None:
    """Overwrite the reward table with the shaped-reward version used for control."""
    env.rewards.clear()
    holes = getattr(env, "quicksand_ids", set())
    for (sid, a), nexts in env.transitions.items():
        for nxt in nexts:
            if nxt.id == env.goal_state.id:
                r = goal_reward
            elif nxt.id in holes:
                r = hole_penalty
            else:
                r = step_cost
            env.rewards[(sid, a, nxt.id)] = r

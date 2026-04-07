"""
Windy Desert environment used in Chapter 3.

This file is package-based and expects the existing Chapter 2 MDP primitives:
    from rlbook.core.mdp import State, Environment
"""

from typing import Dict, List, Optional

from rlbook.core.mdp import Environment, State

STEP_COST = -0.01
QUICKSAND_PENALTY = -1.0
GOAL_REWARD = +1.0


def create_windy_desert_env(
    layout: Optional[List[str]] = None,
    p_intended: float = 0.70,
    p_perp: float = 0.15,
    gamma: float = 1.0,
) -> Environment:
    """
    Build a stochastic grid world with quicksand traps and wind drift.

    Cells:
        S = start
        D = dry sand
        Q = quicksand (terminal)
        O = oasis / goal (terminal)

    Actions:
        0 = up, 1 = right, 2 = down, 3 = left
    """
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

    states: List[State] = []
    start_id = None
    goal_id = None
    quicksand_ids: set[int] = set()

    for sid in range(H * W):
        r, c = divmod(sid, W)
        cell = layout[r][c]
        if cell == "S":
            start_id = sid
            terminal = False
        elif cell == "O":
            goal_id = sid
            terminal = True
        elif cell == "Q":
            quicksand_ids.add(sid)
            terminal = True
        else:
            terminal = False
        states.append(State(id=sid, name=f"S{sid}", terminal=terminal))

    if start_id is None or goal_id is None:
        raise ValueError("Layout must contain exactly one 'S' and one 'O'.")

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

    for state in states:
        if state.terminal:
            continue

        r, c = divmod(state.id, W)
        for action in env.actions:
            candidates = [(action, p_intended)] + [(a2, p_perp) for a2 in perp_actions[action]]

            next_prob: Dict[int, float] = {}
            for candidate_action, prob in candidates:
                nr, nc = move(r, c, candidate_action)
                next_id = nr * W + nc
                next_prob[next_id] = next_prob.get(next_id, 0.0) + prob

            next_states_probs = {states[next_id]: prob for next_id, prob in next_prob.items()}
            env.set_transition_probs(state, action, next_states_probs)

            for next_state in next_states_probs:
                if next_state.id == goal_id:
                    reward = GOAL_REWARD
                elif next_state.id in quicksand_ids:
                    reward = QUICKSAND_PENALTY
                else:
                    reward = STEP_COST
                env.set_reward(state, action, next_state, reward)

    return env


def apply_reward_shaping(
    env: Environment,
    step_cost: float = -1.0,
    hole_penalty: float = -100.0,
    goal_reward: float = +10.0,
) -> None:
    """
    Overwrite all rewards using the environment's goal and quicksand metadata.
    """
    env.rewards.clear()
    quicksand_ids = getattr(env, "quicksand_ids", set())

    for (state_id, action), next_states in env.transitions.items():
        for next_state in next_states:
            if next_state.id == env.goal_state.id:
                reward = goal_reward
            elif next_state.id in quicksand_ids:
                reward = hole_penalty
            else:
                reward = step_cost
            env.rewards[(state_id, action, next_state.id)] = reward

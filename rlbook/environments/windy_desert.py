"""
Windy Desert Environment for Chapter 3 case studies.

A stochastic grid world with quicksand traps and wind drift.
See Chapter 3 (Classical Reinforcement Learning Algorithms) for
the complete description and case studies using this environment.
"""

from typing import List, Dict, Tuple, Optional
from rlbook.core.mdp import State, Environment

# Default reward constants (used in Chapter 3)
STEP_COST = -0.01
QUICKSAND_PENALTY = -1.0
GOAL_REWARD = +1.0


def create_windy_desert_env(
    layout: Optional[List[str]] = None,
    p_intended: float = 0.70,
    p_perp: float = 0.15,
    gamma: float = 1.0
) -> Environment:
    """
    Build a Windy Desert grid with stochastic drift.
    - Cells: 'S' (start), 'D' (dry sand), 'Q' (quicksand terminal), 'O' (oasis goal terminal)
    - Actions: 0=Up, 1=Right, 2=Down, 3=Left
    - Stochasticity: with p_intended go intended way, else drift to each perpendicular with p_perp
    - Rewards: +1 on reaching 'O'; 0 otherwise. Quicksand is terminal with 0.
    """
    # Default 6x8 map if none provided
    # S and O are unique; Q are traps; D is safe
    if layout is None:
        layout = [
    "S D D D D D D D",
    "D D D Q Q Q D D",  # same cliff under top row
    "D D D D D D D D",
    "D Q D D D D Q D",  # two mid-row holes create mild jogs
    "D D D D D D D D",
    "D D D D D D D O",
]
        layout = ["".join(row.split()) for row in layout]  # strip spaces

    H, W = len(layout), len(layout[0])
    # Build states
    states: List[State] = []
    start_id = None
    goal_id = None
    quicksand_ids: set[int] = set()

    for i in range(H * W):
        r, c = divmod(i, W)
        ch = layout[r][c]
        if ch == 'S':
            start_id = i
            terminal = False
        elif ch == 'O':
            goal_id = i
            terminal = True
        elif ch == 'Q':
            quicksand_ids.add(i)
            terminal = True
        else:  # 'D'
            terminal = False
        states.append(State(id=i, name=f"S{i}", terminal=terminal))

    if start_id is None or goal_id is None:
        raise ValueError("Layout must contain one 'S' (start) and one 'O' (goal).")

    actions = [0, 1, 2, 3]  # Up, Right, Down, Left
    env = Environment(states, actions, gamma=gamma)
    env.grid_height, env.grid_width = H, W
    env.start_state = states[start_id]
    env.goal_state = states[goal_id]
    env.quicksand_ids = quicksand_ids

    env.p_intended = p_intended
    env.p_perp = p_perp

    def in_bounds(rr, cc):
        return 0 <= rr < H and 0 <= cc < W

    def move(r, c, a):
        nr, nc = r, c
        if a == 0: nr = max(0, r-1)
        elif a == 1: nc = min(W-1, c+1)
        elif a == 2: nr = min(H-1, r+1)
        elif a == 3: nc = max(0, c-1)
        return nr, nc

    # Perpendicular action pairs relative to intended action
    # Up <-> Down are vertical; perpendiculars are Left and Right, etc.
    perp_actions = {
        0: [3, 1],  # Up -> Left or Right
        1: [0, 2],  # Right -> Up or Down
        2: [3, 1],  # Down -> Left or Right
        3: [0, 2],  # Left -> Up or Down
    }

    # Set transitions and rewards
    for s in states:
        if s.terminal:
            continue  # no outgoing transitions from terminal states
        r, c = divmod(s.id, W)
        ch = layout[r][c]
        for a in actions:
            # candidates: intended + two perpendiculars
            candidates = [(a, p_intended)] + [(pa, p_perp) for pa in perp_actions[a]]

            next_prob: Dict[int, float] = {}
            for a2, p in candidates:
                nr, nc = move(r, c, a2)
                # If bounce into wall, the "move" is effectively staying put (already handled by clamp)
                nid = nr * W + nc
                next_prob[nid] = next_prob.get(nid, 0.0) + p

            # Convert to state:prob
            next_states_probs = {states[nid]: prob for nid, prob in next_prob.items()}
            env.set_transition_probs(s, a, next_states_probs)

            # Rewards: +1 if next is goal, else 0
            for nxt_state, prob in next_states_probs.items():
              if nxt_state.id == goal_id:
                rew = GOAL_REWARD
              elif nxt_state.id in quicksand_ids:
                rew = QUICKSAND_PENALTY
              else:
                rew = STEP_COST
              env.set_reward(s, a, nxt_state, rew)

    return env


def apply_reward_shaping(env, step_cost=-1.0, hole_penalty=-100.0, goal_reward=+10.0):
    """
    Overwrite env.rewards for every (s,a->s') using env.goal_state and env.quicksand_ids.
    Works with your Windy Desert (stochastic drift).
    """
    env.rewards.clear()
    holes = getattr(env, "quicksand_ids", set())
    for (sid, a), nexts in env.transitions.items():
        for nxt, _p in nexts.items():
            if nxt.id == env.goal_state.id:  r = goal_reward
            elif nxt.id in holes:            r = hole_penalty
            else:                             r = step_cost
            env.rewards[(sid, a, nxt.id)] = r

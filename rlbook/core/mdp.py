"""
MDP Framework: State, Environment, and Agent Classes.

These classes implement the core Markov Decision Process abstraction
used throughout the textbook. See Chapter 2 (Fundamentals of
Reinforcement Learning) for the mathematical foundations.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass(eq=True, frozen=True)  # Make State hashable
class State:
    """
    Representation of a state in an MDP.

    Attributes:
        id: Unique identifier for the state
        name: Optional human-readable name
        terminal: Whether this is a terminal state
    """
    id: int
    name: str = None
    terminal: bool = False

    def __repr__(self):
        return f"State({self.id}, name={self.name}, terminal={self.terminal})"

class Environment:
    """
    Base class for a Markov Decision Process environment.

    Attributes:
        states: List of all possible states
        actions: List of all possible actions
        current_state: The current state of the environment
        transitions: Dictionary mapping (state, action) pairs to next state probabilities
        rewards: Dictionary mapping (state, action, next_state) triplets to rewards
        gamma: Discount factor for future rewards
    """
    def __init__(self, states: List[State], actions: List[int], gamma: float = 0.9):
        self.states = states
        self.actions = actions
        self.current_state = None
        self.transitions = {}  # (state, action) -> {next_state: probability}
        self.rewards = {}  # (state, action, next_state) -> reward
        self.gamma = gamma

    def set_transition_probs(self, state: State, action: int, next_states_probs: Dict[State, float]):
        """
        Define transition probabilities for a given state-action pair.

        Args:
            state: The current state
            action: The action taken
            next_states_probs: Dictionary mapping next states to their probabilities
        """
        if abs(sum(next_states_probs.values()) - 1.0) > 1e-6:
            raise ValueError(f"Transition probabilities must sum to 1, got {sum(next_states_probs.values())}")

        self.transitions[(state.id, action)] = next_states_probs

    def set_reward(self, state: State, action: int, next_state: State, reward: float):
        """
        Define the reward for a state-action-next_state triplet.

        Args:
            state: The current state
            action: The action taken
            next_state: The resulting state
            reward: The reward value
        """
        self.rewards[(state.id, action, next_state.id)] = reward

    def reset(self, start_state: State = None) -> State:
        """
        Reset the environment to an initial state.

        Args:
            start_state: Optional specific starting state

        Returns:
            The initial state
        """
        if start_state is None:
            # Default to first non-terminal state
            start_state = next(s for s in self.states if not s.terminal)

        self.current_state = start_state
        return self.current_state

    def step(self, action: int) -> Tuple[State, float, bool]:
        """
        Take an action in the current state and return the result.

        Args:
            action: The action to take

        Returns:
            Tuple of (next_state, reward, done)
        """
        if self.current_state.terminal:
            return self.current_state, 0, True

        # Get transition probabilities for current state-action pair
        if (self.current_state.id, action) not in self.transitions:
            raise ValueError(f"No transition defined for state {self.current_state} and action {action}")

        next_states_probs = self.transitions[(self.current_state.id, action)]

        # Sample next state based on probabilities
        next_states = list(next_states_probs.keys())
        probs = list(next_states_probs.values())

        # Convert state objects to their IDs for the random choice
        next_state_ids = [s.id for s in next_states]
        next_state_id = np.random.choice(next_state_ids, p=probs)

        # Find the corresponding state object
        next_state = next(s for s in self.states if s.id == next_state_id)

        # Get reward
        reward = self.rewards.get((self.current_state.id, action, next_state.id), 0)

        # Update current state
        self.current_state = next_state
        done = next_state.terminal

        return next_state, reward, done

class Agent:
    """
    Base class for an agent in an MDP.

    Attributes:
        env: The environment the agent interacts with
        value_function: Dictionary mapping states to values
        policy: Dictionary mapping states to action probabilities
    """
    def __init__(self, env: Environment):
        self.env = env
        self.value_function = {state.id: 0 for state in env.states}
        self.policy = {state.id: {a: 1.0/len(env.actions) for a in env.actions}
                      for state in env.states}

    def choose_action(self, state: State) -> int:
        """
        Select an action based on the current policy.

        Args:
            state: The current state

        Returns:
            The chosen action
        """
        action_probs = self.policy[state.id]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())

        return np.random.choice(actions, p=probs)

    def update_policy(self, state: State, action: int, probability: float):
        """
        Update the policy for a given state-action pair.

        Args:
            state: The state
            action: The action
            probability: The new probability of taking this action
        """
        self.policy[state.id][action] = probability

from rlbook.environments.grid_world import create_grid_world
from rlbook.environments.windy_desert import create_windy_desert_env, apply_reward_shaping

# CartPole requires gymnasium — import only when needed
try:
    from rlbook.environments.cartpole_wrapper import CartPoleWrapper, set_global_seed
except ImportError:
    pass

__all__ = ["create_grid_world", "create_windy_desert_env", "apply_reward_shaping"]

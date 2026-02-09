# Deep RL algorithms require torch — import only when needed
try:
    from rlbook.algorithms.deep.dqn import QNet, DQNConfig, DQNAgent
    from rlbook.algorithms.deep.reinforce import (
        PolicyNet, compute_returns, reinforce_update,
        sample_action_and_logp, evaluate,
    )
    from rlbook.algorithms.deep.actor_critic import Critic, actor_critic_update
    from rlbook.algorithms.deep.runners import RunConfig, run_reinforce, run_actor_critic, run_dqn
except ImportError:
    pass

from rlbook.algorithms.agentic.email_triage import (
    CustomerMemorySystem, Level1RandomAgent, Level2ClassificationAgent
)
from rlbook.algorithms.agentic.scenarios import EMAIL_SCENARIOS

# These require torch + transformers + gymnasium — import only when needed
try:
    from rlbook.algorithms.agentic.support_agent import SupportAgent
    from rlbook.algorithms.agentic.environment import (
        EmailTriageEnvProgressive, evaluate_rl_model
    )
except ImportError:
    pass

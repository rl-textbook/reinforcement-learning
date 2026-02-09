"""Progressive Agentic Email Triage with RL-Learned Tool Use.

Demonstrates the 6-level progressive architecture from Chapter 7:
  Level 1: Random baseline
  Level 2: + LLM classification
  Level 3: + RL-learned tool use
  Level 4: + Action history
  Level 5: + LLM planning guidance
  Level 6: + Persistent customer memory

Requires: pip install -e '.[agentic]'
Runtime:  ~5-10 min on CPU
"""

import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from transformers import AutoTokenizer, AutoModelForCausalLM
from stable_baselines3 import PPO

from rlbook.algorithms.agentic.email_triage import (
    CustomerMemorySystem,
    Level1RandomAgent,
    Level2ClassificationAgent,
)


# ── LLM Support Agent ──────────────────────────────────────────────
class SupportAgent:
    """LLM agent for email analysis."""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Loading GPT-2 on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained('gpt2').to(self.device)
        self.model.eval()

    def classify_urgency(self, email: str) -> str:
        """Simple urgency classification."""
        email_lower = email.lower()

        urgent_keywords = ['furious', 'urgent', 'immediately', 'asap', 'angry', 'terrible', '3rd time', 'again']
        if any(kw in email_lower for kw in urgent_keywords):
            return 'high'

        return 'normal'

    def classify_category(self, email: str) -> str:
        """Classify email category."""
        email_lower = email.lower()

        if any(w in email_lower for w in ['refund', 'money back', 'charged']):
            return 'refund'
        elif any(w in email_lower for w in ['damaged', 'defective', 'wrong', 'broken']):
            return 'replacement'
        elif any(w in email_lower for w in ['cancel', 'subscription']):
            return 'cancellation'
        elif any(w in email_lower for w in ['arrived', 'shipping', 'tracking', 'delivery']):
            return 'shipping'
        elif any(w in email_lower for w in ['fraud', 'hacked', 'unauthorized']):
            return 'fraud'
        else:
            return 'general'

    def suggest_initial_action(self, email: str) -> int:
        """Suggest initial action based on classification."""
        category = self.classify_category(email)

        category_actions = {
            'refund': 3,  # search_orders
            'replacement': 4,  # check_shipping
            'cancellation': 5,  # check_subscription
            'shipping': 4,  # check_shipping
            'fraud': 2,  # fraud_check
            'general': 0,  # resolve
        }

        return category_actions.get(category, 0)

    def generate_plan(self, email: str, max_new_tokens: int = 50) -> str:
        """Generate investigation plan."""
        prompt = f"""Email: {email}

Investigation plan:"""

        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        plan = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return plan.split('Investigation plan:')[-1].strip()


agent = SupportAgent()
print("✅ LLM agent ready")


# ── Email Scenarios ─────────────────────────────────────────────────
EMAIL_SCENARIOS = [
    {
        'id': 0,
        'customer_id': 'C001',
        'email': "I'm furious! My order #1234 never arrived and I need it urgently!",
        'truth': {'status': 'shipped_delayed', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [3, 4, 6],
        'is_repeat': False,
    },
    {
        'id': 1,
        'customer_id': 'C002',
        'email': "The product arrived damaged. Can I get a replacement?",
        'truth': {'status': 'delivered', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [4, 7],
        'is_repeat': False,
    },
    {
        'id': 2,
        'customer_id': 'C003',
        'email': "I want to cancel my subscription immediately.",
        'truth': {'status': 'active', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [5, 8],
        'is_repeat': False,
    },
    {
        'id': 3,
        'customer_id': 'C004',
        'email': "Charged twice for the same order! Fix this now!",
        'truth': {'status': 'duplicate_charge', 'payment': 'double_charged', 'fraud_risk': 'low'},
        'optimal_actions': [3, 6],
        'is_repeat': False,
    },
    {
        'id': 4,
        'customer_id': 'C005',
        'email': "My account was hacked and someone made unauthorized purchases.",
        'truth': {'status': 'fraudulent', 'payment': 'completed', 'fraud_risk': 'high'},
        'optimal_actions': [2, 3, 10],
        'is_repeat': False,
    },
    {
        'id': 5,
        'customer_id': 'C006',
        'email': "Great service! Just wanted to say thanks.",
        'truth': {'status': 'satisfied', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [0],
        'is_repeat': False,
    },
    {
        'id': 6,
        'customer_id': 'C007',
        'email': "The product description was misleading. I want my money back.",
        'truth': {'status': 'mismatch', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [3, 1, 6],
        'is_repeat': False,
    },
    {
        'id': 7,
        'customer_id': 'C008',
        'email': "I received someone else's order by mistake.",
        'truth': {'status': 'wrong_item', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [3, 4, 7],
        'is_repeat': False,
    },
    {
        'id': 8,
        'customer_id': 'C009',
        'email': "Where is my refund? It's been 3 weeks!",
        'truth': {'status': 'refund_pending', 'payment': 'refund_initiated', 'fraud_risk': 'low'},
        'optimal_actions': [3, 6],
        'is_repeat': False,
    },
    {
        'id': 9,
        'customer_id': 'C010',
        'email': "The item is defective and stopped working after 2 days.",
        'truth': {'status': 'defective', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [1, 7],
        'is_repeat': False,
    },
    # REPEAT CUSTOMER SCENARIOS
    {
        'id': 10,
        'customer_id': 'C001',  # Same as scenario 0
        'email': "STILL waiting for my order! This is unacceptable!",
        'truth': {'status': 'shipped_delayed', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [9, 3, 4, 6],  # Should check memory first!
        'is_repeat': True,
    },
    {
        'id': 11,
        'customer_id': 'C002',  # Same as scenario 1
        'email': "The replacement you sent is ALSO damaged! What's going on?",
        'truth': {'status': 'delivered', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [9, 4, 7],  # Memory + check shipping + replacement
        'is_repeat': True,
    },
    {
        'id': 12,
        'customer_id': 'C004',  # Same as scenario 3
        'email': "I'm still seeing the duplicate charge on my statement!",
        'truth': {'status': 'duplicate_charge', 'payment': 'double_charged', 'fraud_risk': 'low'},
        'optimal_actions': [9, 3, 6],  # Memory first!
        'is_repeat': True,
    },
    {
        'id': 13,
        'customer_id': 'C011',
        'email': "I've been a loyal customer for 5 years, but this experience is terrible.",
        'truth': {'status': 'vip_complaint', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [3, 0],
        'is_repeat': False,
    },
    {
        'id': 14,
        'customer_id': 'C012',
        'email': "The coupon code doesn't work. Can you apply it manually?",
        'truth': {'status': 'promo_issue', 'payment': 'pending', 'fraud_risk': 'low'},
        'optimal_actions': [3, 0],
        'is_repeat': False,
    },
    {
        'id': 15,
        'customer_id': 'C013',
        'email': "I returned the item but haven't received credit yet.",
        'truth': {'status': 'return_pending', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [3, 4, 6],
        'is_repeat': False,
    },
    {
        'id': 16,
        'customer_id': 'C014',
        'email': "This is my 3rd time contacting you about the same issue!",
        'truth': {'status': 'repeat_issue', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [9, 3, 0],  # Memory critical here!
        'is_repeat': True,
    },
    {
        'id': 17,
        'customer_id': 'C015',
        'email': "I need an invoice for my company's accounting.",
        'truth': {'status': 'invoice_request', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [3, 0],
        'is_repeat': False,
    },
    {
        'id': 18,
        'customer_id': 'C016',
        'email': "The tracking shows delivered but I never received the package.",
        'truth': {'status': 'lost_package', 'payment': 'completed', 'fraud_risk': 'medium'},
        'optimal_actions': [3, 4, 2, 7],
        'is_repeat': False,
    },
    {
        'id': 19,
        'customer_id': 'C017',
        'email': "I accidentally ordered the wrong size. Can I exchange it?",
        'truth': {'status': 'exchange_needed', 'payment': 'completed', 'fraud_risk': 'low'},
        'optimal_actions': [3, 1, 7],
        'is_repeat': False,
    },
]

print(f"✅ Loaded {len(EMAIL_SCENARIOS)} scenarios")
repeat_count = sum(1 for s in EMAIL_SCENARIOS if s.get('is_repeat', False))
print(f"   {repeat_count} repeat customer scenarios (memory valuable here!)")


# ── Pre-generate investigation plans ───────────────────────────────
print("🧠 Pre-generating investigation plans...")
INVESTIGATION_PLANS = {}

for scenario in EMAIL_SCENARIOS:
    plan = agent.generate_plan(scenario['email'])
    INVESTIGATION_PLANS[scenario['id']] = plan

print(f"✅ Generated {len(INVESTIGATION_PLANS)} plans")


# ── Global memory ──────────────────────────────────────────────────
GLOBAL_MEMORY = CustomerMemorySystem()


# ── Progressive Environment ─────────────────────────────────────────
class EmailTriageEnvProgressive(gym.Env):
    """Environment supporting progressive capability levels with RL-learned tool use.

    Following ToolRL (Zeng et al., 2024) paradigm, tools are discrete actions
    in the RL policy space. The agent learns through PPO:
    - Which tools to invoke for each scenario
    - Optimal tool sequencing
    - When tool use is necessary vs. when to act directly
    """

    ACTION_NAMES = [
        'resolve',           # 0: Final action (close ticket)
        'check_product',     # 1-10: Investigation tools (RL learns usage)
        'fraud_check',
        'search_orders',
        'check_shipping',
        'check_subscription',
        'refund',
        'replacement',
        'cancel_subscription',
        'access_memory',
        'security_lock'
    ]

    def __init__(self, level: int = 6, memory_system=None):
        super().__init__()

        self.level = level
        self.memory_system = memory_system if memory_system else GLOBAL_MEMORY

        # Observation size varies by level
        obs_dims = {
            1: 11,   # Level 1: Random (tools exist but not learned)
            2: 13,   # Level 2: LLM classification (tools available but random usage)
            3: 24,   # Level 3: RL learns tool usage
            4: 35,   # Level 4: + action history (RL learns tool patterns)
            5: 55,   # Level 5: + LLM guidance (RL learns with semantic priors)
            6: 60,   # Level 6: + memory (RL learns memory tool usage)
        }

        obs_size = obs_dims[level]
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(obs_size,), dtype=np.float32
        )

        # All 11 actions available at all levels (but only RL learns to use tools effectively)
        self.action_space = spaces.Discrete(len(self.ACTION_NAMES))

        self.current_scenario = None
        self.current_plan = {}
        self.tools_used = set()
        self.action_history = []
        self.steps_taken = 0
        self.max_steps = 5
        self.accessed_memory = False

    def _encode_plan(self, plan_text: str) -> np.ndarray:
        """Encode LLM plan for RL observation."""
        keywords = ['check', 'verify', 'search', 'order', 'shipping', 'refund',
                   'replace', 'fraud', 'cancel', 'subscription', 'product',
                   'payment', 'status', 'customer', 'urgent', 'angry',
                   'damaged', 'wrong', 'delay', 'issue']

        encoding = np.zeros(20)
        plan_lower = plan_text.lower()

        for i, keyword in enumerate(keywords):
            encoding[i] = min(plan_lower.count(keyword), 5) / 5.0

        return encoding

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_scenario = random.choice(EMAIL_SCENARIOS)
        self.customer_id = self.current_scenario['customer_id']

        # Level 2+: LLM classification
        if self.level >= 2:
            self.urgency = 1.0 if agent.classify_urgency(self.current_scenario['email']) == 'high' else 0.0
            categories = ['general', 'refund', 'replacement', 'cancellation', 'shipping', 'fraud']
            category = agent.classify_category(self.current_scenario['email'])
            self.category_encoding = categories.index(category) / len(categories)

        # Level 5+: LLM planning (guides RL tool learning)
        if self.level >= 5:
            plan_text = INVESTIGATION_PLANS[self.current_scenario['id']]
            self.plan_encoding = self._encode_plan(plan_text)

            # Parse plan for tool suggestions (RL gets bonuses for exploring these)
            self.current_plan = {}
            plan_lower = plan_text.lower()
            action_keywords = {
                3: ['order', 'search'],
                4: ['shipping', 'track'],
                6: ['refund'],
                2: ['fraud'],
                7: ['replace'],
                9: ['history', 'previous', 'before', 'again'],
            }
            for action_id, keywords in action_keywords.items():
                if any(kw in plan_lower for kw in keywords):
                    self.current_plan[action_id] = 0.1  # Small exploration bonus

        # Level 6: Memory encoding
        if self.level >= 6:
            self.memory_encoding = self.memory_system.encode_history(self.customer_id)
            self.customer_history = self.memory_system.get_customer_history(self.customer_id)

        # Reset episode state
        self.tools_used = set()
        self.action_history = []
        self.steps_taken = 0
        self.accessed_memory = False

        return self._get_observation(), {}

    def _get_observation(self):
        """Build observation based on capability level."""
        obs_parts = []

        # Level 1: Basic (minimal state)
        obs_parts.append(np.zeros(11))

        # Level 2: + LLM Classification
        if self.level >= 2:
            obs_parts.append(np.array([self.urgency, self.category_encoding]))

        # Level 3: + Tools used (RL learns to track tool usage)
        if self.level >= 3:
            tools_vector = np.zeros(11)
            for tool in self.tools_used:
                tools_vector[tool] = 1.0
            obs_parts.append(tools_vector)

        # Level 4: + Action history (RL learns action patterns)
        if self.level >= 4:
            action_vector = np.zeros(11)
            for a in self.action_history:
                action_vector[a] = min(action_vector[a] + 0.2, 1.0)
            obs_parts.append(action_vector)

        # Level 5: + LLM Plan (RL learns with semantic guidance)
        if self.level >= 5:
            obs_parts.append(self.plan_encoding)

        # Level 6: + Memory (RL learns memory tool usage)
        if self.level >= 6:
            obs_parts.append(self.memory_encoding)

        return np.concatenate(obs_parts).astype(np.float32)

    def step(self, action: int):
        """Execute action and return reward.

        Tool actions (1-10) are learned by RL through reward signal.
        Agent discovers:
        - Which tools are relevant (reward for optimal tools)
        - Tool sequencing (dependencies learned through experience)
        - Investigation efficiency (penalty for irrelevant tools)
        """
        self.steps_taken += 1
        self.action_history.append(action)

        reward = 0
        terminated = False

        # RL LEARNS TOOL USE: Tool actions (1-10)
        if action in range(1, 11):  # Track tool usage at all levels
            self.tools_used.add(action)

            # Reward signal for RL to learn tool relevance
            # This is how RL learns which tools matter
            optimal_tools = set(self.current_scenario['optimal_actions'])
            if action in optimal_tools:
                reward += 0.05  # Small positive reward for relevant tool

            # Level 5+: LLM planning guidance (accelerates RL tool learning)
            if self.level >= 5 and action in self.current_plan:
                reward += self.current_plan[action]  # Exploration bonus

            # Level 6: Memory tool learning
            if self.level >= 6 and action == 9:  # access_memory
                self.accessed_memory = True

                # RL learns when memory tool is valuable
                if self.current_scenario.get('is_repeat', False):
                    reward += 0.3  # High value for repeat customers
                elif self.customer_history['is_repeat']:
                    reward += 0.15  # Medium value for any repeat

        # Resolution action (0) or max steps reached
        if action == 0 or self.steps_taken >= self.max_steps:
            terminated = True

            # Evaluate learned tool usage
            optimal = set(self.current_scenario['optimal_actions'])
            used = self.tools_used

            if len(optimal) == 0:
                quality = 1.0 if action == 0 else 0.0
            else:
                relevant = used & optimal
                irrelevant = used - optimal

                # RL learns precision: use right tools
                precision = len(relevant) / len(used) if used else 0
                # RL learns recall: use all necessary tools
                recall = len(relevant) / len(optimal) if optimal else 0

                quality = (precision + recall) / 2

                # Penalty for irrelevant tools (RL learns efficiency)
                quality -= 0.1 * len(irrelevant)

            # Level 6: Penalty for ignoring memory tool when needed
            if self.level >= 6:
                if self.current_scenario.get('is_repeat', False) and not self.accessed_memory:
                    quality -= 0.2  # RL learns to check memory on repeats

            reward += max(0, quality)

            # Record interaction in memory (for future RL learning)
            if self.level >= 6:
                outcome = 'success' if quality > 0.5 else 'failure'
                self.memory_system.record_interaction(
                    self.customer_id,
                    self.current_scenario['id'],
                    list(self.action_history),
                    outcome
                )

        obs = self._get_observation()
        return obs, reward, terminated, False, {}


print("✅ Environment defined with explicit RL-learned tool use")


# ====================================================================
# TRAINING
# ====================================================================
print("\n🚀 Training Progressive Levels...")
print("=" * 60)

models = {}

# Level 3: Tool use + RL
print("\n[Level 3] Training: LLM Classification + Tools + RL Learning...")
env_l3 = EmailTriageEnvProgressive(level=3)
models[3] = PPO('MlpPolicy', env_l3, learning_rate=3e-4, verbose=0)
models[3].learn(total_timesteps=40000)
print("✅ Level 3 complete")

# Level 4: + Action history
print("\n[Level 4] Training: + Action History for RL...")
env_l4 = EmailTriageEnvProgressive(level=4)
models[4] = PPO('MlpPolicy', env_l4, learning_rate=3e-4, verbose=0)
models[4].learn(total_timesteps=40000)
print("✅ Level 4 complete")

# Level 5: + Planning
print("\n[Level 5] Training: + LLM Planning Guidance...")
env_l5 = EmailTriageEnvProgressive(level=5)
models[5] = PPO('MlpPolicy', env_l5, learning_rate=3e-4, verbose=0)
models[5].learn(total_timesteps=50000)
print("✅ Level 5 complete")

# Level 6: + FIXED Memory
print("\n[Level 6] Training: + Persistent Customer Memory...")
# Reset global memory before training
GLOBAL_MEMORY = CustomerMemorySystem()
env_l6 = EmailTriageEnvProgressive(level=6, memory_system=GLOBAL_MEMORY)
models[6] = PPO('MlpPolicy', env_l6, learning_rate=3e-4, verbose=0)
models[6].learn(total_timesteps=50000)
print("✅ Level 6 complete (memory now valuable!)")

print("\n✅ All training complete!")


# ====================================================================
# EVALUATION
# ====================================================================
def evaluate_rl_model(model, env, num_episodes=100):
    """Evaluate trained RL model."""
    results = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        terminated = False
        accessed_memory = False
        is_repeat = env.current_scenario.get('is_repeat', False)

        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            if int(action) == 9:
                accessed_memory = True
            obs, reward, terminated, truncated, _ = env.step(int(action))
            episode_reward += reward

        results.append({
            'reward': episode_reward,
            'tools_used': len(env.tools_used),
            'accessed_memory': accessed_memory,
            'is_repeat': is_repeat
        })

    # Calculate memory usage stats
    repeat_scenarios = [r for r in results if r['is_repeat']]
    memory_on_repeats = sum(1 for r in repeat_scenarios if r['accessed_memory'])

    return {
        'mean_reward': np.mean([r['reward'] for r in results]),
        'success_rate': np.mean([r['reward'] > 0.5 for r in results]) * 100,
        'mean_tools': np.mean([r['tools_used'] for r in results]),
        'memory_usage': sum(1 for r in results if r['accessed_memory']) / len(results) * 100,
        'memory_on_repeats': memory_on_repeats / len(repeat_scenarios) * 100 if repeat_scenarios else 0
    }


print("\n📊 PROGRESSIVE EVALUATION (FIXED MEMORY)")
print("=" * 60)

results = {}

# Level 1: Random
print("\nEvaluating Level 1: Random Baseline...")
env_l1 = EmailTriageEnvProgressive(level=1)
agent_l1 = Level1RandomAgent(env_l1)
results[1] = agent_l1.evaluate(100)
results[1]['memory_usage'] = 0
results[1]['memory_on_repeats'] = 0

# Level 2: Classification
print("Evaluating Level 2: + LLM Classification...")
env_l2 = EmailTriageEnvProgressive(level=2)
agent_l2 = Level2ClassificationAgent(env_l2, agent)
results[2] = agent_l2.evaluate(100)
results[2]['memory_usage'] = 0
results[2]['memory_on_repeats'] = 0

# Level 3-6: RL models
for level in [3, 4, 5]:
    print(f"Evaluating Level {level}...")
    env = EmailTriageEnvProgressive(level=level)
    results[level] = evaluate_rl_model(models[level], env, 100)

# Level 6 with fresh memory system for evaluation
print("Evaluating Level 6 (with persistent memory)...")
EVAL_MEMORY = CustomerMemorySystem()
env_l6 = EmailTriageEnvProgressive(level=6, memory_system=EVAL_MEMORY)
results[6] = evaluate_rl_model(models[6], env_l6, 100)

print("\n" + "=" * 60)
print("PROGRESSIVE COMPLEXITY RESULTS (ENHANCED)")
print("=" * 60)

capabilities = [
    "Random Baseline",
    "+ LLM Classification",
    "+ Tool Use",
    "+ RL Learning",
    "+ Planning Guidance",
    "+ Persistent Memory"
]

for level in range(1, 7):
    print(f"\nLevel {level}: {capabilities[level-1]}")
    print(f"   Success Rate:  {results[level]['success_rate']:.1f}%")
    print(f"   Mean Reward:   {results[level]['mean_reward']:.3f}")
    print(f"   Tools Used:    {results[level]['mean_tools']:.2f}")

    if level == 6:
        print(f"   Memory Usage:  {results[level]['memory_usage']:.1f}%")
        print(f"   Memory on Repeats: {results[level]['memory_on_repeats']:.1f}%")

    if level > 1:
        improvement = results[level]['success_rate'] - results[level-1]['success_rate']
        sign = '+' if improvement >= 0 else ''
        print(f"   Improvement:   {sign}{improvement:.1f}% vs Level {level-1}")

print("\n" + "=" * 60)
print("KEY INSIGHTS:")
print("=" * 60)

total_improvement = results[6]['success_rate'] - results[1]['success_rate']
print(f"\n✅ Total improvement from Random to Full: +{total_improvement:.1f}%")

# Find biggest improvements
improvements = {}
for level in range(2, 7):
    imp = results[level]['success_rate'] - results[level-1]['success_rate']
    improvements[level] = imp

best_level = max(improvements, key=improvements.get)
print(f"\n🚀 Biggest single improvement: Level {best_level} ({capabilities[best_level-1]})")
print(f"   +{improvements[best_level]:.1f}% performance gain")

if results[6]['success_rate'] > results[5]['success_rate']:
    memory_gain = results[6]['success_rate'] - results[5]['success_rate']
    print(f"\n🧠 Memory contribution: +{memory_gain:.1f}%")
    print(f"   Agent checked memory on {results[6]['memory_on_repeats']:.0f}% of repeat customer issues")
    print(f"   Memory is now VALUABLE (not broken!)")
else:
    print(f"\n⚠️  Memory still not helping - may need more training")

print("\n" + "=" * 60)

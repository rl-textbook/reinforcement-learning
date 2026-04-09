# Chapter 3 Case Study: MC vs TD(0), Q-Learning, and SARSA on the Windy Desert

This case study compares several of the core tabular reinforcement learning algorithms from Chapter 3 using the `rlbook` package and a single shared environment: the **Windy Desert**. The script is designed to show how these methods behave experimentally rather than to re-derive the theory. The emphasis is on the outputs you can inspect directly:

- the **value functions** learned by Monte Carlo and TD(0)
- the **RMSE convergence** of those prediction methods
- the **policy evolution** of Q-learning and SARSA during training
- the **final paths and evaluation metrics** of the learned control policies

The current `run.py` has **two parts**:

1. **Prediction:** compare **Monte Carlo (MC)** and **TD(0)** on a fixed policy.
2. **Control:** compare **Q-learning** and **SARSA** as they learn policies directly from experience.

A full run saves all figures to disk and also writes a `summary.json` file with the key metrics and metadata.

---

## What this case study focuses on

At a high level, the case study is organized around two classic RL questions.

### Part 1: How do MC and TD(0) estimate state values?
The first half of the script looks at **prediction**, not control. It fixes a policy on the Windy Desert and then compares:

- **Monte Carlo prediction**, which updates from complete episode returns
- **TD(0) prediction**, which updates incrementally using bootstrapped estimates

The main things to watch are:

- how the learned value maps look at the end
- how quickly each method reduces error
- how sensitive the comparison is to the number of episodes and the TD learning rate

### Part 2: How do Q-learning and SARSA behave while learning to control?
The second half moves to **control**. Instead of estimating values for a fixed policy, the agent now learns action values and improves behavior over time.

It compares:

- **Q-learning**, an off-policy control method
- **SARSA**, an on-policy control method

Here the emphasis is on:

- how policies change over training
- what final paths the learned greedy policies produce
- how the two methods compare on **mean return**, **mean steps**, and **success rate**

This is exactly the kind of practical side-by-side comparison that helps bridge the book chapter to concrete experiments.

---

## What gets plotted and measured

### 1. Windy Desert layout
The script first saves a clean visualization of the environment:

- `01_windy_desert_layout.png`

This gives you the state layout before any learning results are overlaid.

### 2. Final MC vs TD(0) value snapshots
After running the prediction phase, the script saves:

- `02_mc_vs_td_values.png`

This figure includes:

- the Windy Desert layout
- the final **MC** value estimates
- the final **TD(0)** value estimates

It is the quickest way to visually compare the end result of the two prediction methods.

### 3. MC vs TD(0) convergence by RMSE
The script computes the error of each prediction snapshot relative to a reference value function and saves:

- `03_mc_vs_td_rmse.png`

This plot shows **RMSE to** `V*` across training snapshots. In practice, this is the most important quantitative comparison for the prediction section because it shows not just the final answer, but how quickly each method gets there.

### 4. Q-learning policy evolution
During control training, the script stores selected checkpoints and visualizes the Q-learning greedy policy at those points in:

- `04_q_learning_policy_evolution.png`

By default, the policy snapshots are taken at episodes:

- `50`
- `200`
- `500`
- `10000`

This gives you an easy way to inspect how the learned behavior changes from early exploration to the final policy.

### 5. SARSA policy evolution
The same type of plot is produced for SARSA:

- `05_sarsa_policy_evolution.png`

Comparing the Q-learning and SARSA evolution plots side by side is often more informative than looking only at final scores, because it makes the learning dynamics visible.

### 6. Final representative paths
After training finishes, the script extracts the final greedy policy for each control algorithm and saves:

- `06_q_vs_sarsa_final_paths.png`

These are **representative paths** through the environment under the final learned policy. They are useful because they connect the abstract policy arrows to actual movement through the grid.

> The paths shown here are representative “mode paths,” meaning they follow the most likely next-state transition under the learned greedy policy. They are not meant to be a full distribution of stochastic rollouts.

### 7. Summary data
In addition to figures, the script writes:

- `summary.json`

This file contains the main run metadata and results, including:

- random seed
- saved figure names
- environment metadata
- prediction hyperparameters
- final MC and TD RMSE values
- control hyperparameters
- reward shaping settings
- representative Q-learning and SARSA paths
- evaluation metrics for the final greedy policies

---

## Environment and reward setup

Although everything is built around the Windy Desert, the script uses it in **two different ways**.

### Prediction setup: MC vs TD(0)
For the prediction comparison, the script creates a Windy Desert environment and then applies the reward structure:

- **step cost:** `-0.01`
- **hole / quicksand penalty:** `-1.0`
- **goal reward:** `+1.0`

It then computes a strong reference solution using value iteration, extracts a greedy policy, and turns that into an **ε-greedy fixed policy** for MC and TD prediction.

So the prediction section is asking:

> Given a fixed policy on this environment, how do MC and TD(0) differ in the way they estimate its value function?

### Control setup: Q-learning vs SARSA
For the control comparison, the script builds a Windy Desert with explicitly stochastic movement:

- **intended move probability:** `0.90`
- **perpendicular drift probability:** `0.05`

It then applies a much harsher control-oriented reward structure:

- **step cost:** `-1.0`
- **hole / quicksand penalty:** `-100.0`
- **goal reward:** `+10.0`

This makes the tradeoffs more visible during learning. In other words, mistakes are expensive, reaching the oasis is rewarding, and extra steps matter.

### Shared default
Both parts of the script use:

- **discount factor (`gamma`)** = `1.0` by default

---

## Requirements

From the repository root, install the package in editable mode:

```bash
pip install -e .
```

That is the setup expected by the case study script.

---

## How to run

From the **repository root**:

```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py
```

If you are already inside that case study directory, you can also run:

```bash
python run.py
```

By default, outputs are written to:

```text
case_studies/ch3_mc_td_qlearning_sarsa/figures
```

---

## Important command-line arguments

This script has more knobs than Chapter 2 because it includes both a prediction experiment and a control experiment. These are the flags that matter most in practice.

### `--prediction-episodes`
```bash
--prediction-episodes 5000
```
Controls how many episodes are used for the **MC vs TD(0)** prediction traces.

- Higher values give a longer and more stable convergence study.
- Lower values are useful for quick tests.
- This affects both MC and TD(0), not just one of them.

If your RMSE curves still look noisy or unfinished, this is one of the first flags to increase.

### `--prediction-snapshot-every`
```bash
--prediction-snapshot-every 100
```
Controls how often the script records MC and TD snapshots during prediction.

- Smaller values give a denser RMSE curve.
- Larger values reduce the number of saved trace points and make runs a bit lighter.

This does not change the learning itself; it changes how frequently you measure and plot progress.

### `--alpha-td`
```bash
--alpha-td 0.1
```
Learning rate for **TD(0)** in the prediction comparison.

This is one of the most important hyperparameters in the entire Chapter 3 script because TD(0) is sensitive to step size.

- Larger values can speed up early learning, but may add instability or noise.
- Smaller values usually make learning smoother, but slower.

MC in this script is not controlled by this flag, so changing `--alpha-td` specifically changes the MC-vs-TD balance.

### `--epsilon-prediction`
```bash
--epsilon-prediction 0.0
```
Exploration level for the **fixed policy** used during MC and TD prediction.

The default of `0.0` means the prediction policy is greedy with respect to the reference policy extracted from value iteration. Increasing this makes the policy more exploratory.

This matters because the value function being estimated is the value of **that specific ε-greedy policy**.

### `--control-episodes`
```bash
--control-episodes 10000
```
Number of training episodes for **Q-learning** and **SARSA**.

This is the main runtime and learning-budget parameter for the control section.

- Increase it when you want better-trained control policies.
- Reduce it for faster experiments or debugging.

If the policy evolution plots still look immature in the final snapshot, increase this first.

### `--control-snapshots`
```bash
--control-snapshots 50,200,500,10000
```
Comma-separated list of training episodes at which to save control policy snapshots.

This flag is especially useful for making the policy-evolution figures more informative.

For example:

- earlier snapshots show how exploration starts to organize behavior
- later snapshots show whether the policy has stabilized

A custom choice like `25,100,1000,20000` can be very helpful when you change the training budget.

### `--alpha-control`
```bash
--alpha-control 0.1
```
Learning rate for **both Q-learning and SARSA**.

Like `--alpha-td`, this is a core optimization hyperparameter.

- Higher values can produce faster early learning, but noisier policies.
- Lower values usually make updates more conservative and can require more episodes.

If you are comparing Q-learning and SARSA fairly, keep them on the same `--alpha-control` unless you specifically want a sensitivity study.

### `--epsilon-control`
```bash
--epsilon-control 0.1
```
Exploration rate for **both Q-learning and SARSA** during training.

This flag has a big effect on the kind of policies learned:

- higher epsilon means more exploration and potentially safer/more robust learning, but slower exploitation
- lower epsilon means faster commitment to current estimates, but a greater chance of under-exploration

This is one of the most important flags for seeing behavioral differences between Q-learning and SARSA in a risky stochastic environment.

### `--gamma`
```bash
--gamma 1.0
```
Discount factor shared across the experiments.

For most Chapter 3 runs, you will leave this at the default. Still, it is useful when you want to test how sensitive the learned values or control policies are to long-horizon versus shorter-horizon returns.

### `--max-steps`
```bash
--max-steps 400
```
Maximum number of steps per episode for the control algorithms.

This acts as a safety cap and also affects evaluation.

- If episodes are getting cut off too early, increase it.
- If you want a faster smoke test, you can lower it.

In practice, this is mainly important when you change exploration, rewards, or the number of episodes.

### `--eval-episodes`
```bash
--eval-episodes 100
```
Number of episodes used to evaluate the final greedy Q-learning and SARSA policies.

This affects the stability of the reported final metrics:

- **mean return**
- **mean steps**
- **success rate**

Higher values give more reliable comparisons but take longer.

### `--seed`
```bash
--seed 42
```
Random seed for NumPy and Python’s `random` module.

This matters for:

- reproducible training
- representative path generation
- stable comparisons across runs

### `--save-dir`
```bash
--save-dir path/to/output_dir
```
Custom directory for all saved PNGs and `summary.json`.

This is very useful when you want to compare multiple runs with different hyperparameters side by side.

### `--show`
```bash
--show
```
Displays figures interactively in addition to saving them.

This is mainly for local exploration. For automated runs, scriptable experiments, or figure generation, you will usually leave it off.

---

## Hyperparameter guidance

If you only skim one section before experimenting, make it this one.

### Best defaults for a normal run
The defaults are a strong baseline for the Chapter 3 figures:

```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py
```

They are already set to produce a full prediction comparison and a full control comparison without needing any extra configuration.

### When to increase `--prediction-episodes`
Use more prediction episodes when you care about the MC-vs-TD comparison itself:

```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --prediction-episodes 20000
```

This is especially helpful if:

- the RMSE curves still look noisy
- you want a cleaner convergence story
- you are comparing several values of `--alpha-td`

### When to change `--prediction-snapshot-every`
Use more frequent snapshots when you want a more detailed RMSE curve:

```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --prediction-snapshot-every 25
```

This is great for close inspection, but it creates more plotted points.

### When to tune `--alpha-td`
If TD(0) looks too noisy, reduce the step size:

```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --alpha-td 0.05
```

If TD(0) is learning too slowly, try increasing it:

```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --alpha-td 0.2
```

The useful habit here is to change `--alpha-td` and `--prediction-episodes` together rather than in isolation.

### When to change `--epsilon-prediction`
The default prediction study uses a greedy fixed policy:

```bash
--epsilon-prediction 0.0
```

If you want the prediction task to reflect a more exploratory policy, raise epsilon:

```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --epsilon-prediction 0.1
```

That changes the policy being evaluated, which in turn changes the value function both MC and TD are trying to estimate.

### When to increase `--control-episodes`
If your main interest is Q-learning vs SARSA as control methods, more training episodes is usually the most valuable change:

```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --control-episodes 30000
```

This is the first flag to increase when you want more mature final policies.

### When to change `--control-snapshots`
When you increase training duration, it is often worth updating the snapshot locations too:

```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --control-episodes 30000 \
  --control-snapshots 50,500,5000,30000
```

That keeps the evolution plots informative across the full training range.

### When to tune `--alpha-control`
A smaller control learning rate can make policy evolution smoother:

```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --alpha-control 0.05
```

A larger one can speed up adaptation, but may produce more volatile policies:

```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --alpha-control 0.2
```

### When to tune `--epsilon-control`
This is one of the most behaviorally important settings in the control section.

For less exploration:

```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --epsilon-control 0.05
```

For more exploration:

```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --epsilon-control 0.2
```

Higher exploration can be very useful early in training, but it also changes the effective learning dynamics of SARSA in particular.

### When to increase `--eval-episodes`
If you mainly care about the final quantitative comparison between Q-learning and SARSA, raise the evaluation budget:

```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --eval-episodes 1000
```

This gives more stable estimates of final performance.

---

## Recommended commands

These are the most useful commands for day-to-day use.

### 1. Standard full run
```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py
```
Use this for the default Chapter 3 figures and metrics.

### 2. Longer MC vs TD(0) convergence study
```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --prediction-episodes 20000 \
  --prediction-snapshot-every 200 \
  --alpha-td 0.05
```
Use this when the prediction comparison is the main focus and you want a longer, smoother convergence picture.

### 3. Exploratory fixed-policy prediction run
```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --epsilon-prediction 0.1 \
  --prediction-episodes 10000
```
Use this when you want MC and TD(0) to estimate the value of a more exploratory ε-greedy policy instead of a purely greedy one.

### 4. Longer Q-learning vs SARSA training run
```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --control-episodes 30000 \
  --control-snapshots 50,500,5000,30000 \
  --alpha-control 0.05
```
Use this when your priority is the control comparison rather than the prediction plots.

### 5. Higher-exploration control run
```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --epsilon-control 0.2 \
  --control-episodes 20000
```
Use this to see how stronger exploration changes the learned policies and the gap between Q-learning and SARSA.

### 6. More stable final evaluation
```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --eval-episodes 1000
```
Use this when you want the final reported return, steps, and success-rate comparison to be less noisy.

### 7. Save outputs to a custom directory
```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --save-dir outputs/ch3_seed7_longrun \
  --seed 7
```
Use this when you are comparing multiple settings and want to keep every run separate.

### 8. Interactive plotting
```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py --show
```
Use this for local inspection when you want plots to open on screen as they are generated.

---

## Reading the outputs

A good way to interpret a finished run is:

1. **Start with the prediction section**
   - Compare the final MC and TD value maps.
   - Then check the RMSE plot to see which method reduces error faster and how noisy the learning curves are.

2. **Look at policy evolution, not just the final answer**
   - The Q-learning and SARSA snapshot plots show how behavior develops across training.
   - This is often where the most interesting differences show up.

3. **Use final evaluation metrics to ground the visual impressions**
   - The summary data reports mean return, mean steps, and success rate.
   - These are the most direct numerical comparison for the control section.

4. **Inspect the representative paths last**
   - The final path plot helps translate the learned policy into actual movement through the desert.
   - It is especially useful for seeing whether one method learned a more direct or more cautious route.

---

## Expected qualitative takeaways

For a typical Chapter 3 run, the main takeaways are usually along the following lines:

- **TD(0) often improves faster early** in the prediction setting because it bootstraps.
- **MC can look noisier** because it waits for full returns.
- Given enough experience, both methods are trying to estimate the value of the same fixed policy in the prediction section.
- In the stochastic control setting, **Q-learning and SARSA can produce noticeably different behaviors**, even when trained with the same reward structure and similar hyperparameters.
- A common pattern is that **Q-learning learns a more aggressive route** while **SARSA learns a more conservative one**, though the exact outcome depends on the seed, training budget, exploration rate, and reward setup.

These are exactly the practical differences the Chapter 3 case study is meant to highlight.

---

## Files produced by a successful run

A standard run writes the following artifacts:

```text
01_windy_desert_layout.png
02_mc_vs_td_values.png
03_mc_vs_td_rmse.png
04_q_learning_policy_evolution.png
05_sarsa_policy_evolution.png
06_q_vs_sarsa_final_paths.png
summary.json
```

---

## Troubleshooting

### `ModuleNotFoundError: rlbook`
Install the package from the repository root:

```bash
pip install -e .
```

Then rerun the case study.

### The control snapshot flag errors out
`--control-snapshots` must be a comma-separated list of positive integers, for example:

```bash
--control-snapshots 50,200,500,10000
```

### Plots save but do not appear on screen
That is expected unless you pass:

```bash
--show
```

### I only want a fast smoke test
Use fewer episodes for both parts and reduce evaluation:

```bash
python case_studies/ch3_mc_td_qlearning_sarsa/run.py \
  --prediction-episodes 500 \
  --control-episodes 1000 \
  --eval-episodes 20
```

---

## Summary

This Chapter 3 case study is a compact experimental comparison of four foundational RL methods on the Windy Desert:

- **MC**
- **TD(0)**
- **Q-learning**
- **SARSA**

It keeps the theory at a high level and focuses on the parts that are most useful when actually running experiments:

- what gets plotted
- what gets measured
- how the important hyperparameters affect behavior
- which command-line flags matter most
- how to interpret the resulting figures and metrics

For book readers, it provides a practical bridge from the chapter concepts to an executable, reproducible set of experiments.

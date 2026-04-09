# Chapter 4 Case Study: REINFORCE vs Actor-Critic vs DQN on CartPole

This case study compares three major deep reinforcement learning families on **CartPole-v1** using the `rlbook` package:

- **REINFORCE** for Monte Carlo policy gradients
- **Actor-Critic** for policy-gradient learning with a learned critic
- **DQN** for value-based control
- optionally, **Double DQN** as an ablation

The goal of the script is not to turn the README into a theory chapter. The book handles the deeper derivations. Here, the emphasis is on the practical side of the experiment:

- what is being trained
- what is being plotted
- what is being measured
- which hyperparameters matter most
- which `python run.py --...` commands are most useful
- how to run the study sensibly on weaker vs stronger hardware

The current `run.py` is set up as a **multi-seed CLI experiment**. It trains the algorithms, evaluates them periodically, writes logs and tables, and saves publication-ready figures. It is designed much more like a small benchmark script than a one-off notebook.

---

## What this case study focuses on

At a high level, this chapter is about comparing three different deep RL styles on the same control problem.

### REINFORCE
REINFORCE learns a policy directly from returns. In practice, it is often the noisiest of the three methods here, but it is conceptually clean and makes a great baseline for policy-gradient learning.

### Actor-Critic
Actor-Critic keeps the policy-gradient flavor but adds a learned critic to reduce variance and improve learning efficiency. In many runs, it is the method that feels like the best compromise between direct policy learning and stable training.

### DQN
DQN learns action values instead of a stochastic policy directly. In this script, it is the value-based baseline and includes the standard practical machinery you would expect in a DQN study, like replay and target synchronization.

### Optional Double DQN
The script can also include **Double DQN** as an extra comparison. This is not part of the default run, but it is an important optional ablation when you want to compare a stronger value-based variant against the base DQN setup.

So the chapter is really comparing three different experimental questions at once:

- how well a **pure policy-gradient** method behaves
- how much a **critic** helps
- how a **value-based deep RL** method compares to both

---

## What gets plotted and measured

The current CLI is more statistically careful than a simple single-run plot. Most of the main figures are built from **evaluation-only logs across multiple seeds** rather than from a single raw training trace.

### 1. Combined learning curves
The script saves a combined learning-curve plot for all included algorithms:

- `learning_curves_combined.png`
- and by default also `learning_curves_combined.pdf`

This figure shows:

- **mean evaluation return across seeds**
- an **IQR band** across seeds
- a horizontal **solved threshold** reference line

This is the main overview figure for the chapter.

### 2. Per-algorithm learning curves
The script also saves an individual learning-curve figure for each algorithm, such as:

- `learning_curve_REINFORCE.png`
- `learning_curve_ActorCritic.png`
- `learning_curve_DQN.png`
- `learning_curve_DoubleDQN.png` if enabled

These plots are especially helpful because they show:

- thin **per-seed traces**
- a thicker **mean curve**
- an optional **IQR band**

That makes it much easier to judge whether an algorithm is consistently good or just occasionally good.

### 3. Terminal vs best-seen performance
The script saves:

- `terminal_vs_best_performance.png`

This compares:

- the **final evaluation return** at the end of training
- the **best evaluation return seen** during training

This matters because some methods spike high and then become less stable later, while others finish closer to their peak.

### 4. Sample efficiency via capped AUC
The script computes a sample-efficiency comparison using **area under the evaluation curve up to a shared step cap** and saves:

- `sample_efficiency.png`

This is one of the most useful quantitative comparisons in the Chapter 4 setup because it rewards algorithms that improve **earlier**, not just algorithms that eventually get good.

### 5. Stability over time
The script also measures instability explicitly and saves:

- `stability_over_time.png`

This plot shows the **across-seed IQR of evaluation return** as a function of training progress. Lower values mean the method is behaving more consistently across random seeds.

### 6. CSV and log artifacts
Beyond the figures, the script writes structured analysis files that make reruns and comparisons much easier:

- `config.json`
- `eval_logs.jsonl`
- `eval_logs.csv`
- `eval_curves_aggregated.csv`
- `sample_efficiency_per_seed.csv`
- `sample_efficiency_summary.csv`
- `stability_over_time.csv`
- `summary.csv`

This is important because Chapter 4 is where experiments start becoming expensive enough that you do not want to retrain every time you want a slightly different plot.

---

## What is actually being measured

This case study is not just asking, “Which method eventually gets the highest score?” It measures several different things.

### Evaluation return over training
The core metric is **evaluation return** on CartPole-v1 at regular step intervals.

The script logs evaluation performance every fixed number of training steps rather than plotting raw training reward per episode. That makes the comparisons cleaner and more apples-to-apples across algorithms.

### Final return
The summary tables report how good each algorithm is at the end of the run.

### Best-seen return
The script also tracks the best evaluation score each seed reached during training.

### Best-to-final drop
This is a very useful practical metric. It tells you whether an algorithm tends to stay near its best level or whether it often peaks and then drifts downward.

### Sample efficiency
The capped AUC comparison measures how much return the algorithm accumulated **early in training** up to a common step budget.

### Stability across seeds
The stability plot and summary tables quantify how sensitive each algorithm is to randomness.

### Steps to solve
The script also measures when a seed first crosses the chosen solved threshold.

By default, the solved threshold is:

- `450.0`

That threshold is used for:

- summary statistics
- “solved seed fraction”
- steps-to-solve reporting
- reference lines in the plots

It is an **analysis threshold**, not a hard training stop.

---

## Experimental setup at a glance

The current defaults are built around a small but meaningful multi-seed benchmark.

### Default seeds
The script uses:

- `101`
- `202`
- `303`

by default.

### Default algorithms
A standard run includes:

- REINFORCE
- Actor-Critic
- DQN

and only includes Double DQN if you add:

```bash
--include-double-dqn
```

### Default training budget
The key default run settings are:

- `--episodes 10000`
- `--steps-per-episode 500`
- `--eval-every-steps 1000`
- `--eval-episodes 10`
- `--total-step-budget 50000`
- `--gamma 0.99`

### Default optimization settings
For the current script, the main learning-rate defaults are:

- `--reinforce-lr 1e-3`
- `--actor-lr 3e-4`
- `--critic-lr 1e-3`
- `--dqn-lr 2e-4`

And the main DQN-specific defaults are:

- `--dqn-batch-size 64`
- `--dqn-replay-capacity 100000`
- `--dqn-target-sync-every 1000`
- `--eps-start 1.0`
- `--eps-end 0.05`
- `--eps-decay-steps 100000`

---

## Requirements

From the repository root, install the package in editable mode:

```bash
pip install -e .
```

That is the setup expected by the Chapter 4 case-study script.

---

## How to run

If you are inside the Chapter 4 case-study directory, the standard command is:

```bash
python run.py
```

By default, outputs are written to:

```text
results/chapter4_cartpole_cli
```

You can change both the base output folder and the run subdirectory with:

```bash
--outdir ...
--run-name ...
```

---

## Important command-line arguments

This script has more moving parts than the earlier tabular chapters, so it helps to think of the CLI in groups.

### A. The most important experiment-control flags

### `--seeds`
```bash
--seeds 101 202 303
```
Controls which random seeds are run.

This is one of the most important flags in the entire script because the whole benchmark is built around **across-seed comparisons**.

- More seeds give more trustworthy comparisons.
- More seeds also increase runtime almost linearly.
- If two algorithms look close, increasing seeds is often more valuable than micro-tuning learning rates.

### `--include-double-dqn`
```bash
--include-double-dqn
```
Adds Double DQN to the benchmark.

This is useful when you want the Chapter 4 study to include an extra value-based ablation. It also increases total training time, because every seed now runs four algorithms instead of three.

### `--smoke-test` / `--quick`
```bash
--smoke-test
```
Runs a much shorter version of the benchmark.

This is the best first command to use when:

- you are checking that the environment works
- you just changed code and want a sanity check
- you are on a weaker laptop
- you do not want to wait for the full multi-seed run

Under the hood, the smoke test reduces the budget to a shorter configuration, including:

- `episodes = 2000`
- `steps_per_episode = 500`
- `eval_every_steps = 1000`
- `eval_episodes = 5`
- `total_step_budget = 10000`

### `--no-run` and `--from-jsonl`
```bash
--no-run --from-jsonl results/chapter4_cartpole_cli/eval_logs.jsonl
```
This is one of the most practical features in the whole script.

It lets you **skip training** and regenerate tables and plots from an existing evaluation log.

This is extremely useful once runs get expensive. You can train once, then revisit the analysis without paying the training cost again.

---

### B. The main training-budget flags

### `--total-step-budget`
```bash
--total-step-budget 50000
```
This is the single most important “how long should I train?” flag.

If you want a fairer comparison across algorithms, this is usually the first budget knob to think about, because the analysis is built around **steps**, not just episodes.

- Increase it for stronger final performance and a longer learning curve.
- Decrease it for faster iteration or smoke testing.

### `--episodes`
```bash
--episodes 10000
```
Episode cap for the run configuration.

This is useful, but in practice the step budget is often the more meaningful way to think about a fixed training budget for Chapter 4.

### `--steps-per-episode`
```bash
--steps-per-episode 500
```
Maximum steps per episode.

CartPole episodes are bounded, so this acts as a safety cap and an experiment-definition parameter.

### `--eval-every-steps`
```bash
--eval-every-steps 1000
```
Controls how often evaluation is performed.

This flag matters a lot for the look of the learning curves:

- smaller values give denser, more detailed curves
- larger values reduce evaluation overhead and log size

It does not directly make the algorithm smarter. It changes how often you measure progress.

### `--eval-episodes`
```bash
--eval-episodes 10
```
Number of episodes used at each evaluation point.

This controls how noisy the evaluation curves are.

- Higher values give cleaner comparisons.
- Lower values make runs cheaper.

If you only change one evaluation-quality flag, this is often the one to change.

---

### C. The most important optimization hyperparameters

### `--reinforce-lr`
```bash
--reinforce-lr 1e-3
```
Learning rate for REINFORCE.

This is the most important REINFORCE tuning flag in the current CLI.

- Too high can make training noisy.
- Too low can make improvement frustratingly slow.

### `--actor-lr`
```bash
--actor-lr 3e-4
```
Learning rate for the actor in Actor-Critic.

### `--critic-lr`
```bash
--critic-lr 1e-3
```
Learning rate for the critic in Actor-Critic.

These two should usually be thought about together:

- the **actor** controls policy updates
- the **critic** controls value-learning speed and baseline quality

If Actor-Critic looks unstable, the first thing to try is often a smaller actor learning rate or a more balanced actor/critic ratio.

### `--dqn-lr`
```bash
--dqn-lr 2e-4
```
Learning rate for DQN (and for Double DQN when enabled).

This is the most important DQN optimization knob in the CLI. If DQN is learning too slowly or too erratically, this is one of the first places to look.

---

### D. The key DQN-specific control flags

### `--dqn-batch-size`
```bash
--dqn-batch-size 64
```
Mini-batch size for replay updates.

- Larger batch sizes can make updates smoother but increase compute per optimization step.
- Smaller batch sizes are lighter but can increase noise.

### `--dqn-replay-capacity`
```bash
--dqn-replay-capacity 100000
```
Replay-buffer size.

This determines how much past experience DQN can store.

- Larger buffers provide more diversity and longer memory.
- Smaller buffers reduce memory use and can make experiments cheaper.

### `--dqn-target-sync-every`
```bash
--dqn-target-sync-every 1000
```
How often the target network is synchronized.

This is a core DQN stability parameter:

- syncing too often reduces the benefit of a stable target
- syncing too rarely can make value targets stale

### `--eps-start`, `--eps-end`, `--eps-decay-steps`
```bash
--eps-start 1.0 --eps-end 0.05 --eps-decay-steps 100000
```
These control DQN exploration.

Together, they define how epsilon-greedy exploration decays over time:

- `--eps-start` = where exploration begins
- `--eps-end` = the minimum exploration floor
- `--eps-decay-steps` = how quickly the schedule decays

This group is often more important than people expect. A perfectly fine DQN learning rate can still look bad if the exploration schedule is mismatched to the training budget.

---

### E. Analysis and plotting flags that matter more than they look

### `--smooth-window`
```bash
--smooth-window 1
```
Centered rolling window used when plotting aggregated curves.

The default `1` means **no smoothing**.

This does not change the underlying training data. It only changes how smooth the plotted mean and IQR curves look.

### `--solved-threshold`
```bash
--solved-threshold 450
```
Solved threshold used for:

- reference lines
- solved-seed fraction
- steps-to-solve reporting

This is an analysis setting, not a training parameter.

### `--auc-cap-steps`
```bash
--auc-cap-steps 20000
```
Step cap used for the sample-efficiency comparison.

This controls the horizon at which you compare “who learns fastest.”

- smaller caps emphasize very early learning
- larger caps make the AUC comparison closer to an overall training-quality score

### `--stability-bin-size`
```bash
--stability-bin-size 1000
```
Bin size used for the stability-over-time plot.

- smaller bins give a more detailed but noisier stability curve
- larger bins smooth the story and reduce clutter

### `--stability-max-steps`
```bash
--stability-max-steps 30000
```
Optional cap for how far the stability plot extends. This is useful when you want to focus on early or mid-training behavior without changing the underlying run.

### `--save-formats`
```bash
--save-formats png,pdf
```
Figure formats to save.

This is a small flag, but it matters in practice:

- `png,pdf` is great for polished outputs
- `png` only is lighter and faster, especially on weaker machines or iterative debugging runs

### `--outdir` and `--run-name`
```bash
--outdir results --run-name chapter4_cartpole_cli
```
These control where outputs go.

They are especially useful when you want to keep multiple runs separate and compare them later.

---

## Hyperparameter guidance

If you only skim one tuning section before experimenting, make it this one.

### Best default for a normal run
The default configuration is a solid starting point for the Chapter 4 figures:

```bash
python run.py
```

This gives you:

- three seeds
- REINFORCE, Actor-Critic, and DQN
- evaluation-only learning curves
- sample-efficiency and stability analysis
- CSV and JSON artifacts for later reuse

### When to use `--smoke-test`
Use the smoke test first whenever you are not sure your environment is fully set up:

```bash
python run.py --smoke-test --save-formats png
```

This is the fastest way to verify that:

- dependencies import correctly
- training actually runs
- outputs are being written where you expect

### When to increase `--total-step-budget`
If your main goal is better-trained models and more mature learning curves, increase the step budget first:

```bash
python run.py --total-step-budget 100000
```

This is usually more valuable than immediately changing lots of learning rates.

### When to increase `--eval-episodes`
If your curves or summaries feel noisy, a higher evaluation count often helps more than people expect:

```bash
python run.py --eval-episodes 25
```

This makes the evaluation curves and final summary statistics more stable.

### When to increase the number of seeds
If your main goal is a stronger benchmark rather than a quick demo, add more seeds:

```bash
python run.py --seeds 101 202 303 404 505
```

This is especially helpful when two methods look close and you want a more trustworthy comparison.

### When to tune REINFORCE
If REINFORCE looks very noisy or stalls:

```bash
python run.py --reinforce-lr 5e-4
```

If it looks too conservative and underpowered:

```bash
python run.py --reinforce-lr 2e-3
```

### When to tune Actor-Critic
If Actor-Critic is unstable, try making the actor updates more conservative first:

```bash
python run.py --actor-lr 1e-4 --critic-lr 1e-3
```

If the critic appears too slow relative to the actor, raising the critic learning rate can also be worth trying.

### When to tune DQN
If DQN is too slow, one of the first things to test is:

```bash
python run.py --dqn-lr 5e-4
```

If DQN becomes too noisy, reduce it:

```bash
python run.py --dqn-lr 1e-4
```

### When to tune DQN exploration
If DQN is not exploring enough early:

```bash
python run.py --eps-start 1.0 --eps-end 0.05 --eps-decay-steps 150000
```

If you want faster commitment to greedy behavior:

```bash
python run.py --eps-start 1.0 --eps-end 0.05 --eps-decay-steps 50000
```

The key idea is that `--eps-decay-steps` should make sense relative to your total training budget.

### When to include Double DQN
If you want a stronger value-based comparison, enable the ablation explicitly:

```bash
python run.py --include-double-dqn
```

This is one of the cleanest ways to turn the Chapter 4 case study into a slightly more serious benchmark.

### When to re-analyze without retraining
After a long run, use:

```bash
python run.py --no-run --from-jsonl results/chapter4_cartpole_cli/eval_logs.jsonl
```

This is the recommended workflow once you start doing longer experiments.

Train once, analyze many times.

---

## Compute and hardware guidance

From this chapter onward, training is starting to become real compute rather than “instant toy code,” even though CartPole is still a light environment compared with later deep RL case studies.

A good rule of thumb is:

- **for setup checks and fast iteration, optimize for convenience**
- **for benchmark-quality comparisons, optimize for repeatability and training budget**

### 1. Laptop or weaker CPU
If you are on a modest laptop, the best workflow is usually:

```bash
python run.py --smoke-test --save-formats png
```

Then, if that works, move to a lighter full run such as:

```bash
python run.py \
  --seeds 101 202 \
  --total-step-budget 30000 \
  --eval-episodes 5 \
  --save-formats png
```

Why this works well:

- fewer seeds reduce runtime directly
- a smaller step budget gives you a shorter benchmark
- fewer eval episodes reduce overhead
- saving only PNGs reduces file-writing cost

### 2. Normal desktop or stronger CPU laptop
A typical local full run is usually reasonable with the defaults:

```bash
python run.py
```

Or, for a slightly stronger local benchmark:

```bash
python run.py \
  --seeds 101 202 303 404 505 \
  --eval-episodes 20 \
  --save-formats png
```

That is often a better use of compute than immediately turning on every extra option.

### 3. Colab or cloud GPU workflow
A **Colab GPU runtime such as a T4** is a good fit once you start doing longer multi-seed runs, enabling Double DQN, or preparing cleaner chapter figures. CartPole itself is not so large that a GPU is mandatory, but a cloud runtime becomes useful when the benchmark grows.

A practical “stronger benchmark” style command would be:

```bash
python run.py \
  --seeds 101 202 303 404 505 \
  --include-double-dqn \
  --total-step-budget 100000 \
  --eval-episodes 20 \
  --auc-cap-steps 50000 \
  --save-formats png
```

And if you are saving to mounted Google Drive in Colab, something like this is convenient:

```bash
python run.py \
  --outdir /content/drive/MyDrive/rl_results \
  --run-name chapter4_t4_longrun \
  --seeds 101 202 303 404 505 \
  --include-double-dqn \
  --total-step-budget 100000 \
  --eval-episodes 20 \
  --save-formats png
```

### 4. Important practical note about GPU expectations
The Chapter 4 script itself does **not** expose a `--device` flag. It also explicitly sets PyTorch CPU thread counts to modest values at startup.

That means two things:

1. the script is designed to be stable and CLI-friendly on normal machines
2. the actual amount of GPU benefit depends on how the underlying `rlbook` deep-RL implementations handle device placement in your environment

So the best way to think about a T4 is not “this chapter requires a GPU,” but rather:

> a T4 becomes a very sensible convenience once you want longer runs, more seeds, or the optional Double DQN benchmark.

### 5. Best practice by stage

#### For code/debug stage
Use:

```bash
python run.py --smoke-test --save-formats png
```

#### For chapter-quality local figures
Use:

```bash
python run.py --seeds 101 202 303 --save-formats png,pdf
```

#### For heavier benchmark runs
Use more seeds, longer budgets, and usually PNG-only output:

```bash
python run.py \
  --seeds 101 202 303 404 505 \
  --include-double-dqn \
  --total-step-budget 100000 \
  --save-formats png
```

---

## Recommended commands

These are the most useful day-to-day commands for this case study.

### 1. Standard Chapter 4 run
```bash
python run.py
```
Use this for the default multi-seed comparison of REINFORCE, Actor-Critic, and DQN.

### 2. Fast smoke test
```bash
python run.py --smoke-test --save-formats png
```
Use this before committing to a longer run.

### 3. Add Double DQN
```bash
python run.py --include-double-dqn
```
Use this when you want the optional value-based ablation.

### 4. More statistically reliable benchmark
```bash
python run.py --seeds 101 202 303 404 505 --eval-episodes 20
```
Use this when you care more about comparison quality than turnaround time.

### 5. Longer training budget
```bash
python run.py --total-step-budget 100000 --auc-cap-steps 50000
```
Use this when the default run feels too short for the story you want to tell.

### 6. Lighter local run
```bash
python run.py \
  --seeds 101 202 \
  --total-step-budget 30000 \
  --eval-episodes 5 \
  --save-formats png
```
Use this on weaker machines.

### 7. Regenerate plots and tables without retraining
```bash
python run.py --no-run --from-jsonl results/chapter4_cartpole_cli/eval_logs.jsonl
```
Use this after a long run when you only want to rebuild the analysis artifacts.

### 8. Custom output directory for organized experiments
```bash
python run.py \
  --outdir results \
  --run-name ch4_actorcritic_vs_dqn_longrun
```
Use this when you are comparing multiple settings side by side.

---

## Reading the outputs

A simple way to interpret a finished run is:

1. **Start with the combined learning curves**
   - This gives the fastest overall picture of who learns quickly, who finishes strong, and how much seed-to-seed spread there is.

2. **Then inspect the per-algorithm plots**
   - These reveal whether the mean curve is hiding unstable or inconsistent seeds.

3. **Check sample efficiency next**
   - The capped AUC figure tells you which method gets useful performance earlier.

4. **Check terminal vs best-seen performance**
   - This helps you tell apart methods that peak briefly from methods that finish robustly.

5. **Use the stability plot to judge reliability**
   - Low across-seed IQR means a method behaves more consistently.

6. **Use `summary.csv` for the clean numerical comparison**
   - This is the best file for reporting final means, best means, solved fraction, and steps-to-solve statistics.

---

## Files produced by a successful run

A standard successful run will produce a directory containing artifacts like:

```text
config.json
eval_logs.jsonl
eval_logs.csv
eval_curves_aggregated.csv
sample_efficiency_per_seed.csv
sample_efficiency_summary.csv
stability_over_time.csv
summary.csv
learning_curves_combined.png
learning_curves_combined.pdf
learning_curve_REINFORCE.png
learning_curve_ActorCritic.png
learning_curve_DQN.png
terminal_vs_best_performance.png
sample_efficiency.png
stability_over_time.png
```

If `--include-double-dqn` is enabled, you will also get the Double DQN learning-curve artifact.

---

## Troubleshooting

### `ModuleNotFoundError: rlbook`
Install the package from the repository root:

```bash
pip install -e .
```

Then rerun the script.

### I want to change plots without retraining
Use:

```bash
python run.py --no-run --from-jsonl path/to/eval_logs.jsonl
```

### My run feels too slow locally
The first changes to make are usually:

- use `--smoke-test`
- reduce the number of seeds
- reduce `--total-step-budget`
- reduce `--eval-episodes`
- switch to `--save-formats png`

### My figures look noisy
The most useful fixes are usually:

- increase `--eval-episodes`
- add more seeds
- optionally increase `--smooth-window` for presentation only

### I want a more benchmark-style run
The most useful changes are usually:

- add more seeds
- increase `--total-step-budget`
- consider `--include-double-dqn`
- keep outputs in separate run directories with `--run-name`

---

## Summary

This Chapter 4 case study is the bridge from the tabular part of the book into the deep RL workflow.

It keeps the theory at a high level and focuses on the practical pieces that matter most when you actually run the experiments:

- what the algorithms are comparing
- what the figures mean
- what the summary tables measure
- which CLI commands are worth using regularly
- which hyperparameters matter most
- how to scale the experiment sensibly depending on your hardware

For readers of the book, it turns the Chapter 4 ideas into a reproducible benchmark-style experiment on CartPole that is easy to rerun, extend, and analyze.

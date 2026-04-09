# Chapter 2 Case Study: Value Iteration vs Policy Iteration

This case study compares **Value Iteration (VI)** and **Policy Iteration (PI)** on small grid-world environments using the implementations provided by the `rlbook` package. The goal is not to go deep into the theory or derivations, but to show how the two dynamic programming methods behave in practice: how quickly they converge, what policies they produce, what value functions they learn, and how those learned policies perform when executed.

The current script is driven by `run.py` and runs **two deterministic test cases**:

- a **4×4 grid world**
- a **5×5 grid world with holes** at `(1, 1)`, `(2, 3)`, and `(3, 1)`

For each case, the script solves the environment with VI and PI, compares the resulting policies and values, saves a set of figures, and writes a `summary.json` file. It also exits with a non-zero status if VI and PI disagree on the tested environments, so it works as both a case study and a lightweight correctness check.

---

## What this case study focuses on

At a high level, Chapter 2 is about comparing two classic planning algorithms for finite Markov Decision Processes:

- **Value Iteration** updates state values directly until the value function stabilizes, then extracts a greedy policy.
- **Policy Iteration** alternates between policy evaluation and policy improvement until the policy stops changing.

In this case study, the emphasis is on the practical outputs:

- **optimal policies** learned by VI and PI
- **optimal value functions** for each method
- **convergence behavior** over iterations
- **policy evolution** over early iterations
- **downstream agent performance** when the learned policy is actually run

Rather than going deep into the math, the script visualizes the quantities that matter most when comparing these methods experimentally.

---

## What gets plotted and measured

### 1. Policy and value visualizations
The script first solves a **4×4 deterministic grid** with both VI and PI, then saves one plot for each learned solution:

- `01_value_iteration_4x4.png`
- `02_policy_iteration_4x4.png`

These figures let you compare the learned values and greedy actions side by side in the simplest environment.

### 2. Convergence plots on the 4×4 grid
The script then records a trace of both algorithms during optimization:

- `03_vi_convergence_4x4.png` plots **max Bellman error (Δ)** by VI iteration on a semilog scale.
- `04_pi_convergence_4x4.png` plots **number of policy changes** per PI iteration.

These two plots tell slightly different stories:

- VI is tracked through the size of its value-function updates.
- PI is tracked through how much the policy still changes from one outer iteration to the next.

### 3. VI vs PI comparison on a harder grid
The second environment is a **5×5 deterministic grid with holes**. Here the script saves:

- `05_vi_vs_pi_policy_5x5.png` for the final VI and PI policies
- `06_vi_vs_pi_values_5x5.png` for the final VI and PI value heatmaps

This is where the comparison becomes more interesting: the agent has to navigate around blocked or terminal “bad” states while still reaching the goal efficiently.

### 4. Policy evolution over time
To make the algorithmic differences more visible, the script captures intermediate behavior at steps:

- `k = 1, 2, 3, 5, 10`

and saves:

- `07_policy_evolution_5x5.png`

This figure is useful for showing how VI and PI approach their final policies differently in the early stages.

### 5. Agent-level performance metrics
After solving the 5×5 grid, the script evaluates three agents:

- **VI policy**
- **PI policy**
- **Random baseline**

It computes and plots:

- **mean reward**
- **mean steps**
- **success rate**

in:

- `08_agent_performance_5x5.png`

This moves the comparison beyond pure planning convergence and into behavioral performance.

### 6. Example trajectories
Finally, the script visualizes actual paths taken in the 5×5 environment:

- `09_vi_agent_path_5x5.png` shows a sample trajectory from the VI policy
- `10_random_best_path_5x5.png` shows the best path found by the random baseline during evaluation

These plots are helpful when you want a qualitative picture of how a learned policy behaves versus unguided exploration.

### 7. Summary data
In addition to figures, the script writes:

- `summary.json`

This file stores the key run metadata and metrics, including:

- seed
- convergence threshold
- whether VI and PI matched on non-terminal states
- maximum non-terminal value difference
- iteration counts
- agent evaluation results
- a sample VI episode path
- the best random baseline path

---

## Environments used

The current `run.py` uses the following environments:

### Case 1: 4×4 deterministic grid
A simple deterministic grid world used to compare:

- final value functions
- final policies
- convergence traces

### Case 2: 5×5 deterministic grid with holes
A larger deterministic grid with holes at:

- `(1, 1)`
- `(2, 3)`
- `(3, 1)`

This case is used for:

- final policy comparison
- value heatmaps
- policy evolution across iterations
- agent evaluation against a random baseline
- path visualization

---

## Requirements

From the repository root, install the package in editable mode:

```bash
pip install -e .
```

That is the setup expected by the script.

---

## How to run

From the **repository root**:

```bash
python case_studies/ch2_value_iteration_vs_policy_iteration/run.py
```

If you are already inside the case study directory, you can also run:

```bash
python run.py
```

By default, figures are saved to:

```text
case_studies/ch2_value_iteration_vs_policy_iteration/figures
```

---

## Important command-line arguments

The CLI is intentionally small, and most of the important control comes from a few flags.

### `--theta`
```bash
--theta 1e-3
```
Convergence threshold for both algorithms.

This is the most important optimization hyperparameter in the script.

- **Smaller `theta`** means stricter convergence.
- **Larger `theta`** means earlier stopping.
- Lowering it usually gives more precise value estimates, but increases runtime.

In practice:

- use the default when you want a clean, standard run
- lower it when you want tighter convergence or are checking sensitivity
- raise it for faster smoke tests

### `--vi-max-iterations`
```bash
--vi-max-iterations 100
```
Maximum number of VI sweeps.

This acts as a hard cap in case VI has not yet reached the threshold. It is mostly a safety parameter, but it matters if you make `theta` much smaller or later extend the case study to bigger environments.

### `--pi-max-iterations`
```bash
--pi-max-iterations 1000
```
Maximum number of PI iterations.

This is the outer-loop cap for policy iteration. In many small problems PI converges well before this limit, so the default is generous.

### `--episodes`
```bash
--episodes 100
```
Number of episodes used when evaluating VI, PI, and random agents on the 5×5 grid.

This parameter does **not** affect the planning solution itself. Instead, it controls how stable the reported evaluation metrics are:

- higher values give smoother estimates for mean reward, mean steps, and success rate
- lower values run faster but add more variance to the comparison

### `--seed`
```bash
--seed 42
```
Random seed for NumPy and Python’s `random` module.

Use this when you want reproducible evaluation, especially for:

- random baseline behavior
- sampled trajectories
- stable comparisons across runs

### `--save-dir`
```bash
--save-dir path/to/output_dir
```
Custom directory for all generated figures and `summary.json`.

This is useful when:

- comparing multiple runs with different hyperparameters
- keeping outputs from the book figures separate from experiments
- generating clean artifacts for a report or presentation

### `--show`
```bash
--show
```
Displays plots interactively in addition to saving them.

This is mainly useful during local exploration. For automated runs, CI, or batch figure generation, you will usually leave it off.

---

## Hyperparameter guidance

If you only want one section to read before experimenting, make it this one.

### Best defaults for a normal run
The defaults are a good baseline:

```bash
python case_studies/ch2_value_iteration_vs_policy_iteration/run.py
```

They are tuned for a clean comparison on the two included environments and usually do not need to be changed.

### When to lower `--theta`
Use a smaller threshold when you want:

- tighter convergence
- more stable value comparisons
- cleaner agreement checks between VI and PI

Example:

```bash
python case_studies/ch2_value_iteration_vs_policy_iteration/run.py --theta 1e-5
```

Tradeoff: slower convergence and potentially more required iterations.

### When to raise `--theta`
Use a larger threshold for quick iteration while developing or debugging:

```bash
python case_studies/ch2_value_iteration_vs_policy_iteration/run.py --theta 1e-2
```

Tradeoff: faster runs, but rougher convergence.

### When to increase `--episodes`
Use more evaluation episodes when you care about reliable comparisons in the performance bar chart:

```bash
python case_studies/ch2_value_iteration_vs_policy_iteration/run.py --episodes 500
```

This is especially useful if you are comparing outputs across seeds.

### When to increase iteration caps
You may want to increase iteration caps when using a smaller threshold:

```bash
python case_studies/ch2_value_iteration_vs_policy_iteration/run.py \
  --theta 1e-5 \
  --vi-max-iterations 500 \
  --pi-max-iterations 2000
```

For the included environments, the defaults are usually enough, but raising the caps prevents premature stopping when you tighten convergence.

---

## Recommended commands

Here are the most useful commands for day-to-day use.

### 1. Standard baseline run
```bash
python case_studies/ch2_value_iteration_vs_policy_iteration/run.py
```
Use this for the default figures and metrics.

### 2. Tighter convergence study
```bash
python case_studies/ch2_value_iteration_vs_policy_iteration/run.py \
  --theta 1e-5 \
  --vi-max-iterations 500 \
  --pi-max-iterations 2000
```
Use this when you want a stricter convergence comparison.

### 3. More stable evaluation metrics
```bash
python case_studies/ch2_value_iteration_vs_policy_iteration/run.py \
  --episodes 1000
```
Use this when the main thing you care about is the quality of the agent-performance comparison.

### 4. Reproducible alternate seed
```bash
python case_studies/ch2_value_iteration_vs_policy_iteration/run.py \
  --seed 7
```
Use this to test sensitivity of the random baseline and sampled trajectory outputs.

### 5. Save outputs to a custom directory
```bash
python case_studies/ch2_value_iteration_vs_policy_iteration/run.py \
  --save-dir outputs/ch2_run_theta_1e5
```
Use this when comparing multiple runs side by side.

### 6. Interactive plotting
```bash
python case_studies/ch2_value_iteration_vs_policy_iteration/run.py --show
```
Use this for local inspection when you want plots to open on screen as they are generated.

---

## Reading the outputs

A good way to interpret a finished run is:

1. **Check agreement first**
   - VI and PI should match on non-terminal states in both test cases.
   - If they do not, the script returns a non-zero exit code.

2. **Look at convergence behavior**
   - VI’s convergence plot shows how quickly Bellman error shrinks.
   - PI’s convergence plot shows how quickly the policy stabilizes.

3. **Compare final policies and values**
   - The 5×5 policy and value plots make it easy to verify that both algorithms are finding the same solution structure.

4. **Check agent performance**
   - VI and PI should strongly outperform the random baseline on success rate and usually on reward and efficiency as well.

5. **Look at actual trajectories**
   - The path plots help connect abstract value/policy outputs to concrete behavior in the grid.

---

## Expected qualitative takeaways

For the environments in this script, the main takeaways are usually:

- **VI and PI should agree on the optimal policy** on the tested grids.
- **PI often stabilizes in fewer outer iterations**, though each iteration can be more involved.
- **VI gives a very transparent convergence story** through its Bellman-error curve.
- On the 5×5 grid, both learned policies should perform far better than the random baseline.

These are exactly the kinds of comparisons Chapter 2 is meant to highlight before later chapters move into more complex settings.

---

## Files produced by a successful run

A standard run writes the following artifacts:

```text
01_value_iteration_4x4.png
02_policy_iteration_4x4.png
03_vi_convergence_4x4.png
04_pi_convergence_4x4.png
05_vi_vs_pi_policy_5x5.png
06_vi_vs_pi_values_5x5.png
07_policy_evolution_5x5.png
08_agent_performance_5x5.png
09_vi_agent_path_5x5.png
10_random_best_path_5x5.png
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

### Plots save but do not appear on screen
That is expected unless you pass:

```bash
--show
```

### I only want quick validation, not polished metrics
Use a looser threshold and fewer episodes:

```bash
python case_studies/ch2_value_iteration_vs_policy_iteration/run.py \
  --theta 1e-2 \
  --episodes 20
```

---

## Summary

This Chapter 2 case study is a compact experimental comparison of **Value Iteration** and **Policy Iteration** on deterministic grid worlds. It emphasizes:

- what each algorithm learns
- how each algorithm converges
- whether their solutions agree
- how the resulting policies actually perform

For book readers, this makes it a practical bridge between the high-level ideas of dynamic programming and the more detailed treatment in the chapter text.

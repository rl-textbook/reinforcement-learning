# Chapter 5 Case Study: End-to-End RLHF Alignment from `run.py`

This case study turns Chapter 5 into a **full RLHF pipeline you can run from the command line**.

The script does not just train one model. It runs a staged comparison built around **Anthropic HH-RLHF** preference data:

- **SFT** as the supervised baseline
- a **reward model (RM)** trained on chosen vs rejected responses
- **DPO** starting from the SFT model
- **PPO (vanilla)** starting from the SFT model
- **PPO (mitigated)** starting from the same SFT model, but with more conservative PPO settings
- a final evaluation package with preference metrics, helpfulness and safety comparisons, qualitative examples, plots, CSV logs, and JSON summaries

Chapter 5 is a **pipeline chapter** where the run configuration determines:

- how expensive the whole experiment is
- whether PPO is stable enough to finish
- whether evaluation fits in memory
- how much of the story is about SFT vs DPO
- how much of the story is about PPO vanilla vs PPO mitigated

This README explains:

- what the script actually does
- what gets saved
- which CLI flags matter most
- how to choose **CPU vs GPU**, **smoke test vs real run**, and **single-PPO vs full comparison** settings
- which knobs affect **memory**, **runtime**, and **result quality** the most

---

## What this case study focuses on

At a high level, the chapter is asking:

**What changes when you take a small language model through the classic RLHF pipeline, and do simple PPO mitigations help?**

The current script is built around five comparison points.

### 1. SFT model
This is the first policy you actually evaluate as a generator.

The script fine-tunes a causal language model on the preferred responses. That gives you the supervised baseline before preference optimization or RL begins.

### 2. Reward model
Separately, the script trains a reward model on chosen vs rejected preference pairs.

This reward model is then used in two roles:

- as the **training signal** for PPO
- as a **helpfulness judge** during final evaluation

So the reward model is not just an intermediate artifact. It is part of both the optimization story and the evaluation story.

### 3. DPO model
The script trains a DPO policy starting from the SFT weights.

This gives you the “direct preference optimization” branch of the chapter.

### 4. PPO (vanilla)
The script also trains a PPO policy starting from the same SFT initialization.

This is the more classic RLHF branch: generate responses, score them with the reward model, and optimize with PPO while penalizing drift from the reference model.

### 5. PPO (mitigated)
The final policy branch is not a different trainer implementation. It is the **same PPO trainer** run with more conservative settings:

- a separate KL coefficient
- a repetition penalty
- reward clipping

This is one of the most important practical ideas in the script.
The chapter is not only comparing DPO vs PPO. It is also comparing:

```text
PPO (vanilla) vs PPO (mitigated)
```

and asking whether a few stabilization choices produce cleaner behavior.

So the Chapter 5 story is really:

```text
Base LM → SFT
Base LM → Reward Model
SFT → DPO
SFT → PPO (vanilla)
SFT → PPO (mitigated)
```

with final evaluation comparing the trained policy branches side by side.

---

## What the script actually does

The current `run.py` follows a clear staged workflow.

### Stage A. Parse CLI overrides and build the config
The script creates a `Config()` object and then automatically exposes every dataclass field as a CLI override.

That means most chapter settings can be changed from the command line with flags like:

```bash
--sft-steps
--rm-steps
--dpo-steps
--ppo-outer-steps
--max-length
--max-new-tokens
```

It also adds a few explicit extra flags such as:

```bash
--output-dir
--save-checkpoints
--disable-detoxify
--skip-vanilla-ppo
--skip-mitigated-ppo
--eval-source
--ppo-mitigated-kl-coef
--ppo-mitigated-repetition-penalty
--ppo-mitigated-reward-clip
```

### Stage B. Load tokenizer, device, and data
The script:

- sets the seed
- picks **CUDA if available**, otherwise CPU
- loads the configured tokenizer and model family
- forces **left padding** and **left truncation**
- clips `max_length` to the tokenizer’s real model limit
- loads the **`Anthropic/hh-rlhf`** training split

### Stage C. Build the SFT and preference splits
The script slices the dataset in sequence:

- the first `sft_train_size + sft_val_size` rows become the SFT pool
- the next `pref_train_size + pref_val_size` rows become the preference pool

Then it runs a **quality filter** over the preference pairs.

That matters a lot in practice:

- the filtered preference set can be smaller than requested
- the script will automatically **shrink `pref_train_size`** if needed
- if the filtered set is too small even to form validation, the run errors out

This means your requested dataset sizes are not always the exact sizes you end up training on.

### Stage D. Train SFT
The script loads the base causal LM and runs supervised fine-tuning on the chosen responses.

It logs:

- SFT loss during training
- a held-out **preference margin** after SFT finishes

That final SFT preference margin becomes an important horizontal reference in later DPO and PPO plots.

### Stage E. Train the reward model
Next the script loads a fresh base model and converts it into a reward model.

It trains with the standard pairwise ranking loss:

- chosen reward should be higher than rejected reward

It logs:

- RM loss
- RM train accuracy
- RM held-out accuracy at the end

### Stage F. Train DPO
The script loads a fresh policy model, initializes it from the SFT weights, and creates a frozen reference copy.

Then it runs DPO on the preference pairs.

It logs:

- DPO loss
- DPO train margin
- held-out preference margin on the probe set

### Stage G. Train PPO (vanilla)
If you do **not** pass `--skip-vanilla-ppo`, the script launches a PPO stage from the SFT initialization.

This branch uses:

- the SFT model as the starting policy
- a frozen SFT copy as the PPO reference policy
- the trained reward model for scalar rewards

It also includes some stability checks:

- skips steps with non-finite rollout statistics
- snapshots the last finite policy/value-head state
- restores the last good snapshot if PPO becomes non-finite

### Stage H. Train PPO (mitigated)
If you do **not** pass `--skip-mitigated-ppo`, the script runs the same PPO trainer again, but with the mitigated PPO knobs:

- `--ppo-mitigated-kl-coef`
- `--ppo-mitigated-repetition-penalty`
- `--ppo-mitigated-reward-clip`

This is not just cosmetic.
It is the script’s main “make PPO behave better” comparison.

### Stage I. Save logs, plots, and evaluation artifacts
After training, the script:

- writes per-stage CSV logs
- saves the chapter figures
- reloads the saved model states for evaluation
- checks for non-finite parameters
- optionally loads **Detoxify**
- builds general and harmful prompt sets
- generates final evaluation responses from all active policies
- computes helpfulness and safety comparisons
- writes a qualitative showcase text file
- writes a final `summary.json`

That means the output is much richer than “one final score.”

---

## What gets plotted and saved

This script is designed to leave behind a full experiment folder, not just console output.

### 1. Per-stage CSV logs
The script saves CSV logs for any stage that ran:

- `sft.csv`
- `rm.csv`
- `dpo.csv`
- `ppo_vanilla.csv`
- `ppo_mitigated.csv`

These are useful if you want to compare runs later in a notebook or spreadsheet.

### 2. SFT and RM curves
The script saves:

- `sft_loss_curve.png`
- `rm_loss_curve.png`
- `rm_train_accuracy_curve.png`

These are the simplest diagnostics:

- is SFT converging cleanly?
- is the reward model learning the chosen > rejected comparison?

### 3. DPO curves
The script saves:

- `dpo_preference_margin.png`
- `dpo_train_margin.png`
- `dpo_loss_curve.png`

These separate two related but different ideas:

- the **train-batch DPO signal**
- the **held-out preference margin** on the probe set

That distinction is useful because a DPO run can look active on train batches without clearly improving the held-out probe.

### 4. PPO diagnostic curves
The script saves several PPO comparison plots:

- `ppo_preference_margins.png`
- `ppo_approx_kl.png`
- `ppo_rm_raw_mean.png`
- `ppo_reward_mean.png`
- `ppo_response_length.png`
- `ppo_total_loss.png`

This is one of the best parts of the script.

Instead of treating PPO as a black box, the run gives you multiple ways to inspect what it is doing:

- **preference margin**: is PPO improving held-out preference behavior?
- **approx KL**: how far is PPO drifting from the reference policy?
- **raw RM score**: is the reward model score actually rising?
- **normalized reward**: what is PPO optimizing after penalties / normalization?
- **response length**: is PPO “improving” mostly by changing length?
- **total loss**: is the optimization behaving sensibly at all?

### 5. Helpfulness comparison artifacts
The script saves:

- `helpfulness_winrate_matrix.csv`
- `helpfulness_winrate_matrix.png`
- `mean_helpfulness_rm_score.png`

The win-rate matrix answers a question like:

**Using the reward model as the judge, how often does one policy beat another on the same prompts?**

That is more informative than a single average score because it shows pairwise comparisons between models.

### 6. Safety comparison artifacts
If **Detoxify** is available, the script saves:

- `safety_winrate_matrix.csv`
- `safety_winrate_matrix.png`
- `mean_safety_score.png`

If Detoxify is unavailable or you use `--disable-detoxify`, the script falls back to a refusal-regex proxy and instead saves:

- `safety_refusal_proxy_winrate_matrix.csv`
- `safety_refusal_proxy_winrate_matrix.png`
- `mean_refusal_proxy.png`

So the exact safety files depend on whether Detoxify is active.

### 7. Qualitative outputs
The script also writes:

- `qualitative_showcase.txt`

This file includes:

- a helpfulness-oriented showcase prompt
- a harmful-prompt showcase
- generated responses from the aligned policies
- the relevant helpfulness or safety scores

The qualitative showcase is especially valuable for chapter writing because it lets you inspect actual policy behavior rather than relying only on metrics.

### 8. Config, summary, and optional checkpoints
The script writes:

- `config.json`
- `summary.json`

And if you use `--save-checkpoints`, it also saves:

- `sft_state.pt`
- `dpo_state.pt`
- `ppo_vanilla_state.pt`
- `ppo_mitigated_state.pt`

This is useful for experiment management, but it can also consume a meaningful amount of disk space.

---

## What is actually being measured

The point of Chapter 5 is not one single metric.
The script measures several different things.

### SFT loss
This is the straightforward supervised-learning diagnostic.

It tells you whether the SFT stage is fitting the chosen responses cleanly.

### Reward model accuracy
The reward model tracks whether it scores the chosen response higher than the rejected one.

You get:

- train accuracy during RM training
- held-out RM accuracy at the end

This is the most direct check of whether the learned reward signal is doing anything sensible.

### Preference margin
This is one of the most important chapter metrics.

The script computes a preference margin by comparing response log-probability on the chosen sequence versus the rejected one.

It uses this for:

- final SFT evaluation
- DPO held-out tracking
- PPO held-out tracking

This is the main “is the policy preferring the preferred answer?” diagnostic.

### DPO train margin
DPO also logs a train-batch implicit reward margin.

That is useful because it gives you a more local view of the optimization signal, separate from the held-out probe.

### PPO KL drift
The PPO plots include an approximate KL-per-response-token metric.

This is extremely important in practice because PPO problems often look like:

- reward goes up
- but policy drift becomes too large
- or generation gets weird

So `ppo_approx_kl.png` is one of the first plots to inspect when PPO behaves strangely.

### PPO raw reward-model score and normalized reward
The PPO plots track both:

- the raw reward-model score
- the normalized / PPO reward that is actually being optimized

This matters because a policy can sometimes improve one without giving you the behavior you actually want.

### Response length
The script also tracks PPO response length.

That is a very useful anti-self-deception diagnostic.
If reward goes up mainly because the model is generating differently sized outputs, the length plot can reveal that.

### Helpfulness win-rates
The helpfulness matrix compares policies pairwise using the reward model as judge.

This is closer to a tournament view than a single scalar score.

### Safety win-rates
The safety matrix does something similar for harmful prompts:

- **Detoxify-based safety** if available
- otherwise a **refusal regex proxy**

This means the safety result is dependent on your evaluation mode, so do not compare Detoxify-based runs and refusal-proxy runs as if they were the same benchmark.

### Held-out preference accuracy summary
The final `summary.json` also stores held-out preference-style accuracies.

One subtle but important detail:

- **SFT** is measured with a raw log-likelihood preference test
- **DPO / PPO** are measured with an **implicit preference** comparison relative to the DPO reference formulation

So those summary fields are all useful, but they are not a single perfectly uniform metric family.

### Non-finite parameter checks
The final evaluation also checks whether any model parameters are non-finite.

That matters most for PPO.
If PPO becomes numerically unstable, the non-finite count is an immediate red flag.

---

## Why this chapter is especially compute-sensitive

Earlier chapters can often be summarized as “train one thing, then evaluate it.”

Chapter 5 is different.
A full run can include:

- SFT training
- reward-model training
- DPO training
- PPO vanilla training
- PPO mitigated training
- final evaluation across all active models

So the total cost is the sum of multiple expensive stages.

A useful rough mental model is:

```text
Total run cost
≈ SFT cost
+ RM cost
+ DPO cost
+ PPO vanilla cost
+ PPO mitigated cost
+ final multi-model evaluation cost
```

A second rough mental model is:

```text
SFT / RM / DPO cost ∝ steps × batch_size × sequence_length
PPO cost ∝ ppo_outer_steps × ppo_batch_size × ppo_inner_epochs × generation_length
```

That is not exact, but it is directionally very helpful.

There is also a Chapter 5 detail that matters a lot for hardware planning:

### Final evaluation can be the memory bottleneck
The script reloads multiple models onto the active device for final evaluation:

- SFT
- DPO
- PPO vanilla if present
- PPO mitigated if present
- reward model
- an extra reference model for preference evaluation
- possibly Detoxify too

So even if training fit comfortably, **the final comparison stage can still run out of memory**.

This is one of the most important practical differences between:

- a run that finishes all the way through
- a run that trains fine and then crashes during evaluation

Because of that, Chapter 5 is often more sensitive to:

- model size
- number of policy branches kept for final comparison
- whether Detoxify is enabled

than you might expect from the training loops alone.

---

## The most important hyperparameters and config choices

This is the most important section of the README.

Unlike some other chapters, Chapter 5 exposes **most config fields directly through the CLI**. That is good news: you can control a lot without editing Python.

But not every important behavior is a CLI flag, so it helps to think in two layers:

1. what you can override directly from the command line
2. what is fixed inside the helper logic

### 1. `--model-name`
This is one of the biggest practical knobs in the entire script.

Why it matters:

- bigger models increase training time
- bigger models increase memory use
- bigger models make final evaluation much heavier because multiple models are loaded simultaneously

For this Chapter 5 script, model size has **compound cost**, not just single-model cost.

A very practical takeaway:

- **GPT-2-scale models are the safest default for experimentation**
- moving to much larger models can make final evaluation harder than training

There is also a compatibility caveat: the reward-model construction reads `rm_base.transformer.wte.weight`, so GPT-style causal LMs are the safest replacements.

### 2. `--max-length`
This controls the main sequence length used in SFT, RM, and DPO tokenization.

Why it matters:

- longer sequences raise memory usage
- longer sequences slow every non-PPO training stage
- longer sequences also affect how many preference pairs survive filtering

This is one of the best first knobs to reduce if the chapter feels too heavy.

### 3. `--max-new-tokens`
This is the most important generation-length knob.

Why it matters:

- PPO repeatedly generates responses during training
- final evaluation also generates many responses
- longer generations are much slower
- longer generations usually make PPO the dominant cost sooner

If Chapter 5 feels slow, `--max-new-tokens` is usually one of the first things to reconsider.

### 4. `--batch-size`
This is the main batch size for:

- SFT
- reward-model training
- DPO
- preference validation loaders

Why it matters:

- it affects memory and throughput for most of the non-PPO stages
- it is usually the first knob to reduce when SFT, RM, or DPO hit memory limits

### 5. `--ppo-batch-size`
This is the PPO-specific batch size.

Why it matters:

- PPO uses its own prompt batch size
- larger PPO batches make generation and updates much heavier
- this is often the first PPO-specific memory knob to reduce

Keep `--batch-size` and `--ppo-batch-size` mentally separate.
They control different parts of the chapter.

### 6. `--sft-steps`, `--rm-steps`, `--dpo-steps`, `--ppo-outer-steps`
These are the main run-length knobs.

Why they matter:

- they directly control how long each training stage runs
- they are the cleanest way to switch between smoke tests and real runs

A very practical way to think about them is:

- `sft_steps`: how much supervised grounding the policy gets
- `rm_steps`: how good the learned reward judge can become
- `dpo_steps`: how much direct preference optimization you allow
- `ppo_outer_steps`: how much RLHF optimization PPO gets

If you only remember one runtime rule for Chapter 5, remember this:

**`ppo_outer_steps` is usually the most expensive single “make it longer” knob.**

### 7. `--ppo-inner-epochs`
This controls how many PPO update passes are taken per rollout batch.

Why it matters:

- more inner epochs make each PPO outer step heavier
- it can improve sample use, but it can also make PPO slower and touchier

If PPO is the bottleneck, reducing `--ppo-inner-epochs` is a clean way to lighten it.

### 8. `--sft-train-size`, `--sft-val-size`, `--pref-train-size`, `--pref-val-size`
These control how much of the HH-RLHF dataset is allocated to the two branches.

Why they matter:

- they affect how much unique data each stage sees before repeating
- they affect initial preprocessing time and memory
- `pref_val_size` directly affects held-out evaluation size

A subtle but important point:

Because the training loops are step-based rather than epoch-based, larger dataset slices do **not** increase runtime as directly as increasing the step counts.

Instead, bigger slices mostly buy you:

- more diverse data before the loader repeats
- more meaningful evaluation coverage

So for Chapter 5, if you want a faster run, reducing steps is often the first move.
Reducing dataset size is more about making the run smaller and more debuggable overall.

### 9. `--min-response-tokens`
This is one of the easiest Chapter 5 flags to overlook.

It affects the preference-pair quality filter.

Why it matters:

- if you set it too high, more pairs get filtered out
- the script may shrink `pref_train_size` automatically
- if too few pairs remain, the run can fail before training starts

This is especially important when you request a large preference split or when you reduce `max_length` aggressively.

### 10. `--beta-dpo`
This is the main DPO sharpness / scaling knob.

Why it matters:

- it changes how strongly DPO pushes preference differences relative to the reference
- it is one of the first things to tune if DPO feels too weak or too aggressive

In practice, this is one of the most chapter-specific “algorithm behavior” flags in the CLI.

### 11. `--kl-coef-ppo`
This is the KL penalty coefficient for vanilla PPO.

Why it matters:

- higher KL keeps PPO closer to the reference policy
- lower KL lets PPO move more aggressively
- this is one of the first knobs to inspect when PPO reward goes up but behavior gets weird

### 12. Mitigated PPO knobs
The most important mitigation-specific flags are:

```bash
--ppo-mitigated-kl-coef
--ppo-mitigated-repetition-penalty
--ppo-mitigated-reward-clip
```

These are central to the Chapter 5 PPO comparison.

Why they matter:

- KL coefficient controls how tightly the mitigated PPO branch stays near the reference
- repetition penalty discourages repetitive degeneration during generation
- reward clip limits extreme reward values and can help tame unstable PPO updates

If you care about the chapter’s PPO story, these are among the most important flags in the whole script.

### 13. `--pref-probe-size` and `--pref-probe-every`
These control the held-out preference probe used during DPO and PPO tracking.

Why they matter:

- larger probe size gives a less noisy held-out margin estimate
- smaller `pref_probe_every` gives denser measurements but adds more overhead

These are mostly **measurement knobs**, not learning knobs.

### 14. `--eval-benign-prompts` and `--eval-harmful-prompts`
These control how many general and harmful prompts are used in final evaluation.

Why they matter:

- larger values make the final comparison more stable and meaningful
- larger values also slow final generation and judging

These are especially important because Chapter 5 final evaluation can already be memory-heavy.

### 15. `--showcase-num-candidates`
This controls how many candidate responses are generated for qualitative showcase selection.

Why it matters:

- larger values can produce a nicer qualitative example
- larger values also add evaluation cost

This is an analysis-quality knob, not a training-quality knob.

### 16. `--eval-source`
This has two choices:

```bash
--eval-source pref_val
--eval-source combined
```

How to think about them:

- `pref_val` is the cleaner held-out default
- `combined` broadens the prompt pool used for final evaluation

In practice, `pref_val` is the safer setting when you want a more benchmark-like evaluation story.
`combined` is more like a broader stress test of the filtered preference pool.

### 17. `--disable-detoxify`
This flag matters more than it looks.

Why it matters:

- it changes the safety evaluation mode
- it can simplify setup
- it can reduce evaluation overhead and resource pressure

Use this when:

- Detoxify is not installed
- Detoxify installation is awkward in your environment
- you want the cheaper refusal-proxy fallback

### 18. `--skip-vanilla-ppo` and `--skip-mitigated-ppo`
These are some of the most useful practical flags in the entire chapter.

Why they matter:

- PPO is expensive
- running both PPO branches doubles that part of the workload
- skipping one PPO branch can make local experimentation dramatically easier

This is the biggest Chapter 5-specific simplification switch.

If you only care about the “better behaved PPO” story, using:

```bash
--skip-vanilla-ppo
```

is often a very sensible local-run choice.

### 19. `--output-dir` and `--save-checkpoints`
These are operational, but still important.

Use `--output-dir` to keep runs separate.
That is highly recommended because the filenames are fixed and otherwise easy to overwrite.

Use `--save-checkpoints` only when you really want the `.pt` state files.
That can be helpful, but it also increases disk usage.

---

## Important settings that are **not** fully exposed through the CLI

The Chapter 5 script exposes a lot, but not everything.

A few important behaviors are currently hard-coded in the script or helper logic.

### 1. Safe evaluation generation settings
The evaluation helper uses fixed generation recipes:

- a greedy pass first
- sampled “rescue” generations if the response looks degenerate
- fixed top-p, top-k, and repetition settings for the rescue path
- fixed `no_repeat_ngram_size=4`

That means reproducibility is not controlled by `max_new_tokens` alone.

### 2. Prompt-only generation truncates to 128 tokens
The prompt tokenization used for PPO prompt loading and single-prompt generation is capped at 128 tokens inside the script.

That is an important hidden behavior if you experiment with longer prompts or different datasets.

### 3. Final qualitative ranking heuristics
The script uses custom quality heuristics, refusal heuristics, and candidate-selection logic for the showcase file.

Those are useful and practical, but they are not exposed as CLI knobs.

### 4. `--eval-batch-size` is less influential than it sounds
The final guarded generation helper currently loops over prompts one at a time.

So while `eval_batch_size` exists in the config and is passed into the helper, it is **not** the big throughput lever you might expect from the name.

For Chapter 5, the more important final-eval knobs are usually:

- number of models being compared
- number of evaluation prompts
- `max_new_tokens`
- whether Detoxify is enabled

---

## Requirements

From the chapter directory, the script’s own docstring expects:

```bash
pip install -e '.[alignment]'
pip install transformers datasets pandas matplotlib
```

Optional safety evaluation dependency:

```bash
pip install detoxify
```

Then run the chapter script:

```bash
python run.py
```

If your repository structure keeps this file elsewhere, use the corresponding path from the repo root.

---

## Default runtime behavior at a glance

A few defaults are explicit in the script itself:

- `--output-dir case_study_outputs`
- `--eval-source pref_val`
- `--ppo-mitigated-kl-coef 0.05`
- `--ppo-mitigated-repetition-penalty 1.1`
- `--ppo-mitigated-reward-clip 5.0`

The script also automatically:

- uses **CUDA** if available, otherwise CPU
- sets tokenizer padding and truncation to the left side
- clips `max_length` to the tokenizer’s real model limit
- writes `config.json` to the output folder

And one more very practical detail:

### The selected dataset rows are deterministic
The script selects fixed ranges from the HH-RLHF training split before training begins.

That means:

- `--seed` affects optimization randomness and shuffling
- but `--seed` does **not** change which raw dataset rows are selected for SFT and preference training

If you want a different initial slice of data, that is a code-level change, not just a seed change.

---

## The most important command-line arguments

It helps to group the Chapter 5 CLI into categories.

## A. Model and data-shape flags

### `--model-name`
```bash
--model-name gpt2
```
Safest major knob for fitting the full pipeline on your hardware.

### `--max-length`
```bash
--max-length 384
```
Controls tokenized sequence length for SFT, RM, and DPO.

### `--max-new-tokens`
```bash
--max-new-tokens 96
```
Controls generated response length for PPO and final evaluation.

### `--sft-train-size`
```bash
--sft-train-size 1000
```
Controls the size of the SFT slice.

### `--sft-val-size`
```bash
--sft-val-size 128
```
Reserves additional rows in the SFT block before the preference block begins.

### `--pref-train-size`
```bash
--pref-train-size 1500
```
Controls the requested preference-train slice before filtering.

### `--pref-val-size`
```bash
--pref-val-size 128
```
Controls the held-out preference validation size.

### `--min-response-tokens`
```bash
--min-response-tokens 16
```
Controls preference-pair filtering strictness.

## B. Main training-size flags

### `--batch-size`
```bash
--batch-size 4
```
Main batch size for SFT, RM, DPO, and preference validation.

### `--ppo-batch-size`
```bash
--ppo-batch-size 4
```
PPO-specific rollout / update batch size.

### `--sft-steps`
```bash
--sft-steps 100
```
Length of SFT training.

### `--rm-steps`
```bash
--rm-steps 150
```
Length of reward-model training.

### `--dpo-steps`
```bash
--dpo-steps 150
```
Length of DPO training.

### `--ppo-outer-steps`
```bash
--ppo-outer-steps 75
```
Length of PPO training. Usually the heaviest single stage knob.

### `--ppo-inner-epochs`
```bash
--ppo-inner-epochs 2
```
Controls PPO update work per outer step.

## C. Optimization-specific flags

### `--sft-lr`
```bash
--sft-lr 1e-5
```
SFT learning rate.

### `--rm-lr`
```bash
--rm-lr 1e-5
```
Reward-model learning rate.

### `--warmup-fraction`
```bash
--warmup-fraction 0.1
```
Reward-model LR warmup fraction.

### `--dpo-lr`
```bash
--dpo-lr 1e-6
```
DPO learning rate.

### `--beta-dpo`
```bash
--beta-dpo 0.15
```
Main DPO preference-scaling knob.

### `--kl-coef-ppo`
```bash
--kl-coef-ppo 0.1
```
Vanilla PPO KL regularization coefficient.

### Mitigated PPO flags
```bash
--ppo-mitigated-kl-coef 0.05
--ppo-mitigated-repetition-penalty 1.1
--ppo-mitigated-reward-clip 5.0
```
These define the “PPO (mitigated)” branch.

## D. Evaluation and analysis flags

### `--pref-probe-size`
```bash
--pref-probe-size 128
```
Held-out probe size for DPO/PPO preference-margin tracking.

### `--pref-probe-every`
```bash
--pref-probe-every 25
```
How often DPO and PPO record held-out preference margins.

### `--eval-benign-prompts`
```bash
--eval-benign-prompts 24
```
General-prompt evaluation size.

### `--eval-harmful-prompts`
```bash
--eval-harmful-prompts 24
```
Harmful-prompt evaluation size.

### `--eval-source`
```bash
--eval-source pref_val
```
Use the held-out preference validation rows as the prompt source.

```bash
--eval-source combined
```
Use a broader filtered preference pool for prompt selection.

### `--showcase-num-candidates`
```bash
--showcase-num-candidates 3
```
How many response candidates to consider for qualitative showcase selection.

## E. Operational flags

### `--output-dir`
```bash
--output-dir results/ch5_trial_a
```
Where all figures, JSON, text, CSV, and optional checkpoints are written.

### `--save-checkpoints`
```bash
--save-checkpoints
```
Save `.pt` state files for the trained policy stages.

### `--disable-detoxify`
```bash
--disable-detoxify
```
Use the refusal-proxy safety fallback instead of Detoxify.

### `--skip-vanilla-ppo`
```bash
--skip-vanilla-ppo
```
Do not run the vanilla PPO branch.

### `--skip-mitigated-ppo`
```bash
--skip-mitigated-ppo
```
Do not run the mitigated PPO branch.

### `--seed`
```bash
--seed 123
```
Controls training randomness and loader shuffling, not the initial raw row selection.

---

## Recommended run profiles

This chapter benefits a lot from intentional run profiles.

The most useful distinction is not just “small vs large.”
It is also:

- **all stages vs one PPO branch**
- **Detoxify vs refusal proxy**
- **tiny sanity run vs comparison run**

Below are practical starting points.

## 1. CPU smoke test or laptop sanity check
Use this when:

- you want to make sure the full pipeline basically works
- you are on CPU only
- you are validating code changes
- you want cheap end-to-end artifacts before committing to a real run

Example:

```bash
python run.py \
  --model-name distilgpt2 \
  --max-length 256 \
  --max-new-tokens 48 \
  --batch-size 2 \
  --ppo-batch-size 2 \
  --sft-train-size 128 \
  --sft-val-size 32 \
  --pref-train-size 128 \
  --pref-val-size 32 \
  --sft-steps 20 \
  --rm-steps 20 \
  --dpo-steps 20 \
  --ppo-outer-steps 10 \
  --ppo-inner-epochs 1 \
  --pref-probe-size 32 \
  --pref-probe-every 10 \
  --eval-benign-prompts 6 \
  --eval-harmful-prompts 6 \
  --showcase-num-candidates 2 \
  --disable-detoxify \
  --output-dir results/ch5_cpu_smoke
```

This is **not** the setting to judge final chapter performance.
It is the setting to confirm the whole Chapter 5 pipeline can complete.

## 2. Modest local GPU run with a single PPO branch
Use this when:

- you have a modest local GPU
- you want a meaningful run without paying for both PPO branches
- you mainly care about SFT vs DPO vs mitigated PPO

Example:

```bash
python run.py \
  --model-name gpt2 \
  --max-length 384 \
  --max-new-tokens 64 \
  --batch-size 4 \
  --ppo-batch-size 4 \
  --sft-train-size 1000 \
  --sft-val-size 128 \
  --pref-train-size 1500 \
  --pref-val-size 128 \
  --sft-steps 100 \
  --rm-steps 150 \
  --dpo-steps 150 \
  --ppo-outer-steps 75 \
  --ppo-inner-epochs 2 \
  --pref-probe-size 64 \
  --pref-probe-every 25 \
  --eval-benign-prompts 16 \
  --eval-harmful-prompts 16 \
  --showcase-num-candidates 3 \
  --skip-vanilla-ppo \
  --disable-detoxify \
  --output-dir results/ch5_local_gpu
```

This is often the most sensible Chapter 5 local template because it removes the most expensive extra branch while still preserving the main comparison story.

## 3. Colab / T4-style full chapter comparison
Use this when:

- you want the full SFT vs DPO vs PPO vanilla vs PPO mitigated story
- you have a midrange cloud GPU
- you want the chapter figures, not just a debug pass

Example:

```bash
python run.py \
  --model-name gpt2 \
  --max-length 384 \
  --max-new-tokens 96 \
  --batch-size 4 \
  --ppo-batch-size 4 \
  --sft-train-size 1500 \
  --sft-val-size 128 \
  --pref-train-size 2500 \
  --pref-val-size 256 \
  --sft-steps 150 \
  --rm-steps 200 \
  --dpo-steps 200 \
  --ppo-outer-steps 100 \
  --ppo-inner-epochs 2 \
  --pref-probe-size 128 \
  --pref-probe-every 25 \
  --eval-benign-prompts 24 \
  --eval-harmful-prompts 24 \
  --showcase-num-candidates 3 \
  --output-dir results/ch5_full_compare
```

If this kind of run trains successfully but crashes during evaluation, the first things to try are:

- `--disable-detoxify`
- `--skip-vanilla-ppo`
- a smaller `--model-name`

## 4. Stronger-GPU, fuller benchmark-style GPT-2-scale run
Use this when:

- you care more about the chapter comparison than cheap iteration
- you have room for a longer run
- you still want to stay within the script’s practical multi-model evaluation limits

Example:

```bash
python run.py \
  --model-name gpt2 \
  --max-length 512 \
  --max-new-tokens 128 \
  --batch-size 8 \
  --ppo-batch-size 8 \
  --sft-train-size 3000 \
  --sft-val-size 256 \
  --pref-train-size 5000 \
  --pref-val-size 512 \
  --sft-steps 250 \
  --rm-steps 300 \
  --dpo-steps 300 \
  --ppo-outer-steps 150 \
  --ppo-inner-epochs 2 \
  --pref-probe-size 256 \
  --pref-probe-every 25 \
  --eval-benign-prompts 32 \
  --eval-harmful-prompts 32 \
  --showcase-num-candidates 4 \
  --save-checkpoints \
  --output-dir results/ch5_benchmark
```

Notice that even this “stronger” run still stays at a GPT-2-scale model.
That is deliberate.
Because final evaluation keeps multiple models resident at once, Chapter 5 usually benefits more from **more steps and better evaluation coverage** than from aggressively increasing model size.

---

## CPU vs GPU guidance

### CPU-only
CPU is fine for:

- smoke tests
- debugging
- checking file outputs
- validating the pipeline after code edits

CPU is generally **not** the right way to judge the full chapter comparison, because PPO generation and the final multi-model evaluation become slow quickly.

If you are on CPU:

- use a small `--model-name`
- reduce `--max-length`
- reduce `--max-new-tokens`
- keep `--batch-size` and `--ppo-batch-size` small
- keep all step counts small
- consider disabling Detoxify
- consider skipping one PPO branch

### Modest GPU
A modest GPU is usually enough for meaningful Chapter 5 work if you keep the setup disciplined.

Best priorities:

1. stay at GPT-2 scale
2. control `--max-length`
3. control `--max-new-tokens`
4. use small-to-moderate batch sizes
5. skip one PPO branch if needed
6. disable Detoxify first if evaluation gets tight

### Stronger GPU
A stronger GPU gives you room for:

- larger evaluation prompt sets
- more PPO steps
- longer sequence lengths
- both PPO branches in the same run
- checkpoint saving without worrying about reruns

But Chapter 5 still has one unusual constraint:

**Final evaluation can punish large models more than you expect.**

So even on stronger hardware, the safest way to improve the chapter run is often:

- keep model size modest
- scale steps and evaluation coverage first

rather than immediately scaling the model itself.

---

## Tuning advice: what to change first

When the script is too slow, too memory-heavy, or too weak, it helps to have an order.

### If SFT, RM, or DPO runs out of memory
Change these first:

1. reduce `--batch-size`
2. reduce `--max-length`
3. reduce `--model-name`
4. reduce dataset slice sizes if preprocessing is also too heavy

### If PPO is too slow or too memory-heavy
Change these first:

1. reduce `--ppo-batch-size`
2. reduce `--max-new-tokens`
3. reduce `--ppo-outer-steps`
4. reduce `--ppo-inner-epochs`
5. skip one PPO branch

### If final evaluation crashes or OOMs
Try these in roughly this order:

1. `--disable-detoxify`
2. `--skip-vanilla-ppo` or `--skip-mitigated-ppo`
3. smaller `--model-name`
4. smaller `--eval-benign-prompts` and `--eval-harmful-prompts`
5. smaller `--showcase-num-candidates`

### If the filtered preference set is unexpectedly too small
Try:

1. lower `--min-response-tokens`
2. increase `--pref-train-size`
3. reduce `--pref-val-size`
4. reduce `--max-length` less aggressively if truncation is filtering pairs out

### If DPO or PPO results feel too weak
Improve these first:

1. strengthen SFT with more `--sft-steps`
2. improve the reward model with more `--rm-steps`
3. increase `--dpo-steps`
4. increase `--ppo-outer-steps`
5. tune `--beta-dpo` and PPO KL settings
6. only then enlarge evaluation sizes for nicer reporting

### If the DPO / PPO curves are too noisy to interpret
Try:

- larger `--pref-probe-size`
- more `--pref-val-size`
- slightly less frequent probing by increasing `--pref-probe-every`
- more total training steps

---

## Suggested workflow for serious Chapter 5 experimentation

A good practical workflow is:

### Step 1. Run a tiny sanity check
Make sure every stage completes and the output folder is populated.

### Step 2. Run a single-PPO comparison locally
Use SFT + DPO + one PPO branch to get meaningful results cheaply.

### Step 3. Decide whether the question is about code correctness or algorithm comparison
If the code works, then scale the run only in the direction that matches your question:

- more PPO steps if the RL story matters
- more RM steps if judging seems weak
- more evaluation prompts if the chapter writing needs stronger comparison plots

### Step 4. Only then run the full four-policy comparison
Once you know the script is stable, compare:

- SFT
- DPO
- PPO vanilla
- PPO mitigated

### Step 5. Keep each run in its own output folder
This chapter produces enough artifacts that overwriting old results becomes confusing very quickly.

---

## Interpreting the main results

A few comparisons are especially useful.

### SFT vs DPO
This tells you what direct preference optimization buys you over the supervised baseline.

If DPO improves held-out preference margin and helpfulness without wrecking safety, that is a strong chapter result.

### SFT vs PPO
This tells you whether RLHF with the learned reward model is actually adding value beyond SFT.

If PPO gets higher RM scores but worse safety or uglier qualitative outputs, that is also an important chapter result.

### PPO vanilla vs PPO mitigated
This is one of the most interesting comparisons in the script.

Useful questions:

- does mitigated PPO keep KL more controlled?
- does it reduce repetition or degeneration?
- does it keep more safety while staying competitive on helpfulness?

### Helpfulness vs safety
Chapter 5 is not just about maximizing helpfulness.

If one policy wins the RM-judged helpfulness matrix but loses badly on harmful prompts, that tradeoff is part of the actual story.

### Held-out preference accuracy vs RM-judged helpfulness
These are related, but not identical.

A policy can look better on one and not dominate on the other.
That is why the script keeps both.

### Qualitative showcase
Never ignore the showcase text file.

The whole point of Chapter 5 is to compare real alignment behavior, not only curves.
If a model looks strong numerically but its actual responses look brittle, repetitive, or evasive in a bad way, the qualitative file will often reveal it.

---

## Outputs you will usually want to keep

For chapter writing or experiment tracking, the most important artifacts are usually:

- `summary.json`
- `helpfulness_winrate_matrix.png`
- the relevant safety matrix PNG
- `dpo_preference_margin.png`
- `ppo_preference_margins.png`
- `ppo_approx_kl.png`
- `qualitative_showcase.txt`

If you are iterating heavily, it is also smart to keep:

- `config.json`
- the stage CSV logs

Those make it much easier to compare runs later.

---

## Practical bottom line

If you want the shortest useful summary of how to run Chapter 5 well, it is this:

- start with a **small GPT-2-scale model**
- remember that Chapter 5 is a **multi-stage pipeline**, not one trainer
- treat **`max_length`** and **`max_new_tokens`** as core compute multipliers
- treat **`batch-size`** and **`ppo-batch-size`** as the main memory knobs
- treat **`ppo_outer_steps`** as the main PPO runtime knob
- use **`--skip-vanilla-ppo`** or **`--skip-mitigated-ppo`** aggressively for local work
- use **`--disable-detoxify`** when evaluation setup or memory becomes annoying
- prefer **`--eval-source pref_val`** when you want the cleaner held-out evaluation story
- remember that **final evaluation may be the memory bottleneck, not training**

That is the core practical lesson of the current Chapter 5 `run.py`.

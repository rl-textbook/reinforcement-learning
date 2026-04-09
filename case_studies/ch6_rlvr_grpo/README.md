# Chapter 6 Case Study: RLVR with GRPO on GSM8K

This case study moves from classic RL into a much more modern setup: **reinforcement learning with verifiable rewards** for language models.

The script compares three stages of the same overall pipeline on **GSM8K**:

- a **base language model**
- that same model after **supervised fine-tuning (SFT)** on chain-of-thought style reasoning
- the SFT model after **GRPO-based RLVR** training

The point of the chapter is not to turn the README into a theory lecture. The book handles the deeper RLVR ideas, the GRPO math, and the reasoning-model background. Here, the focus is practical:

- what the script is doing
- what gets plotted and measured
- which hyperparameters matter most
- which `python run.py --...` commands are the most useful
- how to think about **CPU vs GPU**, **local vs Colab**, and **small vs benchmark-style runs**

This is one of the first chapters where **compute configuration is part of the experiment design**, not just a convenience issue. A Chapter 6 run can feel tiny or expensive depending on a few core knobs, especially:

- model size
- token cap
- GRPO group size
- prompts per step
- evaluation batch size
- dataset slice size

---

## What this case study focuses on

At a high level, the chapter asks a simple question:

**Can verifiable RL improve mathematical reasoning beyond ordinary supervised fine-tuning?**

The script answers that by comparing three stages.

### 1. Base model
This is the untuned starting model loaded from the configuration.

It gives you the baseline performance before any task-specific learning.

### 2. SFT model
The script next runs **supervised fine-tuning** on reasoning-style demonstrations.

This stage teaches the model the format and structure of the task more directly. In practice, SFT is the bridge between “generic language model” and “task-oriented math reasoner.”

### 3. GRPO / RLVR model
After SFT, the script runs **Group Relative Policy Optimization (GRPO)** with a **verifiable math reward**.

Instead of relying on a learned reward model, the setup uses a verifier that can check whether the final numeric answer is correct. That makes the reward signal far more concrete than in many open-ended RLHF-style pipelines.

So the Chapter 6 story is really:

```text
Base model → SFT → RLVR with GRPO
```

and the experiment is asking how much each stage improves measurable reasoning performance.

---

## What the script actually does

The current `run.py` follows a clean staged workflow.

### Stage A. Load data and model
The script:

- loads **GSM8K** with `datasets`
- selects train and test subsets from the configured sizes
- builds a **formatter** for prompt / answer structure
- builds a **math verifier** for exact-answer checking
- loads the configured model and tokenizer
- chooses **CUDA if available**, otherwise CPU

### Stage B. Evaluate the base model
Before any training, the script evaluates the raw model on the held-out test set.

That produces:

- **Pass@k** metrics
- diagnostic reasoning metrics
- compute-scaling metrics

This baseline matters because it tells you how much of the final result comes from SFT versus GRPO.

### Stage C. Run SFT
The script then fine-tunes the model on the training subset using supervised chain-of-thought style learning.

### Stage D. Evaluate the SFT model
After SFT, the script evaluates again using the same style of metrics.

It also caches a subset of **early trajectories** that will later be compared against final GRPO trajectories.

### Stage E. Run GRPO
The script then launches the GRPO training loop.

This is the expensive part of the chapter. Each GRPO step involves sampling grouped completions, verifying them, and updating the model based on relative within-group performance.

### Stage F. Final evaluation and artifacts
After RLVR training, the script:

- evaluates the final model again
- builds the chapter plots
- writes summary tables and JSON results
- prints example “early vs late” reasoning trajectories

That means the output is not just one score. It is a compact benchmark package.

---

## What gets plotted and saved

This script is much more than a one-line training command. It saves multiple figures and structured result files.

### 1. Learning curves across GRPO training
The script saves:

- `CaseStudy_Fig1_LearningCurves_v3.png`

This figure has three panels:

- **Pass@1 vs GRPO step**
- **Pass@4 vs GRPO step**
- **Pass@8 vs GRPO step**

Each panel includes:

- the **GRPO curve over training**
- a horizontal reference for the **base model**
- a horizontal reference for the **SFT model**

This is the main “did RLVR help?” figure.

It shows whether GRPO is improving over the SFT starting point and how that improvement changes as you allow more test-time samples.

### 2. Test-time compute scaling
The script saves:

- `CaseStudy_Fig2_TestTimeScaling.png`

This figure compares measured **Pass@N** as you increase the number of test-time samples.

It includes curves for:

- Base
- SFT
- GRPO
- an idealized **independent-samples** reference curve for GRPO

This is one of the most useful plots in the chapter because it separates two different questions:

- did the model get better at generating good answers at all?
- how much additional value do you get from **best-of-N sampling** at inference time?

### 3. Reasoning-quality evolution
The script saves:

- `CaseStudy_Fig3_ReasoningQualityEvolution.png`

This figure tracks diagnostic reasoning signals over GRPO training. The saved plot uses three panels:

- **response length in tokens**
- **reasoning steps** using a heuristic counter
- **math markers** using a heuristic signal

Like the learning-curve figure, it also compares GRPO against horizontal baselines from Base and SFT.

This plot is especially useful because it shows that Chapter 6 is not only about final accuracy. It is also about how the model’s reasoning traces change over training.

### 4. Summary tables and machine-readable outputs
The script also writes:

- `chapter6_summary.csv`
- `chapter6_results.json`

The summary table compares:

- Base
- SFT
- GRPO

and includes columns such as:

- `Pass@1`
- `Pass@4`
- `Pass@8`
- `LenTok`
- `Steps`
- `SelfCorr`

The JSON file stores the config and the structured evaluation outputs, which is helpful if you want to compare runs later without manually copying console logs.

### 5. Qualitative trajectory examples
The script also prints example reasoning trajectories to the console.

It tries to find cases where:

- the **early policy** fails
- the **late policy** succeeds

Then it prints the prompt, ground truth, earlier response, and later response side by side.

That is extremely helpful for chapter writing because it turns the study from “just metrics” into something you can actually inspect qualitatively.

---

## What is actually being measured

The main point of this chapter is not a single headline number. The script measures several different things.

### Pass@k
This is the core quantitative metric.

The current run evaluates at least:

- **Pass@1**
- **Pass@4**
- **Pass@8**

These tell you how often the model gets the answer right when allowed 1, 4, or 8 sampled attempts.

That matters a lot for reasoning models because the quality of the policy and the value of test-time sampling are closely connected but not identical.

### Full-test versus curve-subset evaluation
The script evaluates both:

- a smaller **curve subset** for repeated measurement during training
- the **full held-out test** for more complete reporting

This is a practical compromise between accuracy and runtime.

### Test-time compute scaling
The script also measures how performance changes as you increase the number of samples drawn at inference time.

This matters because one model may have only modest Pass@1 improvement but a much stronger best-of-N profile.

### Diagnostic reasoning signals
The script tracks several analysis-style metrics, including:

- output length in tokens
- heuristic reasoning-step count
- heuristic math-marker count
- self-correction rate in the summary table

These are not replacements for correctness, but they are useful for seeing whether the model is merely longer, genuinely more structured, or more likely to correct itself.

### Early-vs-late trajectory improvement
The printed qualitative examples let you inspect whether later GRPO behavior looks more careful, more structured, or more verifier-aligned than earlier SFT behavior.

---

## Why this chapter is much more compute-sensitive

In earlier chapters, you could often “just run the script” and wait a bit longer.

Chapter 6 is different.

A GRPO step is not cheap because it combines:

- prompt formatting
- autoregressive generation
- grouped sampling
- verifier scoring
- backward passes for policy updates
- repeated evaluation passes during training

So the total cost is heavily affected by a few interacting terms:

- **model size**
- **number of prompts used**
- **group size**
- **token cap / generation length**
- **evaluation frequency**
- **evaluation subset size**

A useful mental model is:

```text
GRPO training cost ∝ prompts_per_step × group_size × generation_length × number_of_steps
```

It is not exact, but it is directionally very helpful.

That is why this chapter benefits so much from thoughtful configuration.

---

## The most important hyperparameters and config choices

This is the most important section of the README for Chapter 6.

Not every important setting is exposed as a CLI flag. The chapter uses a mix of:

- **CLI overrides** in `run.py`
- deeper defaults stored in **`CFG()`**

So when you tune Chapter 6, think in two layers:

1. what you can override directly from the command line
2. what you may want to edit in the underlying config

### 1. Group size
The most important GRPO knob is:

```bash
--group-size
```

This controls how many completions are sampled per prompt for the GRPO update.

Why it matters:

- Larger groups usually give a stronger relative-comparison signal inside GRPO.
- Larger groups also increase sampling cost and memory pressure very quickly.
- For this chapter, **group size is one of the biggest levers for both quality and runtime**.

Practical guidance:

- **`group_size = 2`** is a local-friendly setting and a sensible default when trying to keep Chapter 6 runnable on smaller hardware.
- **`group_size = 8`** is much closer to a stronger benchmark-style setting and usually gives the best results, but it is significantly more expensive.

A very important intuition here is that moving from 2 to 8 is not a tiny change. It means **4× more sampled completions per prompt** during GRPO.

### 2. Token cap / max tokens
A recent practical change for local runs is keeping the generation cap at **96 tokens**.

This is one of the most valuable changes for making the chapter usable on everyday hardware.

Why it matters:

- Longer generations increase wall-clock time.
- Longer generations also increase memory use.
- In grouped RLVR training, that cost multiplies across all samples in the group.

So even if group size stays fixed, letting generations get too long can make Chapter 6 much slower than expected.

If you only remember two practical Chapter 6 tuning choices, remember these:

- **group size**
- **token cap**

Together, they control a huge fraction of the compute budget.

### 3. Prompts per step
The main CLI flag is:

```bash
--prompts-per-step
```

This controls how many prompts are used in each training step.

Why it matters:

- Larger values can give a richer update.
- Larger values also multiply total sampled completions because the GRPO update now spans more prompts.

A helpful way to think about this is:

```text
effective samples per RL step = prompts_per_step × group_size
```

That product is often the clearest single number for how heavy a GRPO update really is.

### 4. GRPO steps
The main CLI flag is:

```bash
--grpo-steps
```

This controls how many RLVR update steps are run.

Why it matters:

- More steps usually give a better chance for the GRPO curve to improve.
- More steps also directly increase total runtime.
- If you want stronger benchmark-style results, increasing `grpo_steps` is often necessary.

But increasing `grpo_steps` without controlling group size or token length can make runs expensive very quickly.

### 5. SFT steps
The main CLI flag is:

```bash
--sft-steps
```

This controls how much supervised adaptation happens before RLVR starts.

Why it matters:

- A stronger SFT starting point usually makes GRPO more stable.
- Too little SFT can make GRPO spend time fixing basic format or reasoning issues rather than improving actual answer quality.

In practice, SFT is not just a warm-up stage. It strongly shapes what GRPO can build on.

### 6. Train and test subset sizes
The main CLI flags are:

```bash
--train-n
--test-n
```

These control how many GSM8K examples are selected from the full train and test splits.

Why they matter:

- Smaller subsets are much cheaper and are ideal for debugging.
- Larger subsets give more trustworthy metrics and stronger training signals.

If you are deciding between “more steps” and “more data,” the right choice depends on whether you are debugging or benchmarking:

- for debugging, shrink the dataset first
- for benchmark-style reporting, keep the dataset meaningfully sized before you chase extra steps

### 7. Evaluation frequency
The main CLI flag is:

```bash
--eval-every
```

This controls how often the GRPO loop evaluates the model during training.

Why it matters:

- Smaller values give a denser, nicer learning curve.
- Smaller values also add more evaluation overhead.
- Larger values train faster between evaluations, but the learning curve becomes coarser.

This is mostly a **measurement-cost** knob, not a learning-quality knob.

### 8. Evaluation curve subset size
The main CLI flag is:

```bash
--eval-curve-n
```

This controls how many held-out prompts are used for the repeated curve evaluations during training.

Why it matters:

- Larger values make the GRPO curve less noisy.
- Larger values also slow evaluation substantially.

This is one of the most useful places to save time without changing training itself.

### 9. Evaluation batch size
The main CLI flag is:

```bash
--batch-prompts-eval
```

This controls how many prompts are evaluated together in a batch during Pass@k and diagnostic evaluation.

Why it matters:

- Larger values improve throughput if you have enough VRAM.
- Larger values can cause OOM issues on smaller GPUs.
- Smaller values are safer and usually the first thing to reduce when evaluation fails on memory.

This flag is mostly about **evaluation memory and throughput**, not about the actual learning dynamics.

### 10. Gradient checkpointing
The main CLI flag is:

```bash
--gradient-checkpointing
```

This is one of the most useful practical switches for this chapter.

Why it matters:

- It reduces memory pressure.
- It often makes the difference between “this fits on my GPU” and “this crashes.”
- The tradeoff is extra compute time.

For Chapter 6, that tradeoff is often worth it.

### 11. Model choice
The main CLI flag is:

```bash
--model-name
```

This overrides the model from the chapter config.

Why it matters:

- Bigger models usually cost more memory and time.
- Smaller models are much easier for local iteration.
- Model choice changes the entire practical feel of the chapter more than almost any single optimization flag.

If a chapter run feels impossible on your hardware, model size is one of the first things to reconsider.

### 12. Qualitative-analysis controls
The most useful flags are:

```bash
--traj-subset-n
--qualitative-k
```

These do not change the model itself. They control how the script searches for and prints trajectory examples.

Why they matter:

- Larger `traj-subset-n` gives the script more held-out prompts to search for “early fail, late success” examples.
- Larger `qualitative-k` prints more examples to inspect.

These are analysis knobs, but they still affect runtime.

---

## Important settings that live in `CFG()` rather than the CLI

The current CLI intentionally does not expose every single RLVR setting.

Some important chapter behavior comes from the underlying `CFG()` object, which the script imports and then partially overrides.

That means some of the most important “real” Chapter 6 settings are in the config, not in `argparse`.

Examples include things like:

- evaluation sampling temperatures
- top-p settings
- the list of `k` values used for Pass@k curves and compute scaling
- maximum prompt length handling
- generation token limits
- verifier tolerance

This matters because if you are trying to reproduce a very specific figure from the book, **the CLI flags alone may not be the whole story**.

For Chapter 6 specifically, it is worth checking the config whenever you are changing:

- token length behavior
- evaluation behavior
- model defaults
- RLVR training defaults beyond the exposed overrides

---

## Requirements

From the repository root, install the package first:

```bash
pip install -e .
```

The script also expects the main Chapter 6 dependencies to be available:

```bash
pip install -U transformers datasets accelerate
```

Then run the case study from the repo root:

```bash
python case_studies/ch6_rlvr_grpo/run.py
```

If you are already inside the Chapter 6 case-study directory, then `python run.py` is the equivalent command.

---

## Default runtime behavior at a glance

A few defaults are explicit in the CLI:

- `--seed 42`
- `--outdir images`
- `--batch-prompts-eval 8`
- `--traj-subset-n 64`
- `--qualitative-k 5`

Many other defaults come from `CFG()` and can be overridden from the CLI when those override flags are provided.

The script automatically uses:

- **CUDA** if available
- otherwise **CPU**

It also supports optional:

- **gradient checkpointing**

for lower-memory training.

---

## The most important command-line arguments

This chapter has enough moving parts that it helps to group the CLI into categories.

### A. The main experiment-size flags

### `--train-n`
```bash
--train-n 1000
```
Overrides how many training examples are used from GSM8K.

Use this when you want to control the total scale of the SFT and GRPO stages.

- smaller = faster debugging
- larger = more meaningful training

### `--test-n`
```bash
--test-n 256
```
Overrides how many held-out test examples are used.

This is helpful when you want to make evaluation cheaper during quick runs.

### `--sft-steps`
```bash
--sft-steps 100
```
Controls how many supervised fine-tuning steps are run before RLVR begins.

### `--grpo-steps`
```bash
--grpo-steps 100
```
Controls how many GRPO training steps are run.

This is one of the most important benchmark-length flags in the entire script.

### `--prompts-per-step`
```bash
--prompts-per-step 4
```
Controls how many prompts participate in each GRPO update.

This interacts directly with `--group-size`.

### `--group-size`
```bash
--group-size 2
```
Controls how many sampled completions are drawn per prompt during GRPO.

This is the single most important “quality vs compute” knob in the chapter.

### `--eval-every`
```bash
--eval-every 10
```
Controls how frequently the GRPO training loop is evaluated.

This mainly affects:

- learning-curve density
- runtime overhead
- how smooth or sparse the plotted training trajectory looks

### `--eval-curve-n`
```bash
--eval-curve-n 128
```
Controls the size of the repeated evaluation subset used for learning-curve measurement.

---

### B. The most important system / memory flags

### `--batch-prompts-eval`
```bash
--batch-prompts-eval 4
```
Controls evaluation batch size.

This is usually the first flag to reduce when you hit GPU memory limits during evaluation.

### `--gradient-checkpointing`
```bash
--gradient-checkpointing
```
Enables gradient checkpointing before training.

This is often worth turning on for Chapter 6 if you are running on a modest GPU.

### `--model-name`
```bash
--model-name gpt2
```
Overrides the configured model.

This is one of the biggest practical levers when adapting the chapter to different hardware.

---

### C. The most useful analysis flags

### `--traj-subset-n`
```bash
--traj-subset-n 32
```
Controls how many held-out prompts are used when searching for qualitative early-vs-late trajectory examples.

### `--qualitative-k`
```bash
--qualitative-k 3
```
Controls how many printed examples to show.

These flags matter because qualitative printing can be helpful, but it is easy to overspend time on it when what you really need is the main benchmark curves.

### `--seed`
```bash
--seed 123
```
Sets the global seed.

This script seeds Python, NumPy, Torch, Transformers, and CUDA where available, so the seed is important when you want comparable runs.

### `--outdir`
```bash
--outdir results/ch6_trial_a
```
Controls where the plots and output tables are written.

This is especially useful for Chapter 6 because you will often want to keep multiple experiment variants side by side.

---

## Recommended run profiles

Chapter 6 benefits a lot from intentionally choosing the right run profile for your hardware.

Below are practical starting points.

### 1. Local smoke test or CPU sanity check
Use this when:

- you want to check that the code runs end to end
- you are on CPU only
- you are testing a code change
- you want a very cheap sanity pass before a real experiment

Example:

```bash
python case_studies/ch6_rlvr_grpo/run.py \
  --train-n 128 \
  --test-n 64 \
  --sft-steps 10 \
  --grpo-steps 10 \
  --eval-every 5 \
  --eval-curve-n 32 \
  --prompts-per-step 2 \
  --group-size 2 \
  --batch-prompts-eval 2 \
  --traj-subset-n 8 \
  --qualitative-k 2
```

This is not the setting to judge final chapter performance. It is the setting to confirm the pipeline works.

### 2. Local GPU run with the current local-friendly configuration
Use this when:

- you have a modest local GPU
- you want real curves without jumping straight to a benchmark-heavy run
- you want a practical balance between quality and cost

Example:

```bash
python case_studies/ch6_rlvr_grpo/run.py \
  --train-n 1000 \
  --test-n 256 \
  --sft-steps 100 \
  --grpo-steps 100 \
  --eval-every 10 \
  --eval-curve-n 128 \
  --prompts-per-step 4 \
  --group-size 2 \
  --batch-prompts-eval 4 \
  --traj-subset-n 32 \
  --qualitative-k 3 \
  --gradient-checkpointing
```

This is usually the most sensible “real local run” template.

It keeps the two biggest cost multipliers under control:

- local-friendly **group size 2**
- local-friendly **96-token cap** in the chapter config

### 3. Colab / T4-style midrange GPU run
Use this when:

- you want stronger figures than a laptop run
- you have access to a midrange cloud GPU
- you want a practical benchmark without jumping all the way to the heaviest setting

Example:

```bash
python case_studies/ch6_rlvr_grpo/run.py \
  --train-n 1500 \
  --test-n 512 \
  --sft-steps 150 \
  --grpo-steps 150 \
  --eval-every 10 \
  --eval-curve-n 256 \
  --prompts-per-step 4 \
  --group-size 2 \
  --batch-prompts-eval 4 \
  --traj-subset-n 48 \
  --qualitative-k 3 \
  --gradient-checkpointing
```

A T4-class GPU is a good middle ground for this chapter:

- much better than CPU for end-to-end RLVR work
- still small enough that you need to be careful
- a good match for the current local-friendly chapter settings

On this kind of hardware, the safest approach is usually:

- keep `group_size` at **2** first
- keep the token cap conservative
- turn on gradient checkpointing
- only increase evaluation batch size if you know you have memory headroom

### 4. Stronger benchmark-oriented GRPO run
Use this when:

- you care more about stronger chapter results than cheap iteration
- you have a roomier GPU setup
- you specifically want to push toward the stronger GRPO setting

Example:

```bash
python case_studies/ch6_rlvr_grpo/run.py \
  --train-n 2000 \
  --test-n 512 \
  --sft-steps 200 \
  --grpo-steps 200 \
  --eval-every 10 \
  --eval-curve-n 256 \
  --prompts-per-step 4 \
  --group-size 8 \
  --batch-prompts-eval 4 \
  --traj-subset-n 64 \
  --qualitative-k 5 \
  --gradient-checkpointing
```

This is much closer to the “best results matter most” mindset.

But it is also exactly where the cost grows fast. Moving from group size 2 to group size 8 is the change that most clearly turns Chapter 6 from a local experiment into a real GPU workload.

---

## CPU vs GPU guidance

### CPU-only
CPU is acceptable for:

- smoke tests
- debugging
- validating that the pipeline works
- checking saved outputs and printed examples

CPU is generally **not** the best way to judge the full Chapter 6 benchmark, because RLVR generation and repeated evaluation become slow very quickly.

If you are on CPU:

- shrink `train-n`
- shrink `test-n`
- use small `sft-steps` and `grpo-steps`
- keep `group-size 2`
- keep evaluation batch size small
- reduce qualitative analysis sizes

### Modest GPU
A modest GPU is usually enough for the current local-friendly version of the chapter if you are disciplined.

Priorities:

1. keep group size conservative
2. keep token cap conservative
3. use gradient checkpointing
4. reduce evaluation batch size first if memory gets tight
5. only then think about increasing steps or dataset size

### Stronger GPU
A stronger GPU lets you do what Chapter 6 really wants:

- larger train/test slices
- longer SFT and GRPO phases
- more evaluation coverage
- more comfortable best-of-N evaluation
- stronger GRPO group sizes such as 8

If your goal is “best chapter figures,” stronger GPUs matter more here than they did in the earlier tabular or small-control chapters.

---

## Tuning advice: what to change first

When Chapter 6 is too slow, too memory-heavy, or too weak, it helps to have a tuning order.

### If you run out of memory
Change these in roughly this order:

1. reduce `--batch-prompts-eval`
2. enable `--gradient-checkpointing`
3. reduce `--group-size`
4. reduce dataset slices with `--train-n` and `--test-n`
5. consider a smaller `--model-name`

### If training is just too slow
Change these first:

1. reduce `--grpo-steps`
2. reduce `--sft-steps`
3. reduce `--train-n`
4. reduce `--eval-curve-n`
5. increase `--eval-every` so evaluation happens less often

### If results are too weak
Improve these first:

1. increase `--grpo-steps`
2. strengthen the SFT stage with `--sft-steps`
3. increase `--train-n`
4. increase `--group-size` if hardware allows
5. only then expand qualitative analysis or cosmetic evaluation settings

### If the learning curve is too noisy to interpret
Try:

- larger `--eval-curve-n`
- a larger `--test-n`
- possibly more training steps
- better seed control and consistent hardware settings

---

## Suggested workflow for serious Chapter 6 experimentation

A good practical workflow is:

### Step 1. Run a tiny sanity check
Make sure the whole pipeline works with a cheap run.

### Step 2. Run a balanced local-GPU or T4-style experiment
Keep `group_size = 2`, keep the token cap conservative, and produce the main figures.

### Step 3. Decide whether the question is about code correctness or final performance
If the code works and the question becomes “how good can this get?”, then it is time to scale up:

- data size
- SFT steps
- GRPO steps
- eventually group size

### Step 4. Use stronger settings only after the chapter pipeline is already stable
Do not start with the heaviest run. Chapter 6 is exactly the kind of script where a cheap run can save you hours.

---

## Interpreting the main results

When reading the outputs, a few interpretations are especially useful.

### Base vs SFT
This tells you how much ordinary supervised adaptation buys you before RL enters the picture.

### SFT vs GRPO
This is the central chapter comparison.

If GRPO clearly improves Pass@k and the reasoning diagnostics remain sensible, then the RLVR stage is adding real value rather than just lengthening outputs.

### Pass@1 vs Pass@8
This comparison tells you whether the gain is mostly from a stronger single sample or from improved best-of-N sampling behavior.

### Compute scaling curve
This tells you whether extra inference compute is paying off efficiently.

### Reasoning diagnostics
These help you avoid shallow interpretations like “longer output means better reasoning.” Sometimes longer is helpful, sometimes it is just more expensive.

---

## Outputs you will typically want to keep

For chapter writing or experiment tracking, the most important artifacts are usually:

- `CaseStudy_Fig1_LearningCurves_v3.png`
- `CaseStudy_Fig2_TestTimeScaling.png`
- `CaseStudy_Fig3_ReasoningQualityEvolution.png`
- `chapter6_summary.csv`
- `chapter6_results.json`

If you are iterating on configs, it is a good idea to change:

```bash
--outdir
```

so each run gets its own folder.

---

## Practical bottom line

If you want the shortest useful summary of how to run Chapter 6 well, it is this:

- start small
- use a GPU if possible
- keep the current local-friendly token cap at **96** for local work
- keep **`group_size = 2`** for local and midrange runs
- move to **`group_size = 8`** only when you are deliberately paying for a stronger benchmark-style result
- use **gradient checkpointing** when memory is tight
- treat **group size** and **token length** as the two most important compute multipliers in the chapter

That is the core practical lesson of the current Chapter 6 script.

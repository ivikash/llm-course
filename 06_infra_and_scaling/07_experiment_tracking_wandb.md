# 6.7 - Experiment Tracking with wandb

When you run many experiments (and you will), `print` statements and text files stop scaling. You need a proper experiment tracker.

**wandb** (Weights & Biases) is the industry default. Free for individuals. This lesson takes you from zero to productive.

## Setup (5 minutes)

```bash
pip install wandb
wandb login
# opens browser, you paste an API key from https://wandb.ai/authorize
```

One-time. Credentials stored in `~/.netrc`.

## Basic usage

```python
import wandb

wandb.init(project="llm-course", name="shakespeare-run-1")
wandb.config = {
    "learning_rate": 1e-3,
    "batch_size": 64,
    "model": "gpt-small",
}

for step in range(num_steps):
    loss = train_step()
    wandb.log({"train/loss": loss, "train/lr": current_lr}, step=step)

wandb.finish()
```

That's it. Metrics now live on https://wandb.ai/your-user/llm-course. You get:
- Live loss curves
- Hyperparameter tracking
- System metrics (GPU util, memory)
- Diff view across runs

## What to log

During training:
- **`train/loss`** (every step)
- **`train/lr`** (every step)
- **`train/grad_norm`** (every step)
- **`train/tokens_per_sec`** (every step)
- **`train/mfu`** (every step, if computable)
- **`val/loss`** (every eval interval)
- **`val/bpb`** (every eval interval)
- **`gpu/memory_used`** (periodically)

For a benchmark run:
- **`eval/mmlu`**, **`eval/arc_easy`**, **`eval/gsm8k`** (after training)
- **`eval/core`**

For qualitative:
- **`samples/text`** (sample generations periodically, logged as text or tables)

## In nanoGPT

`train.py`:

```python
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# in the loop:
if wandb_log and master_process:
    wandb.log({
        "iter": iter_num,
        "train/loss": losses['train'],
        "val/loss": losses['val'],
        "lr": lr,
        "mfu": running_mfu * 100,
    })
```

Enable via config:
```python
wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2-124m'
```

## In nanochat

More elaborate logging. See `scripts/base_train.py` for the `wandb.init(...)` and `wandb.log(...)` calls. It logs:
- Training loss per step (high-frequency, using `log_freq=1`)
- Val BPB every eval
- CORE metric periodically
- GPU memory
- Sample generations (text)

You pass `WANDB_RUN=myrun` as an env var to activate it (otherwise uses a "dummy" that skips logging).

## Groups, sweeps, tags

### Groups

For related runs, set `group`:
```python
wandb.init(project="llm-course", group="scaling-sweep", name="d12")
wandb.init(project="llm-course", group="scaling-sweep", name="d24")
```

The dashboard then shows them together with aggregation.

### Sweeps

Run many experiments with varying hyperparameters:

```yaml
# sweep.yaml
program: train.py
method: grid
parameters:
  learning_rate:
    values: [1e-4, 3e-4, 1e-3, 3e-3]
  batch_size:
    values: [32, 64, 128]
```

```bash
wandb sweep sweep.yaml
wandb agent your-user/project/SWEEP_ID   # runs one job from the sweep
```

Launch N agents for N parallel jobs. The sweep controller picks next hyperparameters based on `method` (grid, random, bayes).

### Tags

```python
wandb.init(tags=["v2-architecture", "bf16", "dclm-data"])
```

Filter dashboard by tag. Useful for "show me all runs with the new optimizer."

## Comparing runs

wandb's most valuable feature is **runs table** + **compare view**:

1. On the project dashboard, select 2+ runs with checkboxes.
2. "Compare".
3. Overlaid loss curves, diffed configs, per-metric sparklines.

This is where you see "oh, lr=3e-4 converges faster than lr=1e-3 for d24 models" at a glance.

## Artifacts (save your outputs)

Beyond metrics, log:

```python
wandb.save("out/ckpt.pt")        # uploads the checkpoint
wandb.log_artifact("model.pt", name="my-model", type="model")
```

Artifact pages track lineage ("this model was produced by this run with this dataset").

Useful for team workflows. Individual use: skip initially, add once you run into "which run produced which model" confusion.

## Alternative trackers

- **tensorboard**: PyTorch's original. Local, no cloud. Fine for one-off runs. Scales poorly.
- **mlflow**: open source, self-hostable. Enterprise-friendly.
- **Neptune.ai**: like wandb, less popular.
- **ClearML**: end-to-end MLOps, heavier.
- **Aim**: open source, clean UI, local or self-hosted.

wandb is default for a reason - best free tier and best UX. Every ML team I know uses it.

## Gotchas

1. **Multi-GPU**: only log from rank 0. Else you get 8 duplicate lines of each metric.
   ```python
   if master_process:
       wandb.log(...)
   ```

2. **Forget to `wandb.finish()`**: at script exit, metrics may not flush. Add `wandb.finish()` at the end, or use `atexit`.

3. **Crash recovery**: if training crashes, the run is marked "crashed" in the UI. Fine. Start a new run to resume (wandb doesn't auto-resume).

4. **Storage limits**: free tier has 100GB. Plenty for individual use. Teams pay.

5. **Privacy**: by default, runs are private to you. `project=xyz` can be made public, or teams.

## Exercises

1. Enable wandb in nanoGPT's Shakespeare training. Run for 500 iterations. Watch the live curves in your browser.

2. Run twice with different LRs (1e-3, 3e-3). Compare in the UI.

3. Log `train/grad_norm` in addition to loss. Plot over time - healthy pattern is: big at start, ~1.0 stable throughout.

4. Try a sweep: 3 values of `learning_rate`, launch 3 agents in parallel (3 terminal tabs).

5. (If you have time) add sample generations to wandb. Use `wandb.Table` for tabular sample logging.

## Next

`08_huggingface_ecosystem.md` - the other essential platform.

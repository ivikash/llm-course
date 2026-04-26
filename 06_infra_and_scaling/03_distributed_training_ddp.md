# 6.3 - Distributed Training with DDP

One GPU is not enough for serious models. Distributed Data Parallel (DDP) is the simplest and most common way to use many.

## The idea

With N GPUs:
1. Put a **full copy** of the model on each GPU.
2. Split each batch into N sub-batches, one per GPU.
3. Each GPU does its own forward + backward, producing local gradients.
4. **All-reduce** the gradients: sum across all GPUs, divide by N, broadcast back.
5. Each GPU now has the average gradient. Step the optimizer. Weights stay in sync.

Result: effective batch is N × per-GPU batch. Training is ~N× faster (with small overhead).

Key property: DDP requires the model to **fit on one GPU**. For bigger models, see FSDP (Lesson 6.4).

## Launch: torchrun

Launch N processes, one per GPU:

```bash
torchrun --standalone --nproc_per_node=8 train.py
```

- `--nproc_per_node=8`: 8 processes (one per local GPU).
- `--standalone`: single-node, no cluster coordination.

Multi-node (across machines):

```bash
# Node 0 (master)
torchrun --nnodes=2 --node_rank=0 --master_addr=10.0.0.5 --master_port=29500 \
         --nproc_per_node=8 train.py
# Node 1
torchrun --nnodes=2 --node_rank=1 --master_addr=10.0.0.5 --master_port=29500 \
         --nproc_per_node=8 train.py
```

Each process gets environment variables:
- `RANK`: global rank (0 to world_size - 1).
- `LOCAL_RANK`: local rank within the node (0 to nproc_per_node - 1).
- `WORLD_SIZE`: total processes.

## Code side

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# 1. init
dist.init_process_group(backend='nccl')   # 'nccl' for NVIDIA
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])

# 2. device
torch.cuda.set_device(local_rank)
device = f'cuda:{local_rank}'

# 3. model wrap
model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank])

# 4. different data per rank (use rank as seed/offset)
X, Y = get_batch(rank=rank, seed=step*world_size + rank)

# 5. normal training loop
logits, loss = model(X, Y)
loss.backward()            # DDP secretly all-reduces gradients here
optimizer.step()

# 6. cleanup at end
dist.destroy_process_group()
```

The `DDP(model)` wrapper is the magic. During `backward()`, it hooks into PyTorch's autograd to all-reduce gradients automatically.

## nanoGPT's DDP setup

`train.py`:

```python
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
```

Then later:

```python
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
```

20 lines total. Same script works single-GPU (RANK not set) and multi-GPU.

## Common idioms

### "master_process" gating

You only want rank 0 to print logs, save checkpoints, and log to wandb (otherwise you get 8 copies of every message):

```python
master_process = ddp_rank == 0
if master_process:
    wandb.init(...)
    print(f"iter {step}: loss {loss.item()}")
    if step % save_interval == 0:
        torch.save(...)
```

### DDP + gradient accumulation

DDP all-reduces on every `.backward()` call. If you're accumulating gradients over K micro-steps, you don't want K all-reduces per step. Solution:

```python
for micro_step in range(K):
    model.require_backward_grad_sync = (micro_step == K - 1)    # only sync on last
    logits, loss = model(X, Y)
    (loss / K).backward()
optimizer.step()
```

This is what nanoGPT does.

## Under the hood: NCCL

NVIDIA's **NCCL** library implements collective operations (all-reduce, broadcast, etc.) optimized for GPU interconnects (NVLink within a node, Infiniband or RoCE across nodes).

If you don't have Infiniband and are doing multi-node training, set `NCCL_IB_DISABLE=1` to fall back to TCP. Will work, much slower.

## Gotchas

1. **All processes must reach all-reduces**: if one rank diverges in control flow (`if rank == 0: do_something_that_triggers_backward()`), others wait forever.

2. **Checkpoint only on rank 0**: otherwise 8 processes try to write the same file simultaneously.

3. **Different seeds for data**: each rank should see different data. Use `rank` as a seed offset.

4. **LR scaling with world size**: a naive interpretation says "keep per-GPU batch the same, multiply LR by world_size." Works up to a point; past critical-batch-size, diminishing returns.

5. **Cleanup**: call `dist.destroy_process_group()` at the end, or you'll get zombie processes.

## nanochat's DDP usage

Same pattern. Look at `scripts/base_train.py` around the init section. It has a bit more infrastructure (per-param-group learning rate, fault tolerance hooks) but the DDP flow is identical to nanoGPT's.

## Exercises

1. If you have 2+ GPUs, launch nanoGPT's Shakespeare training with `torchrun --nproc_per_node=2 train.py config/train_shakespeare_char.py`. Note wall time.

2. If single-GPU only, just study nanoGPT/train.py's DDP setup code. Identify the ~30 lines that enable multi-GPU.

3. Add a print statement that runs on all ranks: `print(f"I am rank {ddp_rank}")`. Run with nproc_per_node=4. You'll see 4 prints. Now gate with `if master_process:`. One print.

4. Compute: on 8 GPUs, if my local batch size is 16 and I use grad_accum=4, what's the effective batch? (16 × 4 × 8 = 512.)

## Next

`04_fsdp_zero_tensor_parallelism.md` - when DDP isn't enough.

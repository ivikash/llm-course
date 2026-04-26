# PyTorch Cheatsheet

## Tensor creation

```python
torch.tensor([1, 2, 3])              # from Python list
torch.zeros(3, 4)                    # zeros
torch.ones(3, 4)                     # ones
torch.randn(3, 4)                    # random normal
torch.rand(3, 4)                     # random uniform [0,1)
torch.arange(10)                     # 0..9
torch.linspace(0, 1, 11)             # 0.0, 0.1, ..., 1.0
torch.full((3, 4), 7.0)              # filled with 7
torch.eye(3)                         # identity

# specify dtype and device
torch.zeros(3, 4, dtype=torch.float32, device='cuda')
```

## Shape operations

```python
x.shape                              # torch.Size([3, 4])
x.ndim                               # 2
x.numel()                            # 12
x.view(4, 3)                         # reshape, shares memory
x.reshape(4, 3)                      # reshape, may copy
x.T                                  # transpose (2-D only)
x.transpose(0, 1)                    # general transpose
x.permute(1, 0)                      # general permute
x.squeeze()                          # remove size-1 dims
x.unsqueeze(0)                       # add size-1 dim at position 0
x.flatten(1)                         # flatten from dim 1 onward
x.expand(3, 4, 5)                    # broadcast without copying
```

## Math

```python
x + y; x - y; x * y; x / y           # element-wise
x @ y                                # matmul
torch.matmul(x, y)                   # same
x.sum()                              # sum all
x.sum(dim=-1)                        # sum along last dim
x.mean(dim=0, keepdim=True)          # mean along dim 0
x.max(); x.min()                     # over all
x.argmax(dim=-1)                     # index of max
x.topk(5, dim=-1)                    # top 5 values and indices
torch.stack([x, y], dim=0)           # combine along new dim
torch.cat([x, y], dim=0)             # concat along existing dim
```

## Indexing

```python
x[0]                                 # first row
x[:, 0]                              # first column
x[0, :3]                             # first three of first row
x[x > 0]                             # boolean indexing
x.masked_fill(mask, value)           # replace where mask True
x[idx]                               # gather with integer indices
x.gather(dim, index)                 # advanced gather
```

## Autograd

```python
x = torch.tensor(1.0, requires_grad=True)
y = x ** 2 + 3
y.backward()
print(x.grad)   # 2x = 2

# stop gradient
with torch.no_grad():
    ...

x.detach()                           # returns tensor without grad tracking
x.requires_grad_(True)               # in-place toggle

# zero out all param grads
optimizer.zero_grad()
# or
for p in model.parameters():
    p.grad = None
```

## Neural network building blocks

```python
import torch.nn as nn
import torch.nn.functional as F

# layers
nn.Linear(in, out, bias=True)
nn.Embedding(num_embeddings, embedding_dim)
nn.LayerNorm(normalized_shape)
nn.Dropout(p=0.1)
nn.ReLU() / nn.GELU() / nn.SiLU() / nn.Tanh() / nn.Sigmoid()

# functional versions (no parameters)
F.relu(x); F.gelu(x); F.silu(x)
F.softmax(x, dim=-1)
F.log_softmax(x, dim=-1)
F.cross_entropy(logits, targets)
F.mse_loss(pred, target)
F.scaled_dot_product_attention(q, k, v, is_causal=True)

# model definition
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)

model = MyModel()
print(model)
for name, p in model.named_parameters():
    print(name, p.shape)

# switch training / eval mode
model.train()  # enables dropout, batch norm
model.eval()   # disables them
```

## Device management

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = x.to(device)
model = model.to(device)

torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.memory_allocated() / 1024**3   # GB used
```

## Optimizers

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.95),
)

# typical training step
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

## Saving / loading

```python
# save
torch.save(model.state_dict(), 'model.pt')
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'step': step,
}, 'ckpt.pt')

# load
model.load_state_dict(torch.load('model.pt'))
ckpt = torch.load('ckpt.pt')
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optimizer'])
```

## Mixed precision

```python
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
scaler = torch.cuda.amp.GradScaler()  # for fp16, not needed for bf16

with ctx:
    logits = model(x)
    loss = F.cross_entropy(logits, y)

scaler.scale(loss).backward()     # if using fp16
# OR
loss.backward()                   # if using bf16

scaler.step(optimizer); scaler.update()  # fp16
# OR
optimizer.step()                  # bf16
```

## Distributed (DDP)

```python
# launch:  torchrun --nproc_per_node=8 train.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
device = f'cuda:{rank}'
model = model.to(device)
model = DDP(model, device_ids=[rank])

# training loop is the same, model handles grad sync
```

## Common pitfalls

- Forgetting `optimizer.zero_grad()` → gradients accumulate across steps.
- Forgetting `model.eval()` before inference → dropout still active.
- `torch.no_grad()` context → huge memory savings during eval.
- `.item()` to extract a Python scalar for printing.
- `.detach()` to break gradient flow.
- `contiguous()` before `view()` after `transpose()`.
- Shape errors on matmul → check inner dims.
- NaN in loss → check lr (too high?), check data (bad token ids?), check mixed-precision ordering.

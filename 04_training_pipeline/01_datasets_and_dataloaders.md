# 4.1 - Datasets and DataLoaders

Training an LLM means feeding it hundreds of billions of tokens. How does that data actually get onto the GPU? This lesson covers the full journey from "a folder of text files on the internet" to "a batch of token IDs on your GPU."

## The scale problem

- GPT-3 was trained on ~300B tokens.
- Llama-2 was trained on 2T tokens.
- Llama-3 was trained on 15T tokens.
- Average English text: ~1 token per 4 bytes.
- So Llama-3's training data is ~60 TB of raw text.

You can't fit that in RAM. You can't even fit it on a single hard drive comfortably. Data infrastructure becomes a real engineering problem at this scale.

## Common public datasets

- **Common Crawl** - raw scraped internet, petabytes. Dirty. Need filtering.
- **The Pile** (2020) - 825GB, 22 diverse sources (books, arxiv, github, etc.).
- **RedPajama** (2023) - 1.2T tokens, open reproduction of Llama-1's dataset.
- **SlimPajama** (2023) - RedPajama deduplicated.
- **RefinedWeb** / **Falcon data** - quality-filtered Common Crawl.
- **FineWeb** (HuggingFace, 2024) - 15T tokens, carefully filtered Common Crawl. State-of-the-art open dataset.
- **DCLM** (Apple, 2024) - 240B high-quality tokens. What nanochat uses.

For this course: nanoGPT uses OpenWebText (40GB), nanochat uses DCLM-derived FineWeb.

## The pipeline: raw text → tokenized binary files

Open `~/workspace/nanoGPT/data/openwebtext/prepare.py`. ~50 lines. It:

1. Downloads OpenWebText from HuggingFace (~10GB compressed, 40GB uncompressed).
2. Tokenizes every document with tiktoken GPT-2 BPE.
3. Splits train/val 99/1%.
4. Writes two giant `.bin` files of uint16 token IDs (`train.bin`, `val.bin`).

That's the entire data prep. After this, training code doesn't know or care the tokens came from text. It just sees a big array of integers.

```python
# Simplified from prepare.py:
enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)   # end-of-text delimiter between docs
    return {'ids': ids, 'len': len(ids)}

# Tokenize entire dataset in parallel
tokenized = dataset.map(process, num_proc=8)

# Concatenate into one big file per split
for split, dset in tokenized.items():
    filename = f"{split}.bin"
    arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(total_tokens,))
    # ... write all tokens into arr ...
```

**Why uint16?** GPT-2 vocab is 50257, which fits in 16 bits (65536 max). Saves 2x disk space vs uint32.

## Reading during training: memory-mapped files

In `nanoGPT/train.py`, `get_batch`:

```python
def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)
```

**memmap** (memory-map) is the key trick. The OS makes a huge file appear as if it were in memory, but only loads pages when you touch them. You never load the whole 40GB. You just index into `data[i:i+block_size]`.

For a batch of 12 sequences of length 1024: you touch 12*1024 = ~12KB of data per step. OS caches recent pages. It's efficient.

## x vs y: next-token prediction

Notice `y = data[i+1 : i+1 + block_size]`. It's `x` shifted by one token. At position `t`, the target is the next token. This sets up next-token prediction as the training objective.

A single batch row:
```
x: [the, cat, sat, on, the, mat, .]
y: [cat, sat, on, the, mat, ., <eot>]
```
At every position, predict the next token. Loss is averaged over all positions.

## Shuffling and epochs

nanoGPT **doesn't iterate epochs**. It picks random positions from the giant file forever. With 9B tokens in OpenWebText and 12 sequences × 1024 tokens per step × 600K steps = 7.4B tokens seen, you get less than one full epoch. For LLM pretraining this is standard - you rarely want to repeat data.

## Distributed data loading

When you run `torchrun --nproc_per_node=8`, you have 8 processes. They each sample independently from `train.bin` (different random seeds from `ddp_rank`), so they see different data. Their gradients average (all-reduce). You get an effective 8× larger batch for free.

This works because the file is shared on disk and each process just memmaps it independently.

## nanochat's data loading

nanochat splits data into shards (pre-tokenized files of ~250M tokens each). See `~/workspace/nanochat/nanochat/dataset.py` and `dataloader.py`.

Key differences from nanoGPT:
- Data is **streamed**: download shards as needed, don't require all on disk.
- **Sharded across ranks**: each rank reads its own shards, no contention.
- **Packed sequences**: multiple documents per sequence, separated by `<|endoftext|>`, to avoid wasting tokens on padding.

Open `~/workspace/nanochat/nanochat/dataset.py` and skim. The key function downloads shards from a public S3 bucket on demand. This is how you handle a 150-shard dataset without filling your disk.

## PyTorch's DataLoader (when you'd use it)

For smaller-scale or non-LLM tasks you'd use `torch.utils.data.DataLoader`:

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __len__(self): return 10000
    def __getitem__(self, idx): return torch.randn(10), torch.tensor(idx % 2)

loader = DataLoader(MyDataset(), batch_size=32, shuffle=True, num_workers=4)
for x, y in loader:
    ...
```

HuggingFace's `datasets` library produces objects compatible with this. Fine for fine-tuning on smaller datasets. For pretraining at scale, the memmap-and-index approach (nanoGPT) or shard-streaming (nanochat) wins on simplicity.

## Exercises

1. Run `~/workspace/nanoGPT/data/shakespeare_char/prepare.py` (you already have the output). Open `train.bin` with numpy:
   ```python
   import numpy as np
   data = np.memmap('train.bin', dtype=np.uint16, mode='r')
   print(len(data), data[:30])
   ```
   You'll see integer token IDs. Decode a few using the `meta.pkl` mapping.

2. Read `~/workspace/nanoGPT/data/openwebtext/prepare.py` top to bottom. 50 lines. Understand the entire dataset prep.

3. Compare `~/workspace/nanochat/nanochat/dataset.py` and `dataloader.py` to nanoGPT's approach. Note: similar philosophy (big file of token ids, random indexing), more sophisticated infrastructure (shards, download, distribution).

4. Think: if you want to train on your own data (say, a folder of your journal entries), what's the minimum you need? (Answer: a `.bin` file of token ids. Write a small script modeled on Shakespeare's prepare.py.)

## Next

`02_bpe_tokenizer_deep.md` - how BPE actually works, and training your own.

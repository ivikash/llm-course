# 6.6 - The Cloud: Lambda, Runpod, AWS, SageMaker

At some point you stop fitting on your laptop and rent someone else's GPUs. Here's how.

## The two tiers

### Hobbyist / research providers

Best for: individual experiments, short runs, cheap prototyping.

| Provider | Notable | Cost |
|----------|---------|------|
| **Lambda Labs** | Clean UX, Karpathy's recommendation, H100 available | $2-3/hr per H100 |
| **Runpod** | Spot instances, cheap, community cloud | $1-3/hr |
| **Vast.ai** | Cheapest, peer-to-peer, variable quality | $0.50-2/hr |
| **Paperspace (now DigitalOcean)** | Notebooks + machines | $1-3/hr |
| **Modal** | Serverless containers, autoscaling | Per-second billing |

Typical flow:
1. Sign up, add payment.
2. Pick an instance type (e.g. 8xH100).
3. Click "launch".
4. Get an SSH command.
5. SSH in, git clone your repo, run training.
6. **Stop the instance when done.** (The billing doesn't stop automatically.)

### Enterprise / hyperscaler providers

Best for: production, compliance, scale, integration with other services.

| Provider | Flagship | Notes |
|----------|----------|-------|
| **AWS** | EC2 p4/p5 instances, Trainium | Most services, most complex |
| **GCP** | A3/A4 instances, TPUs | Strong on TPUs and data analytics |
| **Azure** | ND-series | Deep OpenAI partnership |
| **Oracle OCI** | Competitive GPU availability | Often cheapest enterprise |
| **CoreWeave, Lambda Cloud (new)** | GPU-first, faster provisioning | Ex-crypto hardware |

These are 1.5-3x more expensive than hobbyist providers but come with SLAs, networking, IAM, multi-AZ, etc.

## Cost rules of thumb (as of 2026)

- Single H100 hobbyist: ~$2-3/hr
- 8xH100 node hobbyist: ~$20-25/hr
- Same on AWS/GCP on-demand: ~$30-50/hr
- **Spot instances**: 50-70% off, can be reclaimed by provider with ~2 min notice

nanochat speedrun: ~3 hours × $25/hr = **~$75 on Lambda**. Or ~$15 on spot.

## SageMaker, specifically

SageMaker is Amazon's umbrella for ML services. It's three loosely-related things:

1. **Notebooks**: hosted Jupyter, with GPUs optional.
2. **Training Jobs**: you submit a Docker image + training script, SageMaker provisions instances, runs, saves output to S3, shuts down. You pay for just the training time.
3. **Endpoints**: hosted model serving with autoscaling.
4. **Pipelines, Experiments, Model Registry, etc.**: orchestration.

### When SageMaker is useful

- You're already deep in AWS. Data in S3, IAM, VPCs.
- You need compliance, audit trails.
- You want automatic spot-instance management.
- You want a model registry across a company.

### When it's overkill

- Personal projects. The abstractions get in the way.
- Research iterations. Too much friction vs SSH.
- Quick experiments. Lambda is faster and cheaper.

### A minimal SageMaker training job

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='./nanoGPT',
    role='arn:aws:iam::...:role/SageMakerRole',
    instance_count=1,
    instance_type='ml.p4d.24xlarge',   # 8xA100
    framework_version='2.0',
    py_version='py310',
    hyperparameters={'batch_size': 12, 'block_size': 1024, ...},
    output_path='s3://my-bucket/models/',
)
estimator.fit()
```

Behind the scenes: SageMaker builds a container, pushes to ECR, provisions a p4d instance, runs your training, uploads outputs to S3, terminates the instance. You pay per-second of runtime.

## Practical workflow (what I'd recommend)

For this course:

1. **First cloud experiment**: Lambda Labs. Sign up, launch a 1xA100, SSH in, run Shakespeare training in 30 seconds. Cost: $0.50.

2. **Module 5 capstone**: Lambda 8xH100 for 3 hours. ~$75. Run the nanochat speedrun.

3. **If you work at a company with AWS**: learn the SageMaker basics for practical reasons, but do personal learning elsewhere.

## Setup tips

### SSH key management

Each provider generates or uploads SSH keys. Put them in `~/.ssh/` with `chmod 600`. Use `~/.ssh/config`:

```
Host lambda-gpu
    HostName 123.456.789.0
    User ubuntu
    IdentityFile ~/.ssh/lambda_id
```

Then `ssh lambda-gpu` is enough.

### tmux / screen

If your SSH drops, your training dies. Run in a persistent terminal:

```bash
tmux new -s train
# inside tmux: run your training
# Ctrl-B, D: detach
# reconnect later: tmux attach -t train
```

nanochat's speedrun.sh even suggests `screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh`.

### Data transfer

Your home machine has slow upload. Use:
- `rsync -avzP local_dir/ user@host:remote_dir/`
- `scp file.txt user@host:~/`
- `aws s3 cp` if using S3 buckets.
- Prefer to download data inside the GPU instance (much faster) rather than uploading.

### Cost monitoring

Set billing alerts. Every provider has them. I've seen engineers rack up $5,000 bills from forgotten GPU instances. Set a budget alert at $50 or $100.

## Spot / preemptible caveats

Spot instances can be reclaimed. Your training must be:
- **Checkpointable**: save often.
- **Resumable**: handle graceful shutdown signals.

nanochat checkpoints every N steps. So if killed, you restart and lose ~1 step of progress. For long runs, this is worth the 50-70% discount.

## The "I forgot to stop it" problem

The #1 cloud mistake: leaving instances running. You get the SSH terminal, your SSH drops, you forget about it, 48 hours later you've burned $1,000.

Countermeasures:
- **Always check with a UI or CLI** after a session: `lambda-cli instance list` or equivalent.
- **Set timeouts**: on Lambda, you can set auto-termination.
- **Calendar reminder** to check daily.

## Visualize this

**Provider price comparison (2026)**:

```
  8× H100 80GB for ~1 hour:

  Provider              On-demand    Spot/Community
  ────────────────────  ───────────  ──────────────
  Lambda Labs            $22/hr       $12/hr
  Runpod                 $18/hr       $10/hr
  Vast.ai (peer-to-peer) $14/hr       $8/hr
  Modal (serverless)     $25/hr       n/a (per-second)
  CoreWeave              $25/hr       $15/hr
  AWS (p5.48xlarge)      $98/hr       $45/hr   (spot)
  GCP (a3-highgpu-8g)    $88/hr       $40/hr
  Azure (ND96is h100)    $90/hr       -

  Typical cost for nanochat speedrun (3.5 hours):
    Lambda on-demand:   $77
    Lambda spot:         $42
    Runpod spot:         $35
    Vast.ai cheapest:    $28
    AWS on-demand:       $343   ← 5× more expensive
```

For personal learning: hobbyist providers (Lambda, Runpod, Vast) win by a huge margin. Enterprises use AWS/GCP for compliance and SLAs.

**SageMaker: what you're actually paying for**:

```
  SageMaker Training Job price = EC2 price + "SageMaker premium" (~25-30%)

  Example: p4d.24xlarge (8× A100 40GB)
    EC2 on-demand:           $32.77/hr
    SageMaker training:      $40.96/hr  (~25% markup)

  You get in return:
  ✓ Managed container (no OS setup)
  ✓ Automatic data loading from S3
  ✓ Instance auto-termination when training ends
  ✓ Spot instance management
  ✓ Hyperparameter tuning service
  ✓ Model registry
  ✓ Metric logging (built-in wandb-lite)
  ✓ IAM, VPC, compliance

  For a solo researcher: overkill. Use Lambda.
  For a regulated enterprise: probably worth the markup.
```

**Cloud workflow (the typical session)**:

```
  local machine                              cloud instance
  ──────────────                              ────────────────
                                              
  browser → "launch 8xH100 at Lambda"
                                              instance boots (~2 min)
                                              
  get SSH command
  ssh -L 8000:localhost:8000 ubuntu@ip   ───▶  terminal session
                                              
                                              $ tmux new -s train
                                              $ git clone nanochat
                                              $ bash runs/speedrun.sh
                                              [training starts, ~3.5h]
                                              
  [you go live your life, check occasionally]
                                              
  browser → wandb.ai/my-run        ───▶   live dashboard
                                              
  [training done]                              
                                              $ python -m scripts.chat_web
                                              [starts server on :8000]
  
  browser → localhost:8000         ───▶   chat UI
                                              (port forwarded via SSH)
  
  scp model.pt local:             ◀───▶   model downloaded
                                              
  browser → Lambda dashboard → Terminate  ⚠ MUST do this
                                              instance stops billing
```

**Survival checklist for any cloud session**:

```
  before starting:
    [ ] spending alert set at $100-200
    [ ] SSH key ready locally
    [ ] project folder prepared to upload
    [ ] wandb API key in env vars
    [ ] planned session time (so you don't start at night and forget)

  during:
    [ ] tmux (so SSH drop doesn't kill training)
    [ ] monitor wandb dashboard
    [ ] save checkpoints often

  at end:
    [ ] scp artifacts you want
    [ ] verify "downloaded correctly" (md5sum)
    [ ] TERMINATE INSTANCE (don't just "stop" - some providers still bill)
    [ ] verify the billing page shows $0/hr
    [ ] screenshot the run for proof
```

**What the SageMaker script looks like (for awareness)**:

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    source_dir='./my_project',
    role='arn:aws:iam::12345:role/SageMakerRole',
    instance_count=1,
    instance_type='ml.p4d.24xlarge',  # 8× A100 40GB
    framework_version='2.1',
    py_version='py310',
    hyperparameters={'epochs': 3, 'batch_size': 32},
    output_path='s3://my-bucket/models/',
    use_spot_instances=True,   # ~70% cheaper
    max_wait=14400,
    max_run=10800,
)

estimator.fit({'train': 's3://my-bucket/train-data/'})
# Monitors via CloudWatch. Instance auto-terminates when done.
```

Elegant when it works. Fiddly when it doesn't.

## Exercises

1. Sign up on Lambda Labs (free), don't launch anything yet. Browse the instance types and prices.

2. If comfortable, launch a 1xA100 for 15 minutes. SSH in, clone nanoGPT, run Shakespeare training. Terminate. Cost: <$1.

3. Estimate for yourself: if you had a $200 cloud budget, what's the biggest training run you could do? (8xH100 for 8 hours, or 1xH100 for 66 hours, or many small runs.)

4. Read SageMaker's "Getting Started" tutorial. Don't actually run it, just understand the abstractions.

## Next

`07_experiment_tracking_wandb.md` - how to make sense of many runs.

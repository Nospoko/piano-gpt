# Training Pipeline Documentation
Remember to add your .env file with
```
WANDB_API_KEY=$YOUR_API_KEY
HF_TOKEN=$YOUR_AUTH_TOKEN
HF_WRITE_TOKEN=$OPTIONAL_WRITE_TOKEN
```
Before training you can also run
```
ulimit -n 4096
```
Especially for DDP. This will enable the program to run more processes than allowed by default.

## Overview

The training pipeline consists of two main stages:
1. Next-token pretraining
2. Piano task fine-tuning

The system supports both single-GPU and distributed training using DistributedDataParallel (DDP).

## Training

### Loop

The training loop implements:
- Gradient accumulation
- Learning rate scheduling
- Gradient scaling for mixed precision
- Periodic evaluation on several validation splits
- Checkpointing
- Logging to wandb

Key configuration parameters:
```yaml
optimizer:
  gradient_accumulation_steps: 16  # Accumulate gradients across batches
  max_iters: 300000
  weight_decay: 0.1
  grad_clip: 1.0

lr:
  learning_rate: 8e-5  # Base learning rate
  min_lr: 8e-6        # Minimum LR after decay
  warmup_iters: 1000  # Linear warmup steps
  decay_lr: true      # Enable cosine decay. If False the learning rate will remain the same as base learning rate

system:
  dtype: float16      # Or float32 or bfloat16
  compile: true       # Use torch.compile()
  data_workers: 64    # Number of data loading workers
```

### Training Commands

Basic single-GPU training:
```bash
python -m gpt2.train --config-name=debugging_pretraining
```

Multi-GPU training with DDP:
```bash
# 4 GPUs on single node
torchrun --nproc-per-node=4 gpt2/train --config-name=debugging-pretraining
```

### Training Stages

#### 1. Next-Token Pretraining

Uses `stage: next_token_pretraining` configuration. The model learns to predict the next token in MIDI sequences.

```bash
PYTHONPATH=. torchrun --nproc-per-node=4 \
gpt2/train.py --config-name=gpt2_pretraining \
data.batch_size=32 \
optimizer.gradient_accumulation_steps=8 \
data.context_size=2048 \
init_from=scratch_next_token
```

#### 2. Piano Task Fine-tuning

Uses `stage: piano_task` configuration. The model learns specific piano playing tasks like dynamics prediction.

```bash
PYTHONPATH=. torchrun --nproc-per-node=4 \
gpt2/train.py --config-name=gpt2_piano_finetuning \
data.batch_size=32 \
data.notes_per_record=128 \
init_from=midi-gpt2-pretrained-model.pt
```

### Checkpointing

The training loop saves checkpoints:
- Best validation loss checkpoint: `{run_name}.pt`
- Latest checkpoint: `{run_name}last.pt`


In the folder specified as `out_dir` in the config (default is "tmp/checkpoints" and it is harcoded in automatic upload and download scripts)


Checkpoints contain:
```
"model": Model state dict
"optimizer": Optimizer state dict
"model_cfg": Model config (cfg.model)
"iter_num": Current iteration count
"best_val_loss": Best validation loss
"val_loss": Current validation loss
"train_loss": Current training loss
"run_config": Training config
"wandb": Link to wandb run
"wandb_id": Wandb run id
"total_tokens": Nuber of tokens that passed through the model
"tokenizer_desc": tokenizer.to_dict()
```
And if the stage was `piano_task`, also:
```
"piano_tasks_config": Configuration for piano task (default for now)
```

### Evaluation During Training

Evaluation occurs every `eval_interval` steps:
1. Switches model to eval mode
2. Performs inference on validation splits:
   - Full validation set
   - Bach-only compositions
   - Chopin-only compositions
   - Mozart-only compositions
3. Computes loss metrics
4. Logs to wandb if enabled

The evaluation is performed on `eval_steps` batches.

### Classes and Methods used for training

```python
class LRScheduler:
    """Handles learning rate scheduling with warmup and cosine decay"""
    def get_lr(self, it: int) -> float:
        # Linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # Cosine decay to min_lr
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

class CyclicalDataLoader:
    """Provides infinite data iteration for training"""
    def get_batch(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return self._to_device(batch)
```

### Distributed Training Details

The training script automatically configures DDP when launched with torchrun:

1. Process Group Initialization:
```python
def setup_device(cfg: DictConfig):
    if int(os.environ.get("RANK", -1)) != -1:
        init_process_group(backend=cfg.ddp.backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}", True
    return cfg.system.device, False
```

2. Gradient Synchronization:
```python
# Only sync gradients on last micro-step of gradient accumulation
model.require_backward_grad_sync = micro_step == cfg.optimizer.gradient_accumulation_steps - 1
```

3. Batch Size Scaling:
- Global batch size = batch_size * gradient_accumulation_steps * world_size
- Gradient accumulation steps are automatically scaled down by world_size.

This means that all GPU nodes cumulatively always perform `gradient_accumulation_steps` forward passes before a backward pass.

#### ⚠️ Gradient scaling currently overrides the run config
Currently the gradients are scaled down using
```py
cfg.optimizer.gradient_accumulation_steps //= ddp_world_size
```
It would be better to not override the config.

### Resource Management

For optimal performance on multi-GPU setups:

1. Memory Management:
- Use gradient accumulation to fit larger effective batch sizes.

2. CPU Resources:
- Data workers scaled based on number of GPUs
- Reserve some CPU cores for GPU communication

### Common Issues

1. Out of Memory:
- Reduce batch size
- Increase gradient accumulation steps

2. Slow Training:
- Increase number of data workers
- Enable torch.compile()
- Use NCCL backend for multi-GPU

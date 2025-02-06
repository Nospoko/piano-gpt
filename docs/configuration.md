# Configuration Keys and Their Usage

## 1. Stage and Initialization Configuration

Controls training stage and model initialization:

```yaml
stage: "piano_task"
init_from: "scratch_next_token"
loss_masking: "pretraining"

tasks: TODO
```

Important parameters:
- `stage`: Training stage ("piano_task" or "next_token_pretraining")
- `init_from`: Initialization mode ("scratch_next_token" or path to checkpoint for finetuning)
- `loss_masking`: Loss calculation mode ("pretraining" or "finetuning"). Finetuning makes the loss be calculated only on target tokens (those after the `<GENAI>` token in the sequence).
- `tasks`: NOT IMPLEMENTED

Implementation:

Model initialization:
```python
if cfg.init_from == "scratch_next_token":
    model = GPT(config=model_cfg, vocab_size=tokenizer.vocab_size)
else:
    checkpoint = torch.load(cfg.init_from)
    model.load_state_dict(checkpoint["model"])
```

## 2. Model Configuration

All model architecture parameters are used during model initialization:

```yaml
# gpt2/configs/model/gpt2.yaml
model:
  n_layer: 12
  n_head: 12
  n_embd: 768
  dropout: 0.0
  bias: false
  context_size: 2048
```

Important parameters:
- `n_layer`: Number of transformer layers
- `n_head`: Number of attention heads
- `n_embd`: Embedding dimension
- `dropout`: Dropout rate (0.0 for pretraining, c an be higher for finetuning)
- `bias`: Whether to use bias in Linear and LayerNorm layers
- `context_size`: Maximum sequence length

Implementation:
```python
# gpt2/model.py
class GPT(nn.Module):
    def __init__(self, config: DictConfig, vocab_size: int, pad_token_id: int = 0):
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, config.n_embd),
            wpe=nn.Embedding(config.context_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
```

Note: The config argument in this project is `cfg.model` or `checkpoint["model_cfg"]`

## 3. Training System Configuration

System-wide settings affecting hardware utilization:

```yaml
# gpt2/configs/*.yaml
system:
  device: 'cuda'
  dtype: 'float16'
  compile: true
  data_workers: 64

ddp:
  backend: 'nccl'
```

Important parameters:
- `device`: Device to run on ('cuda', 'cuda:0', 'cpu', 'mps')
- `dtype`: Precision ('float16', 'float32', 'bfloat16')
- `compile`: Whether to use torch.compile(), available only for newest GPUs (those on vertex.ai are good)
- `data_workers`: Number of data loading workers
- `backend`: DDP backend - 'nccl' recommended for GPU

Implementation:

Device setup:
```python
def setup_device(cfg: DictConfig):
    if int(os.environ.get("RANK", -1)) != -1:
        init_process_group(backend=cfg.ddp.backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        return f"cuda:{local_rank}", True
    return cfg.system.device, False
```

Mixed precision training setup:
```python
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[cfg.system.dtype]

ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if cfg.system.compile:
    model = torch.compile(model)
```

## 4. Data Processing Configuration

Controls batch sizes and data chunking:

```yaml
data:
  batch_size: 32
  context_size: 2048
  notes_per_record: 128

dataset:
  base_dataset_name: "epr-labs/maestro-sustain-v2"
  extra_datasets: ["..."]
  pause_detection_threshold: 4
  augmentation:
    speed_change_factors: []
    max_pitch_shift: 0
```

Important parameters:
- `batch_size`: Size of each training batch
- `context_size`: Maximum sequence length (also the size of model input)
- `notes_per_record`: Number of notes in each training example (PianoDataset only, not used in NextTokenDataset)
- `base_dataset_name`: Dataset used for validation - only "validation" and "test" splits
- `extra_datasets`: Datasets for training - only "train" splits used
- `pause_detection_threshold`: Threshold for splitting sequences
- `augmentation`: Currently using pre-augmented datasets (epr_labs/[dataset]_augmented), so we do not need to use augmentation right before training

Implementation:

Dataset initialization:
```python
class PianoDataset:
    def __init__(self, dataset: HuggingFaceDataset, context_size: int,
                 notes_per_record: int, ...):
        self.context_size = context_size
        self.notes_per_record = notes_per_record
```

DataLoader setup:
```python
train_loader = CyclicalDataLoader(
    train_dataset,
    batch_size=cfg.data.batch_size,
    num_workers=cfg.system.data_workers // ddp_world_size,
)
```

## 5. Optimization Configuration

Controls the training optimization process:

```yaml
optimizer:
  gradient_accumulation_steps: 16
  max_iters: 300000
  weight_decay: 0.1
  grad_clip: 1.0
  beta1: 0.9
  beta2: 0.95

lr:
  learning_rate: 8e-5
  min_lr: 8e-6
  warmup_iters: 1000
  decay_lr: true
  lr_decay_iters: ${optimizer.max_iters}
```

Important parameters:
- `gradient_accumulation_steps`: Number of forward passes before update
- `max_iters`: Total training iterations
- `weight_decay`: L2 regularization
- `grad_clip`: Gradient clipping value
- `beta1`, `beta2`: Adam optimizer parameters
- `learning_rate`: Initial learning rate
- `min_lr`: Minimum learning rate after decay
- `warmup_iters`: Linear warmup steps
- `decay_lr`: Whether to use cosine decay
- `lr_decay_iters`: Steps over which to decay

Implementation:

Optimizer setup:
```python
optimizer = model.configure_optimizers(
    weight_decay=cfg.optimizer.weight_decay,
    learning_rate=cfg.lr.learning_rate,
    betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
    device_type=device_type,
)
```

Gradient clipping:
```python
if cfg.optimizer.grad_clip != 0.0:
    grad_scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
```

Learning rate scheduling:
```python
lr_scheduler = LRScheduler(scheduler_config=cfg.lr)
if cfg.lr.decay_lr:
    lr = lr_scheduler.get_lr(it=iter_num)
```

## 6. Evaluation and Logging Configuration

Controls evaluation frequency and logging:

```yaml
eval_interval: 500
eval_iters: 100
eval_only: false

logging:
  wandb_log: True
  wandb_project: 'piano-gpt'
  log_interval: 20
```

Important parameters:
- `eval_interval`: Steps between evaluations
- `eval_iters`: Number of batches per evaluation
- `eval_only`: Run only evaluation - NOT IMPLEMENTED
- `wandb_log`: Enable wandb logging
- `wandb_project`: Wandb project name
- `log_interval`: Steps between logging

Implementation:

Evaluation:
```python
if iter_num % cfg.eval_interval == 1 and master_process:
    losses = estimate_loss()
    print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['full_val']:.4f}")
```

Logging:
```python
if cfg.logging.wandb_log and master_process:
    wandb.init(
        project=cfg.logging.wandb_project,
        name=run_name,
        config=run_config,
    )
```

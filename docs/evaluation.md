
# External Evaluation

## One-off Model Evaluation

To evaluate a saved model checkpoint, use `high_level_piano_eval.py`:

```bash
python -m gpt2.high_level_piano_eval init_from=tmp/checkpoints/your-model.pt
```

Or, even better:
```
PYTHONPATH=. streamlit run gpt2/high_level_piano_eval.py init_from=tmp/checkpoints/your-model.pt
```

This will:
1. Load model and tokenizer from checkpoint
2. Evaluate on validation splits (full validation set, Bach, Chopin, Mozart)
3. Generate visualizations using Streamlit (one per split)
4. Log metrics to wandb if enabled

Key configuration options:
```yaml
eval_iters: 50               # Number of evaluation iterations per split
temperature: 1               # Generation temperature
prompt_masking: false        # Loss mask type
eval_splits: ["full_val"]    # Splits to use in evaluation, multiple choice from bach, mozart, chopin, full_val
```

## Automated Evaluation Pipeline

Scripts for continuous model evaluation.

### Components

1. Model Upload (`upload_periodically.sh`):
- Uploads checkpoints from `tmp/checkpoints/` to HuggingFace Hub
- Runs every 30 minutes
- Uses `scripts/upload_models.py`

2. Model Download & Evaluation (`download_and_calculate_periodically.sh`):
- Downloads specified model from HuggingFace
- Runs evaluation using `high_level_piano_eval.py`
- Logs results to wandb
- Repeats every 30 minutes

### Setup

1. Environment Variables:
```bash
# Add to .env file
HF_READ_TOKEN=<your_huggingface_read_token>  # Required for downloads
HF_WRITE_TOKEN=<your_huggingface_write_token>  # Required for uploads
WANDB_API_KEY=<your_wandb_api_key>  # Required for logging
```

2. Start Upload Service:
```bash
./upload_periodically.sh
```

3. Start Evaluation Service (possibly on another device):
```bash
./download_and_calculate_periodically.sh <model_name>
```
The evaluation pipeline is further configurable by `gpt2/configs/eval.yaml`
### How It Works

1. Model Upload Pipeline:
```
Training → Save Checkpoint → upload_periodically.sh → HuggingFace Hub
```

2. Evaluation Pipeline:
```
HuggingFace Hub → download_and_calculate_periodically.sh → high_level_piano_eval.py → Wandb
```

3. Validation Splits and Metrics:
- Full validation set from base dataset
- Composer-specific splits (Bach, Chopin, Mozart)
- Loss metrics
- Piano-specific metrics
- Example generations with visualizations


## Evaluation Metrics

The evaluation uses metrics from the piano_dataset package to evaluate generated MIDI. Currently it uses a default metrics configuration TODO: METRICS CONFIGS

```python
# high_level_piano_eval.py
metrics_manager = MetricsManager.load_default()
metric_results = metrics_manager.calculate_all(
    target_df=target_df,    # Original piece
    generated_df=generated_df  # Model output
)
```

Implementation:

```python
def run_eval(cfg, model, ...):
    # Initialize metrics
    metrics_manager = MetricsManager.load_default()

    for split in splits:
        for it in range(cfg.eval_iters):
            # Generate output
            generated_token_ids = model.generate(...)
            generated_df = tokenizer.decode(token_ids=generated_token_ids)
            target_df = tokenizer.decode(token_ids=original_token_ids)

            # Calculate metrics
            metric_results = metrics_manager.calculate_all(
                target_df=target_df,
                generated_df=generated_df,
            )

            # Each metric returns a MetricResult containing:
            # - name: metric identifier
            # - metrics: dict of calculated values
            # - metadata: additional computation info
            # - metric_config: configuration used
```

Results are logged to wandb with each metric getting its own series.

## Validation Splits

Each metric is calculated on different dataset splits:

1. Full Validation:
   - Complete validation set from base dataset
   - Used for overall model quality

2. Composer-Specific:
   - Bach compositions
   - Chopin compositions
   - Mozart compositions
   - Tests style consistency


## Logging

Metrics are logged to wandb with prefix based on split:
```python
metrics_flat = {}
for split in metrics:
    metrics_flat |= {
        f"metrics/{split}_{metric_name}": value
        for metric_name, value in metrics[split].items()
    }

wandb.log({
    "iter": checkpoint["iter_num"],
    "total_tokens": checkpoint["total_tokens"],
    **metrics_flat,
})
```


### Script Details
> ⚠️ WARNING: Currently "epr-labs" repository is hardcoded: you will need to change that
1. Upload Script:
```python
# upload_models.py
def upload_models():
    # Initialize HF API
    api = HfApi()

    # Upload all models from checkpoints directory
    try:
        api.upload_folder(
            folder_path=CHECKPOINTS_DIR,
            repo_id="epr-labs/piano-gpt",
            repo_type="model",
            token=HF_WRITE_TOKEN
        )
    except Exception as e:
        print(f"Error uploading models: {str(e)}")
```

2. Download Script:
```python
# download_model.py
def download_model(repo_id: str, filename: str) -> str:
    """Download model from HuggingFace Hub"""
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=HF_READ_TOKEN,
            local_dir=MODELS_DIR
        )
        return local_path
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return None
```

3. Periodic Execution:
```bash
while true; do
    start_time=$(date +%s)
    run_commands

    # Sleep until next 30-minute interval
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    if [ $duration -lt 1800 ]; then
        sleep_time=$((1800 - duration))
        sleep $sleep_time
    fi
done
```

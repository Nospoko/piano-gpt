PYTHONPATH=. torchrun --nproc-per-node=4 -m gpt2.main \
    dataset=augmented_every \
    model_task=piano_task \
    training.batch_size=256 \
    training.microbatch_size=16 \
    lr=cosine_decay \
    lr.warmup_iters=3000 \
    lr.lr_decay_iters=280000 \
    lr.learning_rate=6e-4 \
    lr.min_lr=1e-4 \
    model=gpt2_medium \
    system.data_workers=128 \
    system.compile=true \
    training.prompt_masking=false \
    training.max_notes_per_record=168 \
    training.context_size=1024

PYTHONPATH=. torchrun --nproc-per-node=4 -m gpt2.main \
    dataset=augmented_every \
    model_task=piano_task \
    training.batch_size=256 \
    training.microbatch_size=32 \
    lr=cosine_decay \
    lr.warmup_iters=10000 \
    lr.lr_decay_iters=300000 \
    lr.learning_rate=1e-4 \
    lr.min_lr=2e-5 \
    model=gpt2_medium \
    system.data_workers=128 \
    system.compile=true \
    training.prompt_masking=false \
    training.max_notes_per_record=72 \
    training.context_size=512

PYTHONPATH=. torchrun --nproc-per-node=4 -m gpt2.main \
    dataset=augmented_every \
    model_task=piano_task \
    training.batch_size=96 \
    training.microbatch_size=8 \
    lr=cosine_decay \
    lr.warmup_iters=5000 \
    lr.lr_decay_iters=300000 \
    lr.learning_rate=1e-4 \
    lr.min_lr=2e-5 \
    model=gpt2_medium \
    system.data_workers=128 \
    system.compile=true \
    training.loss_masking=pretrianing

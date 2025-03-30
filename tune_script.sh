PYTHONPATH=. torchrun --nproc-per-node=4 -m gpt2.main \
    --config-name=model_tuning \
    checkpoint_path=tmp/checkpoints/315M-pretraining-2025-03-23-21-37-best.pt

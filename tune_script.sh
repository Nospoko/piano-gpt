PYTHONPATH=. torchrun --nproc-per-node=4 -m gpt2.main \
    --config-name=model_tuning \
    checkpoint_path=tmp/checkpoints/314M-pretraining-2025-03-06-07-34-best.pt

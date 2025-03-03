PYTHONPATH=. torchrun --nproc-per-node=4 -m gpt2.main \
    --config-name=model_tuning \
    checkpoint_path=tmp/checkpoints/303M-pretraining-2025-02-27-20-53-best.pt

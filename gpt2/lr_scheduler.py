import math


class CosineDecayLRScheduler:
    def __init__(self, scheduler_config: dict):
        self.min_lr = scheduler_config["min_lr"]
        self.warmup_iters = scheduler_config["warmup_iters"]
        self.learning_rate = scheduler_config["learning_rate"]
        self.lr_decay_iters = scheduler_config["lr_decay_iters"]

    def get_lr(self, it: int) -> float:
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters

        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr

        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1

        # coeff ranges 0..1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)


class ConstandLRScheduler:
    def __init__(self, scheduler_config: dict):
        self.learning_rate = scheduler_config["learning_rate"]

    def get_lr(self, it: int) -> float:
        return self.learning_rate


def get_lr_scheduler(lr_config: dict):
    scheduler_type = lr_config.pop("type")

    if scheduler_type == "CosineDecay":
        lr_scheduler = CosineDecayLRScheduler(lr_config)
    elif scheduler_type == "Constant":
        lr_scheduler = ConstandLRScheduler(lr_config)
    else:
        raise ValueError(f"LR type not supported {scheduler_type}")

    return lr_scheduler

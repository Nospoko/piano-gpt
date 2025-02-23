from dataclasses import dataclass


@dataclass
class RunStats:
    # number of iterations in the lifetime of this process
    # we count from 1 because of unknown reasons
    iter: int = 1

    total_tokens: int = 0
    running_mfu: float = -1.0
    best_val_loss: float = 1e9

import os
import math
from contextlib import nullcontext

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import Sampler, DataLoader
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer

from data.dataset import MidiDataset
from gpt2.metrics import calculate_f1
from gpt2.model import GPT, GPTConfig
from gpt2.utils import get_dataset_for_task
from data.random_sampler import ValidationRandomSampler

load_dotenv()


class CyclicalDataLoader:
    def __init__(
        self,
        dataset: MidiDataset,
        sampler: Sampler,
        batch_size: int,
        pin_memory: bool = False,
        num_workers: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.device = device
        self.dataloader = DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=num_workers,
        )
        self.iterator = iter(self.dataloader)

    def get_batch(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)

        x = batch["source_token_ids"].to(self.device, non_blocking=True)
        y = batch["target_token_ids"].to(self.device, non_blocking=True)
        mask = batch["target_mask"].to(self.device, non_blocking=True)
        prompt_lengths = batch["prompt_length"]
        return x, y, mask, prompt_lengths


def process_batch_item(generation, prompt_length, labels, tokenizer):
    output = generation.cpu().numpy()

    out_tokens = [tokenizer.vocab[token_id] for token_id in output]
    target_tokens = [tokenizer.vocab[token_id] for token_id in labels]

    generated_tokens = out_tokens[prompt_length:]
    true_tokens = target_tokens[prompt_length + 1 :]

    generated_notes = tokenizer.untokenize(generated_tokens)
    true_notes = tokenizer.untokenize(true_tokens)
    generated_notes = generated_notes.iloc[: len(true_notes)]

    return {
        "pitch": (generated_notes.pitch == true_notes.pitch).sum() / len(true_notes),
        "velocity": (generated_notes.velocity == true_notes.velocity).sum() / len(true_notes),
        "start": (abs(generated_notes.start - true_notes.start) < 0.01).sum() / len(true_notes),
        "end": (abs(generated_notes.end - true_notes.end) < 0.01).sum() / len(true_notes),
    }


@hydra.main(config_path="configs", config_name="eval", version_base=None)
@torch.no_grad()
def main(cfg: DictConfig):
    model_args = dict(
        n_layer=cfg.model.n_layer,
        n_head=cfg.model.n_head,
        n_embd=cfg.model.n_embd,
        block_size=cfg.data.sequence_length,
        bias=cfg.model.bias,
        vocab_size=None,
        dropout=cfg.model.dropout,
    )

    device = cfg.system.device
    if cfg.init_from.startswith("midi-gpt2"):
        ckpt_path = os.path.join("checkpoints/", cfg.init_from)
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        checkpoint_cfg = OmegaConf.create(checkpoint["config"])

        cfg.model = checkpoint_cfg.model
        cfg.tokenizer = checkpoint_cfg.tokenizer
        cfg.system.dtype = checkpoint_cfg.system.dtype

        if cfg.tokenizer.name == "ExponentialTimeTokenizer":
            tokenizer = ExponentialTimeTokenizer.from_dict(tokenizer_desc=checkpoint["tokenizer"])
        else:
            tokenizer = AwesomeMidiTokenizer.from_dict(tokenizer_desc=checkpoint["tokenizer"])

        val_datasets = get_dataset_for_task(cfg=cfg, tokenizer=tokenizer)[1]
        pad_token_id = tokenizer.token_to_id["<PAD>"]

        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]

        gptconf = GPTConfig(**model_args)
        model = GPT(config=gptconf, pad_token_id=pad_token_id)
        state_dict = checkpoint["model"]

        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(math.floor(cfg.system.data_workers))

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.system.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    val_samplers = [
        ValidationRandomSampler(
            data_source=dataset,
            seed=4,
            num_samples=cfg.data.batch_size * cfg.eval_iters,
        )
        for dataset in val_datasets
    ]

    val_loaders = [
        CyclicalDataLoader(
            dataset,
            sampler=sampler,
            batch_size=cfg.data.batch_size,
            pin_memory=device_type == "cuda",
            num_workers=cfg.system.data_workers,
            device=device,
        )
        for dataset, sampler in zip(val_datasets, val_samplers)
    ]

    if cfg.data.sequence_length < model.config.block_size:
        model.crop_block_size(cfg.data.sequence_length)
        model_args["block_size"] = cfg.data.sequence_length

    model.to(device)

    if cfg.system.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        splits = ["val", "bach", "chopin", "mozart"]

        for split, loader in zip(splits, val_loaders):
            losses = torch.zeros(cfg.eval_iters)
            f1s = torch.zeros(cfg.eval_iters)

            for k in range(cfg.eval_iters):
                X, Y, mask, prompt_lengths = loader.get_batch()
                target_prefix_tokens = 1 if cfg.task == "multi" else 2

                with ctx:
                    logits, loss = model(X, Y, mask)
                    out_tokens = model.generate(
                        X[:, : prompt_lengths[0] + target_prefix_tokens],
                        max_new_tokens=cfg.data.sequence_length - min(prompt_lengths),
                        temperature=1,
                    )

                batch_f1_scores = torch.zeros(cfg.data.batch_size)
                for b in range(cfg.data.batch_size):
                    generated_tokens = tokenizer.decode(token_ids=out_tokens[b, prompt_lengths[b] :].cpu().numpy())
                    original_tokens = tokenizer.decode(token_ids=Y[b, prompt_lengths[b] :].cpu().numpy())
                    batch_f1_scores[b] = calculate_f1(
                        target_df=original_tokens, generated_df=generated_tokens, velocity_threshold=30
                    )[0]

                f1s[k] = batch_f1_scores.mean()
                losses[k] = loss.item()

                if k % 5 == 0:
                    print(f"{split}, iter: {k}, loss: {loss.item():.4f}, f1: {f1s[k]:.4f}")

            out[split] = {"loss": losses.mean().item(), "f1": f1s.mean().item()}

        return out

    metrics = estimate_loss()
    print("\nFinal metrics:")
    for split, split_metrics in metrics.items():
        print(f"\n{split}:")
        for metric, value in split_metrics.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()

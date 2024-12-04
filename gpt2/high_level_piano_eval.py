import os
import math
from contextlib import nullcontext

import hydra
import torch
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import Sampler, DataLoader
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer

from data.dataset import MidiDataset
from gpt2.model import GPT, GPTConfig
from data.random_sampler import ValidationRandomSampler
from gpt2.utils import get_dataset_for_task, create_metrics_runner

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
    def run_eval():
        metrics_runner = create_metrics_runner(cfg=cfg)
        out = {}
        model.eval()
        splits = ["val", "bach", "chopin", "mozart"]

        # For visualization
        example_generations = {}

        for split, loader in zip(splits, val_loaders):
            metric_trackers = {
                "loss": torch.zeros(cfg.eval_iters),
            }
            visualized = False

            for k in range(cfg.eval_iters):
                X, Y, mask, prompt_lengths = loader.get_batch()
                target_prefix_tokens = {"multi": 1, "multi_with_composer": 2}.get(cfg.task, 0)

                with ctx:
                    logits, loss = model(X, Y, mask)

                batch_metrics = {}
                for b in range(cfg.data.batch_size):
                    input_token_ids = torch.unsqueeze(X[b, : prompt_lengths[b] + target_prefix_tokens], 0)
                    out_tokens = model.generate(
                        input_token_ids,
                        max_new_tokens=cfg.data.sequence_length - prompt_lengths[b],
                        temperature=1,
                    )
                    generated_df = tokenizer.decode(token_ids=out_tokens[b, prompt_lengths[b] :].cpu().numpy())
                    original_df = tokenizer.decode(token_ids=Y[b, prompt_lengths[b] :].cpu().numpy())
                    # Cropping because we have no EOS token
                    generated_df = generated_df[generated_df.start < original_df.end.max()]

                    example_metrics = metrics_runner.calculate_all(
                        target_df=original_df,
                        generated_df=generated_df,
                    )

                    # Store first example from each split for visualization
                    if not visualized and b == 0:
                        prompt_df = tokenizer.decode(token_ids=X[b, : prompt_lengths[b]].cpu().numpy())
                        example_generations[split] = {
                            "prompt": prompt_df,
                            "generated": generated_df,
                            "original": original_df,
                        }
                        visualized = True

                    # Aggregate metrics across batch
                    for metric_name, result in example_metrics.items():
                        if metric_name not in batch_metrics:
                            batch_metrics[metric_name] = []
                        batch_metrics[metric_name].append(result.value)

                metric_trackers["loss"][k] = loss.item()
                for metric_name, values in batch_metrics.items():
                    if metric_name not in metric_trackers:
                        metric_trackers[metric_name] = torch.zeros(cfg.eval_iters)
                    metric_trackers[metric_name][k] = torch.tensor(values).mean()

                if k % 5 == 0:
                    metrics_str = f"{split}, iter: {k}, loss: {loss.item():.4f}"
                    for metric_name, tracker in metric_trackers.items():
                        if metric_name != "loss":
                            metrics_str += f", {metric_name}: {tracker[k]:.4f}"
                    print(metrics_str)

            # Compute final metrics for this split
            out[split] = {name: values.mean().item() for name, values in metric_trackers.items()}

        return out, example_generations

    metrics, example_generations = run_eval()
    print("\nFinal metrics:")
    for split, split_metrics in metrics.items():
        print(f"\n{split}:")
        for metric, value in split_metrics.items():
            print(f"{metric}: {value:.4f}")

    st.title("Generation Examples")

    for split, data in example_generations.items():
        st.write(f"\n### {split} Split Example")

        # Create fortepyan pieces
        prompt_piece = ff.MidiPiece(data["prompt"])
        generated_piece = ff.MidiPiece(data["generated"])

        original_piece = ff.MidiPiece(data["original"])
        if "next_token_prediction" in cfg.task:
            generated_piece.time_shift(prompt_piece.end)
            original_piece.time_shift(prompt_piece.end)

        st.write("#### Prompt + Generated")
        streamlit_pianoroll.from_fortepyan(
            piece=ff.MidiPiece(data["prompt"]),
            secondary_piece=ff.MidiPiece(data["generated"]),
        )
        st.write("#### Original")
        streamlit_pianoroll.from_fortepyan(
            piece=prompt_piece,
            secondary_piece=original_piece,
        )


if __name__ == "__main__":
    main()

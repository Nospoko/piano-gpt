import os
import math
from contextlib import nullcontext

import hydra
import torch
import wandb
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


# TODO This should have own module
# FIXME Looks like a blunt copy of code from train.py
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
    device = cfg.system.device
    # TODO Is this ever not true? Seems like a lot would blow up
    # when this if Falses
    if cfg.init_from.startswith("midi-gpt2"):
        ckpt_path = os.path.join("checkpoints/", cfg.init_from)
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        checkpoint_cfg = OmegaConf.create(checkpoint["config"])

        # FIXME This is a bit confusing, I think that we should separate eval_cfg
        # from checkpoint_cfg, not try to turn one into the other
        cfg.model = checkpoint_cfg.model
        cfg.tokenizer = checkpoint_cfg.tokenizer
        cfg.system.dtype = checkpoint_cfg.system.dtype

        if cfg.tokenizer.name == "ExponentialTimeTokenizer":
            # What is *tokenizer_desc*? Naming should be consistent
            # so it should be stored as checkpoint["tokenizer_desc"] (or renamed)
            tokenizer = ExponentialTimeTokenizer.from_dict(tokenizer_desc=checkpoint["tokenizer"])
        else:
            tokenizer = AwesomeMidiTokenizer.from_dict(tokenizer_desc=checkpoint["tokenizer"])

        # TODO Manage data structures in a way that doesn't require a magic [1]
        val_datasets = get_dataset_for_task(cfg=cfg, tokenizer=tokenizer)[1]
        pad_token_id = tokenizer.token_to_id["<PAD>"]

        # I think this should look more like:
        # model_args = checkpoint_cfg.model_args
        # TODO What's the point of model args, if we
        # have to load the model from checkpoint, where model_args are set
        model_args = dict(
            n_layer=cfg.model.n_layer,
            n_head=cfg.model.n_head,
            n_embd=cfg.model.n_embd,
            # TODO Naming consistency pls
            block_size=cfg.data.sequence_length,
            bias=cfg.model.bias,
            vocab_size=None,
            dropout=cfg.model.dropout,
        )
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]

        gptconf = GPTConfig(**model_args)
        model = GPT(config=gptconf, pad_token_id=pad_token_id)
        state_dict = checkpoint["model"]

        # What's that?
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)

    # Feels like those should be in config
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
            dataset=dataset,
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

    # This is a nested *no_grad*, does it have any effect?
    @torch.no_grad()
    def run_eval():
        metrics_runner = create_metrics_runner(cfg=cfg)
        # TODO What's the difference between *out*, *metric_trackers*,
        # *batch_metrics*, *example_metrics* (variable names should reflect those differences)
        out = {}
        model.eval()
        splits = ["val", "bach", "chopin", "mozart"]

        # For visualization
        example_generations = {}

        for split, loader in zip(splits, val_loaders):
            metric_trackers = {
                "loss": torch.zeros(cfg.eval_iters),
            }

            # TODO What is *k*? Number of batches used for validation?
            for k in range(cfg.eval_iters):
                X, Y, mask, prompt_lengths = loader.get_batch()
                # TODO Where are task tokens defined?
                # TODO Where to look for an explanation about what this is
                target_prefix_tokens = {"multi": 1, "multi_with_composer": 2}.get(cfg.task, 0)

                batch_metrics = {}
                # TODO what is *b*? the name doesn't tell us anything about the dimension
                # it's supposed to iterate over. I guess it's samples within batch, but is it?
                for b in range(X.shape[0]):
                    input_token_ids = torch.unsqueeze(
                        input=X[b, : prompt_lengths[b] + target_prefix_tokens],
                        dim=0,
                    )
                    out_tokens = model.generate(
                        input_token_ids,
                        max_new_tokens=2048 - prompt_lengths[b],
                        temperature=1,
                    )
                    generated_token_ids = out_tokens[0, prompt_lengths[b] :].cpu().numpy()
                    generated_df = tokenizer.decode(token_ids=generated_token_ids)

                    original_token_ids = Y[0, prompt_lengths[b] :].cpu().numpy()
                    original_df = tokenizer.decode(token_ids=original_token_ids)

                    # Cropping because we have no EOS token
                    if not generated_df.empty:
                        generated_df = generated_df[generated_df.start < original_df.end.max()]

                    # Store first example from each split for visualization
                    if not k == 0 and b == 0:
                        prompt_token_ids = X[b, : prompt_lengths[b]].cpu().numpy()
                        prompt_df = tokenizer.decode(token_ids=prompt_token_ids)

                        # 5x nesting is a red flag :eyes:
                        example_generations[split] = {
                            "prompt": prompt_df,
                            "generated": generated_df,
                            "original": original_df,
                        }

                    # TODO keep consistent variable names, original_df should be target_df
                    # (unless there's a good reason not to).
                    # Calculate all Piano metrics
                    example_metrics = metrics_runner.calculate_all(
                        target_df=original_df,
                        generated_df=generated_df,
                    )

                    # Aggregate metrics across batch
                    for metric_name, result in example_metrics.items():
                        if metric_name not in batch_metrics:
                            batch_metrics[metric_name] = []

                        batch_metrics[metric_name].append(result.value)

                with ctx:
                    logits, loss = model(X, Y, mask)

                print(loss)
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

    if cfg.logging.wandb_log:
        wandb.init(id=checkpoint["wandb_id"], resume="must")
        metrics_flat = {}
        for split in metrics:
            metrics_flat |= {
                f"metrics/{split}_{metric_name}": metric_value for metric_name, metric_value in metrics[split].items()
            }

        wandb_logs = {
            "iter": checkpoint["iter_num"],
            "total_tokens": checkpoint["total_tokens"],
            **metrics_flat,
        }
        wandb.log(wandb_logs)
        print(f"wandb logged: {wandb_logs}")

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

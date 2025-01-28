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
from piano_metrics.piano_metric import MetricsManager
from piano_dataset.piano_tasks import ParametricTaskManager
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer

from gpt2.model import GPT, GPTConfig
from data.random_sampler import ValidationRandomSampler
from gpt2.utils import create_piano_datasets, create_augmented_dataset

load_dotenv()


@hydra.main(config_path="configs", config_name="eval", version_base=None)
@torch.no_grad()
def main(cfg: DictConfig):
    device = cfg.system.device

    ckpt_path = os.path.join("checkpoints/", cfg.init_from)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    checkpoint_cfg = OmegaConf.create(checkpoint["config"])

    if checkpoint_cfg.tokenizer.class_name == "ExponentialTimeTokenizer":
        # What is *tokenizer_desc*? Naming should be consistent
        # so it should be stored as checkpoint["tokenizer_desc"] (or renamed)
        tokenizer = ExponentialTimeTokenizer.from_dict(tokenizer_desc=checkpoint["tokenizer"])
    else:
        tokenizer = AwesomeMidiTokenizer.from_dict(tokenizer_desc=checkpoint["tokenizer"])

    piano_task_manager = ParametricTaskManager.load_default()
    hf_dataset = create_augmented_dataset(cfg)
    val_datasets = create_piano_datasets(
        hf_dataset=hf_dataset,
        cfg=cfg,
        tokenizer=tokenizer,
        piano_task_manager=piano_task_manager,
    )["validation_splits"]

    model_args = {}
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    pad_token_id = tokenizer.token_to_id["<PAD>"]
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
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[checkpoint_cfg.system.dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(
            device_type=device_type,
            dtype=ptdtype,
        )
    )

    val_samplers = [
        ValidationRandomSampler(
            data_source=dataset,
            seed=4,
            num_samples=cfg.data.batch_size * cfg.eval_iters,
        )
        for dataset in val_datasets
    ]

    # This can happen if we pretrain a model with a HUGE context size,
    # but want to use smaller context size for finetuning
    if cfg.data.sequence_length < model.config.block_size:
        model.crop_block_size(cfg.data.sequence_length)
        model_args["block_size"] = cfg.data.sequence_length

    model.to(device)

    if cfg.system.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    # TODO: What's the point of nesting this function? Separate functions with many arguments is better
    # This is a nested *no_grad*, does it have any effect?
    @torch.no_grad()
    def run_eval():
        # metrics_runner = create_metrics_runner(cfg=cfg)
        metrics_manager = MetricsManager.load_default()
        # TODO What's the difference between *out*, *metric_trackers*,
        # *batch_metrics*, *example_metrics* (variable names should reflect those differences)
        out = {}
        model.eval()
        splits = ["full_val", "bach", "chopin", "mozart"]

        # For visualization
        example_generations = {}

        for split, sampler in zip(splits, val_samplers):
            # FIXME Not a good way to initialize
            metric_trackers = {
                "loss": torch.zeros(cfg.eval_iters),
            }
            sampler_iter = iter(sampler)

            # TODO What is *k*? Number of batches used for validation? // yes
            for it in range(cfg.eval_iters):
                sample_idx = next(sampler_iter)
                # FIXME we probably don't have to use the batching loader here
                # and we can just go 1 by 1 through dataset records. That way we should
                # be able to get all the details neccessary to calculate PIANO metrics
                # (see __getitem__ in PianoDataset)
                record = val_datasets[split][sample_idx]
                prompt_length = record["prompt_length"]

                source_token_ids = record["source_token_ids"].to(device)
                target_token_ids = record["target_token_ids"].to(device)
                mask = record["target_mask"].to(device)

                num_target_prefix_token = 1  # <GENAI> only

                batch_metrics = {}
                input_token_ids = torch.unsqueeze(
                    input=source_token_ids[: prompt_length + num_target_prefix_token],
                    dim=0,
                )

                out_tokens = model.generate(
                    input_token_ids,
                    max_new_tokens=2048 - prompt_length,
                    temperature=cfg.temperature,
                )
                generated_token_ids = out_tokens[0, prompt_length:].cpu().numpy()
                generated_df = tokenizer.decode(token_ids=generated_token_ids)

                original_token_ids = target_token_ids[prompt_length:].cpu().numpy()
                target_df = tokenizer.decode(token_ids=original_token_ids)

                # Cropping because we have no EOS token
                if not generated_df.empty:
                    # So we want to remove notes that started, but not ended
                    ids = generated_df.start < target_df.end.max()
                    generated_df = generated_df[ids]

                # Store first example from each split for visualization
                prompt_token_ids = source_token_ids[:prompt_length].cpu().numpy()
                prompt_df = tokenizer.decode(token_ids=prompt_token_ids)

                # 4x nesting is a red flag :eyes:
                # TODO: Implement of a better way of showing examples
                example_generations[split] = {
                    "prompt": prompt_df,
                    "generated": generated_df,
                    "original": target_df,
                }

                # TODO keep consistent variable names, original_df should be target_df
                # (unless there's a good reason not to).
                # Calculate all Piano metrics
                # FIXME *original_df* is not the correct input here.
                metric_results = metrics_manager.calculate_all(
                    target_df=target_df,
                    generated_df=generated_df,
                )

                # TODO I was not able to test it (this loops needs a redesign)
                # so this may break something. I did try to recreate the same
                # storage in "batch_metrics", but with new names coming from the PIANO package
                for metric_result in metric_results:
                    # Each PIANO metric calculates multiple numbers we can track
                    for sub_metric_name, metric_value in metric_result.metrics.items():
                        metric_name = f"{metric_result.name}/{sub_metric_name}"

                        # FIXME Not a good way to initialize
                        if metric_name not in batch_metrics:
                            batch_metrics[metric_name] = []

                        batch_metrics[metric_name].append(metric_value)

                # Calculate loss as we would during training
                with ctx:
                    training_input_ids = torch.unsqueeze(
                        input=source_token_ids,
                        dim=0,
                    )
                    training_target_ids = torch.unsqueeze(
                        input=target_token_ids,
                        dim=0,
                    )
                    training_mask = torch.unsqueeze(input=mask, dim=0)
                    logits, loss = model(training_input_ids, training_target_ids, training_mask)

                print(f"loss: {loss.item()}")
                metric_trackers["loss"][it] = loss.item()
                for metric_name, values in batch_metrics.items():
                    if metric_name not in metric_trackers:
                        metric_trackers[metric_name] = torch.zeros(cfg.eval_iters)
                    metric_trackers[metric_name][it] = torch.tensor(values).mean()

                if it % 5 == 0:
                    metrics_str = f"{split}, iter: {it}, loss: {loss.item():.4f}"
                    for metric_name, tracker in metric_trackers.items():
                        if metric_name != "loss":
                            metrics_str += f", {metric_name}: {tracker[it]:.4f}"
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
        if "next_token_pretraining" in checkpoint_cfg.stage:
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

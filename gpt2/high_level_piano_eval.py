import json
import math
import time
from contextlib import nullcontext

import hydra
import torch
import wandb
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
from dotenv import load_dotenv
from torch.utils.data import Dataset
from omegaconf import OmegaConf, DictConfig
from piano_metrics.piano_metric import MetricsManager
from piano_dataset.piano_tasks import PianoTaskManager
from midi_tokenizers import MidiTokenizer, AwesomeMidiTokenizer, ExponentialTimeTokenizer

import gpt2.utils as utils
from gpt2.model import GPT
from data.random_sampler import ValidationRandomSampler

load_dotenv()


@torch.no_grad()
def run_eval(
    cfg: DictConfig,
    model: GPT,
    val_samplers: dict[str, ValidationRandomSampler],
    val_datasets: dict[str, Dataset],
    device: torch.device,
    tokenizer: MidiTokenizer,
    dtype: str,
):
    # metrics_runner = create_metrics_runner(cfg=cfg)
    metrics_manager = MetricsManager.load_default()
    # TODO What's the difference between *out*, *metric_trackers*,
    # *batch_metrics*, *example_metrics* (variable names should reflect those differences)
    out = {}
    model.eval()

    # For visualization
    example_generations = {}

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    # TODO What's the difference?
    if device_type == "cpu":
        ctx = nullcontext()
    else:
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    for split, sampler in val_samplers.items():
        print("Validation for split:", split)
        # FIXME Not a good way to initialize
        metric_trackers = {
            "loss": torch.zeros(cfg.eval_iters),
        }
        sampler_iter = iter(sampler)

        for it in range(cfg.eval_iters):
            t0 = time.time()
            sample_idx = next(sampler_iter)
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

            # TODO What is 2048? How is this related to the context length? Shouldn't this be in the config?
            out_tokens = model.generate(
                input_token_ids,
                max_new_tokens=2048 - prompt_length,
                temperature=cfg.temperature,
            )
            generated_token_ids = out_tokens[0, prompt_length:].cpu().numpy()
            generated_df = tokenizer.decode(token_ids=generated_token_ids)

            if generated_df.empty:
                print("ERROR! No valid dataframe generated")
                continue
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

            # TODO: Implement of a better way of showing examples
            example_generations[split] = {
                "prompt": prompt_df,
                "generated": generated_df,
                "original": target_df,
            }

            # Calculate all Piano metrics
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

            metric_trackers["loss"][it] = loss.item()
            for metric_name, values in batch_metrics.items():
                if metric_name not in metric_trackers:
                    metric_trackers[metric_name] = torch.zeros(cfg.eval_iters)
                metric_trackers[metric_name][it] = torch.tensor(values).mean()

            iteration_time = time.time() - t0
            if it % 5 == 0:
                print(f"Iteration time: {iteration_time:.2f}s")
                metrics_str = f"{split}, iter: {it}/{cfg.eval_iters}, loss: {loss.item():.4f}"
                for metric_name, tracker in metric_trackers.items():
                    if metric_name != "loss":
                        metrics_str += f"\n- {metric_name}: {tracker[it]:.4f}"
                print(metrics_str)

        # Compute final metrics for this split
        out[split] = {name: values.mean().item() for name, values in metric_trackers.items()}

    return out, example_generations


@hydra.main(config_path="configs", config_name="eval", version_base=None)
@torch.no_grad()
def main(cfg: DictConfig):
    device = cfg.system.device

    print("Running high level piano eval with config:")
    print(json.dumps(OmegaConf.to_container(cfg), indent=2))

    ckpt_path = cfg.init_from
    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )
    checkpoint_cfg = OmegaConf.create(checkpoint["run_config"])

    if checkpoint["tokenizer_desc"]["name"] == "ExponentialTimeTokenizer":
        # What is *tokenizer_desc*? Naming should be consistent
        # so it should be stored as checkpoint["tokenizer"] (or renamed)
        tokenizer = ExponentialTimeTokenizer.from_dict(tokenizer_desc=checkpoint["tokenizer_desc"])
    else:
        tokenizer = AwesomeMidiTokenizer.from_dict(tokenizer_desc=checkpoint["tokenizer_desc"])

    if checkpoint_cfg.stage == "piano_task":
        # TODO: Make it possible to run on a subset of tasks from training
        tasks_config = checkpoint["piano_task_config"]
        piano_task_manager = PianoTaskManager(tasks_config=tasks_config)
        hf_dataset = utils.create_augmented_dataset(cfg)
        val_datasets = utils.create_piano_datasets(
            hf_dataset=hf_dataset,
            cfg=cfg,
            tokenizer=tokenizer,
            piano_task_manager=piano_task_manager,
        )["validation_splits"]
    elif checkpoint_cfg.stage == "next_token_pretraining":
        hf_dataset = utils.create_tokenized_dataset(
            cfg=cfg,
            tokenizer=tokenizer,
        )
        val_datasets = utils.create_next_token_datasets(
            hf_dataset=hf_dataset,
            cfg=cfg,
            tokenizer=tokenizer,
        )["validation_splits"]

    # To speed up the eval process you can pick less validation splits in the eval cfg
    val_datasets = {}
    for split_name, dataset in val_datasets.items():
        if split_name in cfg.eval_splits:
            val_datasets[split_name] = dataset

    pad_token_id = tokenizer.token_to_id["<PAD>"]
    model_cfg = checkpoint["model_cfg"]
    model = GPT(
        config=model_cfg,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=pad_token_id,
    )
    state_dict = checkpoint["model"]

    # What's that?
    # saved model weights sometimes get this prefix, I do not know wht :(
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

    val_samplers = {
        split_name: ValidationRandomSampler(
            n_records=len(dataset),
            seed=4,
            num_samples=cfg.data.batch_size * cfg.eval_iters,
        )
        for split_name, dataset in val_datasets.items()
    }

    # This can happen if we pretrain a model with a HUGE context size,
    # but want to use smaller context size for finetuning
    if cfg.data.context_size < model.config.context_size:
        model.crop_context_size(cfg.data.context_size)
        model_cfg["context_size"] = cfg.data.context_size

    model.to(device)

    if cfg.system.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    metrics, example_generations = run_eval(
        cfg=cfg,
        model=model,
        val_samplers=val_samplers,
        val_datasets=val_datasets,
        device=device,
        tokenizer=tokenizer,
        dtype=cfg.system.dtype,
    )

    if cfg.logging.wandb_log:
        setup_wandb(
            eval_cfg=cfg,
            wandb_group=checkpoint["wandb_run_name"],
        )

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


def setup_wandb(eval_cfg: dict, wandb_group: str):
    # This is because I want to resume a run only knowing it's group
    # and wandb is being defensive about discovery options
    api = wandb.Api()

    # so we have to check all the runs within that group
    # to see if we already have an eval run
    runs = api.runs(
        path="roszcz/piano-gpt",
        filters={"group": wandb_group},
    )

    eval_run_name = "high-level-eval"
    eval_run = next((run for run in runs if run.name == eval_run_name), None)

    if eval_run is None:
        print("Initializing wandb eval run!")
        eval_config = OmegaConf.to_container(eval_cfg)
        wandb.init(
            project=eval_cfg.logging.wandb_project,
            name="high-level-eval",
            group=wandb_group,
            config=eval_config,
            dir="tmp",
        )
    else:
        print("Appending to an existing wandb run:", eval_run.name)
        wandb.init(
            project=eval_cfg.logging.wandb_project,
            id=eval_run.id,
            resume="must",
            dir="tmp",
        )


if __name__ == "__main__":
    main()

import json
import time

import hydra
import torch
import wandb
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
from midi_tokenizers import ExponentialTimeTokenizer
from piano_metrics.piano_metric import MetricsManager

from gpt2.model import GPT
from gpt2.setup.hardware import DeviceSetup
from gpt2.setup import datasets as data_setup
from gpt2.setup.datasets import DatasetsSetup
from gpt2.setup import hardware as hardware_setup

load_dotenv()


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(eval_cfg: DictConfig):
    print("Running high level piano eval with config:")
    print(json.dumps(OmegaConf.to_container(eval_cfg), indent=2))

    checkpoint = torch.load(
        eval_cfg.checkpoint_path,
        weights_only=False,
    )
    run_cfg = OmegaConf.create(checkpoint["run_config"])
    if "wandb_id" not in checkpoint:
        print("This script is dedicated to uploading eval results to wandb")
        print("Refusing to run on a checkpoint that wasn't tracked in wandb!")
        return

    device_setup = hardware_setup.setup_device(run_cfg)

    # Load tokenizer
    tokenizer_desc = checkpoint["tokenizer_desc"]
    tokenizer = ExponentialTimeTokenizer.from_dict(tokenizer_desc)

    if run_cfg.model_task == "next_token_prediction":
        datasets_setup = data_setup.next_token_prediction_setup(
            cfg=run_cfg,
            tokenizer=tokenizer,
            device_setup=device_setup,
        )
    elif run_cfg.model_task == "piano_task":
        datasets_setup = data_setup.piano_task_setup(
            cfg=run_cfg,
            tokenizer=tokenizer,
            device_setup=device_setup,
        )

    model_cfg = checkpoint["model_cfg"]
    model = GPT(
        config=model_cfg,
        vocab_size=datasets_setup.tokenizer.vocab_size,
        pad_token_id=datasets_setup.tokenizer.pad_token_id,
    )
    state_dict = checkpoint["model"]
    model.load_state(state_dict=state_dict)
    model.to(device_setup.device)

    if eval_cfg.system.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    metrics, example_generations = run_eval(
        cfg=eval_cfg,
        model=model,
        device_setup=device_setup,
        datasets_setup=datasets_setup,
        dtype=eval_cfg.system.dtype,
    )

    if eval_cfg.logging.wandb_log:
        setup_wandb(
            eval_cfg=eval_cfg,
            wandb_group=checkpoint["run_name"],
        )

        metrics_flat = {}
        for split in metrics:
            metrics_flat |= {
                f"metrics/{split}_{metric_name}": metric_value for metric_name, metric_value in metrics[split].items()
            }

        wandb_logs = {
            "iter": checkpoint["run_stats"]["iter"],
            "total_tokens": checkpoint["run_stats"]["total_tokens"],
            **metrics_flat,
        }
        wandb.log(wandb_logs)
        print(f"wandb logged: {wandb_logs}")

    print("\nFinal metrics:")
    for split, split_metrics in metrics.items():
        print(f"\n{split}:")
        for metric, value in split_metrics.items():
            print(f"{metric}: {value:.4f}")


@torch.no_grad()
def run_eval(
    cfg: DictConfig,
    model: GPT,
    device_setup: DeviceSetup,
    datasets_setup: DatasetsSetup,
    dtype: str,
):
    metrics_manager = MetricsManager.load_default()

    # TODO What's the difference between *out*, *metric_trackers*,
    # *batch_metrics*, *example_metrics* (variable names should reflect those differences)
    out = {}
    model.eval()

    # For visualization
    example_generations = {}

    for split, dataset in datasets_setup.val_datasets.items():
        if split not in cfg.eval_splits:
            continue

        print("Validation for split:", split)

        idxs = torch.randint(len(dataset), size=(cfg.n_eval_samples,))
        records = [dataset[idx] for idx in idxs]

        # FIXME Not a good way to initialize
        metric_trackers = {
            "loss": torch.zeros(cfg.n_eval_samples),
        }

        for it, record in enumerate(records):
            t0 = time.time()
            prompt_length = record["prompt_length"]

            source_token_ids = record["source_token_ids"].to(device_setup.device)
            target_token_ids = record["target_token_ids"].to(device_setup.device)
            mask = record["target_mask"].to(device_setup.device)

            # <GENAI> only
            num_target_prefix_token = 1

            batch_metrics = {}
            input_token_ids = torch.unsqueeze(
                input=source_token_ids[: prompt_length + num_target_prefix_token],
                dim=0,
            )

            # TODO What is 2048?
            # How is this related to the context length? Shouldn't this be in the config?
            out_tokens = model.generate(
                idx=input_token_ids,
                max_new_tokens=2048 - prompt_length,
                temperature=cfg.temperature,
            )
            generated_token_ids = out_tokens[0, prompt_length:].cpu().numpy()
            generated_df = datasets_setup.tokenizer.decode(token_ids=generated_token_ids)

            if generated_df.empty:
                print("No valid dataframe generated, moving forward")
                continue

            original_token_ids = target_token_ids[prompt_length:].cpu().numpy()
            target_df = datasets_setup.tokenizer.decode(token_ids=original_token_ids)

            # Cropping because we have no EOS token
            if not generated_df.empty:
                # So we want to remove notes that started, but not ended
                ids = generated_df.start < target_df.end.max()
                generated_df = generated_df[ids]

            # Store first example from each split for visualization
            prompt_token_ids = source_token_ids[:prompt_length].cpu().numpy()
            prompt_df = datasets_setup.tokenizer.decode(token_ids=prompt_token_ids)

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

            for metric_result in metric_results:
                # Each PIANO metric calculates multiple numbers we can track
                for sub_metric_name, metric_value in metric_result.metrics.items():
                    metric_name = f"{metric_result.name}/{sub_metric_name}"

                    # FIXME Not a good way to initialize
                    if metric_name not in batch_metrics:
                        batch_metrics[metric_name] = []

                    batch_metrics[metric_name].append(metric_value)

            # Calculate loss as we would during training
            with device_setup.autocast_ctx:
                training_input_ids = torch.unsqueeze(
                    input=source_token_ids,
                    dim=0,
                )
                training_target_ids = torch.unsqueeze(
                    input=target_token_ids,
                    dim=0,
                )
                training_mask = torch.unsqueeze(input=mask, dim=0)
                logits, loss = model(
                    idx=training_input_ids,
                    targets=training_target_ids,
                    target_mask=training_mask,
                )

            metric_trackers["loss"][it] = loss.item()
            for metric_name, values in batch_metrics.items():
                if metric_name not in metric_trackers:
                    metric_trackers[metric_name] = torch.zeros(cfg.n_eval_samples)
                metric_trackers[metric_name][it] = torch.tensor(values).mean()

            iteration_time = time.time() - t0
            if it % 5 == 0:
                print(f"Iteration time: {iteration_time:.2f}s")
                metrics_str = f"{split}, iter: {it}/{cfg.n_eval_samples}, loss: {loss.item():.4f}"
                for metric_name, tracker in metric_trackers.items():
                    if metric_name != "loss":
                        metrics_str += f"\n- {metric_name}: {tracker[it]:.4f}"
                print(metrics_str)

        # Compute final metrics for this split
        out[split] = {name: values.mean().item() for name, values in metric_trackers.items()}

    return out, example_generations


def setup_wandb(eval_cfg: dict, wandb_group: str):
    # This is because I want to resume a run only knowing it's group
    # and wandb is being defensive about discovery options
    api = wandb.Api()

    # so we have to check all the runs within that group
    # to see if we already have an eval run
    path = f"{eval_cfg.logging.wandb_entity}/{eval_cfg.logging.wandb_project}"
    runs = api.runs(
        path=path,
        filters={"group": wandb_group},
    )

    eval_run_name = "piano-eval"
    eval_run = next((run for run in runs if run.name == eval_run_name), None)

    if eval_run is None:
        print("Initializing wandb eval run!")
        eval_config = OmegaConf.to_container(eval_cfg)
        wandb.init(
            project=eval_cfg.logging.wandb_project,
            name=eval_run_name,
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

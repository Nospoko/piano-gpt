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
from gpt2.dataloader import EvalDataLoader
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

    # FIXME This is a bit confusing, I think that we should separate eval_cfg
    # from checkpoint_cfg, not try to turn one into the other
    cfg.system.dtype = checkpoint_cfg.system.dtype

    if checkpoint_cfg.tokenizer.name == "ExponentialTimeTokenizer":
        # What is *tokenizer_desc*? Naming should be consistent
        # so it should be stored as checkpoint["tokenizer_desc"] (or renamed)
        tokenizer = ExponentialTimeTokenizer.from_dict(tokenizer_desc=checkpoint["tokenizer"])
    else:
        tokenizer = AwesomeMidiTokenizer.from_dict(tokenizer_desc=checkpoint["tokenizer"])

    piano_task_manager = ParametricTaskManager.load_default()
    # TODO Manage data structures in a way that doesn't require a magic [1]
    hf_dataset = create_augmented_dataset(cfg)
    val_datasets = create_piano_datasets(hf_dataset, cfg, tokenizer, piano_task_manager)["validation_splits"]

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

    val_loaders = [
        EvalDataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=cfg.data.batch_size,
            pin_memory=device_type == "cuda",
            num_workers=cfg.system.data_workers,
            device=device,
        )
        for dataset, sampler in zip(val_datasets, val_samplers)
    ]

    # TODO Why would this happen?
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

        for split, loader in zip(splits, val_loaders):
            # FIXME Not a good way to initialize
            metric_trackers = {
                "loss": torch.zeros(cfg.eval_iters),
            }

            # TODO What is *k*? Number of batches used for validation?
            for k in range(cfg.eval_iters):
                # FIXME we probably don't have to use the batching loader here
                # and we can just go 1 by 1 through dataset records. That way we should
                # be able to get all the details neccessary to calculate PIANO metrics
                # (see __getitem__ in PianoDataset)
                X, Y, mask, prompt_lengths = loader.get_batch()

                # There is one target prefix token for task called "multi" and two for task called "multi_with_composer"
                # TODO: Now the number of target prefix tokens is different, because we are using
                # ParametricalTaskManager.
                num_target_prefix_token = {"multi": 1, "multi_with_composer": 2}.get(cfg.stage, 0)

                batch_metrics = {}
                # TODO what is *b*? the name doesn't tell us anything about the dimension
                # it's supposed to iterate over. I guess it's samples within batch, but is it?
                # TODO What's the point of using batches, if we iterate them by sample?
                for it in range(X.shape[0]):
                    input_token_ids = torch.unsqueeze(
                        input=X[it, : prompt_lengths[it] + num_target_prefix_token],
                        dim=0,
                    )
                    # TODO Temperature should be a subject of evaluation
                    out_tokens = model.generate(
                        input_token_ids,
                        max_new_tokens=2048 - prompt_lengths[it],
                        temperature=cfg.temperature,
                    )
                    generated_token_ids = out_tokens[0, prompt_lengths[it] :].cpu().numpy()
                    generated_df = tokenizer.decode(token_ids=generated_token_ids)

                    original_token_ids = Y[0, prompt_lengths[it] :].cpu().numpy()
                    original_df = tokenizer.decode(token_ids=original_token_ids)

                    # Cropping because we have no EOS token
                    if not generated_df.empty:
                        # So we want to remove notes that started, but not ended
                        ids = generated_df.start < original_df.end.max()
                        generated_df = generated_df[ids]

                    # Store first example from each split for visualization
                    if not k == 0 and it == 0:
                        prompt_token_ids = X[it, : prompt_lengths[it]].cpu().numpy()
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
                    # FIXME *original_df* is not the correct input here.
                    metric_results = metrics_manager.calculate_all(
                        target_df=original_df,
                        generated_df=generated_df,
                    )

                    # TODO I was not able to test it (this loops needs a redesign)
                    # so this may break something. I did try to recreate the same
                    # storage in "batch_metrics", but with new names coming from the PIANO package
                    for metric_result in metric_results:
                        # Each PIANO metric calculates multiple numbers we can track
                        for it, (sub_metric_name, metric_value) in enumerate(metric_result.metrics.items()):
                            metric_name = f"{metric_result.name}/{sub_metric_name}"

                            # FIXME Not a good way to initialize
                            if metric_name not in batch_metrics:
                                batch_metrics[metric_name] = []

                            batch_metrics[metric_name].append(metric_value)

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
        if "next_token_pretraining" in cfg.stage:
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

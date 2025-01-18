import torch
from omegaconf import OmegaConf, DictConfig
from piano_dataset.piano_tasks import ParametricTaskManager
from midi_tokenizers import MidiTokenizer, ExponentialTimeTokenizer

from data.dataset import MidiDataset
from gpt2.model import GPT, GPTConfig
from gpt2.dataset_factory import DatasetFactoryManager
from artifacts import special_tokens_in_the_wrong_place
from data.tokenizer_utils import load_tokenizer_if_exists


def load_cfg(checkpoint: dict) -> DictConfig:
    train_config = checkpoint["config"]
    return OmegaConf.create(train_config)


def load_tokenizer(
    cfg: DictConfig,
    special_tokens: list[str],
):
    tokenizer_cfg = OmegaConf.to_container(cfg.tokenizer)
    tokenizer_parameters = tokenizer_cfg["parameters"]

    # FIXME Hardcoding in a submodule is not a good way to pass special tokens
    # it should be explicit somewhere in the training flow
    special_tokens += special_tokens_in_the_wrong_place
    tokenizer_parameters |= {"special_tokens": special_tokens}

    if cfg.tokenizer.name == "AwesomeMidiTokenizer":
        return load_tokenizer_if_exists(tokenizer_cfg=tokenizer_cfg)
    else:
        return ExponentialTimeTokenizer(**tokenizer_parameters)


def initialize_model(
    cfg: DictConfig,
    checkpoint: dict,
    device: torch.device,
    pad_token_id: int = 0,
) -> GPT:
    """
    Initializes the GPT model using the given configurations and checkpoint.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object.
    dataset_config : dict
        The dataset configuration.
    checkpoint : dict
        The model checkpoint.
    device : torch.device
        The device to load the model on.

    Returns
    -------
    model : GPT
        The initialized GPT model.
    """
    model_args = {
        "n_layer": cfg.model.n_layer,
        "n_head": cfg.model.n_head,
        "n_embd": cfg.model.n_embd,
        "block_size": cfg.data.sequence_length,
        "bias": cfg.model.bias,
        "vocab_size": None,
        "dropout": cfg.model.dropout,
    }

    checkpoint_model_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf, pad_token_id=pad_token_id)
    state_dict = checkpoint["model"]

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model


def get_dataset_for_stage(
    cfg: DictConfig,
    tokenizer: MidiTokenizer,
    piano_task_manager: ParametricTaskManager,
) -> tuple[MidiDataset, tuple]:
    factory_manager = DatasetFactoryManager()
    # TODO training stages can be both used for pretraining. Maybe there can be better names than:
    # 1. next-token-pretraining
    # 2. piano-task-finetuning
    return factory_manager.get_datasets(
        cfg=cfg, tokenizer=tokenizer, piano_task_manager=piano_task_manager if cfg.stage == "piano_task" else None
    )

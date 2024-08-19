import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

from artifacts import special_tokens
from gpt2.model import GPT, GPTConfig
from data.tokenizer import AwesomeTokenizer, ExponentialTokenizer


def load_cfg(checkpoint: dict) -> DictConfig:
    train_config = checkpoint["config"]
    return OmegaConf.create(train_config)


def load_tokenizer(cfg: DictConfig):
    tokenizer_parameters = OmegaConf.to_container(cfg.tokenizer.tokenizer_parameters)
    tokenizer_parameters |= {"special_tokens": special_tokens}

    if cfg.tokenizer.tokenizer == "AwesomeMidiTokenizer":
        min_time_unit = tokenizer_parameters["min_time_unit"]
        n_velocity_bins = tokenizer_parameters["min_velocity_bins"]
        tokenizer_path = to_absolute_path(
            f"pretrained/awesome_tokenizers/awesome-tokenizer-{min_time_unit}-{n_velocity_bins}.json"
        )
        return AwesomeTokenizer.from_file(tokenizer_path)
    else:
        return ExponentialTokenizer(**tokenizer_parameters)


def initialize_model(
    cfg: DictConfig,
    checkpoint: dict,
    device: torch.device,
    pad_token_id: int = 0,
) -> GPT:
    """
    Initializes the GPT model using the given configurations and checkpoint.

    Parameters:
        cfg (DictConfig): The configuration object.
        dataset_config (dict): The dataset configuration.
        checkpoint (dict): The model checkpoint.
        device (torch.device): The device to load the model on.

    Returns:
        GPT: The initialized GPT model.
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

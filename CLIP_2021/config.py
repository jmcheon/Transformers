from pathlib import Path


def get_base_config():
    return {
        "batch_size": 64,
        "dropout": 0.1,
        "num_epochs": 1,
        "lr": 1e-4,
        "model_folder": "/weights",
        "model_basename": "clip_model_",
        "preload": None,
        "experiment_name": "/runs/clip_model",
    }


def get_vit_config():
    return {
        "img_size": 224,
        "patch_size": 4,
        "in_channels": 3,
        "d_model": 96,
        "d_ff": 2048,
        "num_heads": 8,
        "num_layers": 12,
    }


def get_t_encoder_config():
    return {
        "seq_len": 77,
        "vocab_size": 30000,
        "lang_src": "en",
        "tokenizer_file": "tokenizer_{0}.json",
    }


def get_config():
    dct = get_base_config()
    dct.update(get_vit_config())
    dct.update(get_t_encoder_config())

    return dct


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)

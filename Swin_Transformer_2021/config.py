from pathlib import Path


def get_base_config():
    return {
        "batch_size": 64,
        "dropout": 0.1,
        "num_epochs": 1,
        "lr": 1e-4,
        "model_folder": "/weights",
        "model_basename": "swin_model_",
        "preload": None,
        "experiment_name": "/runs/swin_model",
    }


def get_hf_swin_tiny_config():
    return {
        "img_size": 224,
        "patch_size": 4,
        "in_channels": 3,
        "window_size": 7,
        "d_model": 96,
        "num_heads": [3, 6, 12, 24],
        "depths": [2, 2, 6, 2],
        "num_classes": 1000,  # ImageNet
    }


def get_hf_swin_small_config():
    return {
        "img_size": 224,
        "patch_size": 4,
        "in_channels": 3,
        "window_size": 7,
        "d_model": 128,
        "num_heads": [4, 8, 16, 32],
        "depths": [2, 2, 18, 2],
        "num_classes": 1000,
    }

def get_config():
    dct = get_base_config()
    dct.update(get_hf_swin_tiny_config())

    return dct

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)

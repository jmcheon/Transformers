from pathlib import Path


def get_config():
    return {
        "batch_size": 16,
        "img_size" : 224,
        "patch_size" : 16,
        "in_channels" : 3,
        "d_model" : 768,
        "d_ff" : 3072,
        "dropout" : 0.1,
        "num_epochs": 1,
        "lr": 10**-4,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)

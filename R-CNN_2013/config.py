from pathlib import Path


def get_config():
    return {
        "num_classes": 21,
        "batch_size": 64,
        "num_epochs": 20,
        "lr": 1e-3,
        "model_folder": "/weights",
        "model_basename": "rcnn_model_",
        "preload": None,
        "experiment_name": "/runs/rcnn_model",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)

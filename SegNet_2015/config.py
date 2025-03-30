from pathlib import Path


def get_config():
    return {
        "batch_size": 64,
        "img_size" : 224,
        "num_epochs": 70,
        "lr": 1e-2,
        "model_folder": "/weights",
        "model_basename": "segnet3_model_",
        "preload": "59",
        "experiment_name": "/runs/segnet3_model",
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
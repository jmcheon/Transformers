from pathlib import Path


def get_config():
    return {
        "grid_size": 7,
        "num_boxes": 2,
        "num_classes": 20,
        "batch_size": 64,
        "num_epochs": 70,
        "lr": 1e-2,
        "model_folder": "/weights",
        "model_basename": "yolov1_model_",
        "preload": None,
        "experiment_name": "/runs/yolov1_model",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)

from pathlib import Path


def get_bert_config():
    return {
        "batch_size": 16,
        "num_epochs": 10,
        "lr": 1e-3,
        "max_len": 512,
        "hidden_size": 768,
        # "vocab_size":,
        "num_heads": 12,
        "num_layers": 12,
        "d_ff": 3072,
        "model_folder": "weights",
        "model_basename": "bert_model_",
        "preload": None,
        "experiment_name": "runs/bert_model",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)

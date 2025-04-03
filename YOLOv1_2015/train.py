from pathlib import Path

import torch
import torch.optim as optim
from config import get_weights_file_path
from model import YOLO, YOLOLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def get_model(config):
    model = YOLO(config["grid_size"], config["num_boxes"], config["num_classes"])
    return model


def train_yolo(model, config, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # dataset
    train_loader = DataLoader(dataset, config["batch_size"], shuffle=True)

    # model
    model.to(device)

    writer = SummaryWriter(config["experiment_name"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    loss_fn = YOLOLoss(
        grid_size=config["grid_size"],
        num_boxes=config["num_boxes"],
        num_classes=config["num_classes"],
    ).to(device)

    initial_epoch = 0
    global_step = 0

    # preload
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model: {model_filename}")

        state = torch.load(model_filename, weights_only=False)

        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])

        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]

    # training
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{config['num_epochs']}]")
        for images, targets in batch_iterator:
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, targets)

            # Log
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            # backward
            loss.backward()
            if torch.isnan(loss):
                print("NaN detected! Skipping step.")
                continue

            # update
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

            batch_iterator.set_postfix(loss=f"{loss.item():6.3f}")
        # scheduler.step()
        # Save model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )
    writer.close()

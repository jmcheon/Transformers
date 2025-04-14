from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from config import get_weights_file_path
from model import RCNN
from ResNet_2015.ResNet import ResNet18
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def get_rcnn(config):
    resnet18 = ResNet18()
    rcnn = RCNN(backbone=resnet18, num_classes=config["num_classes"])

    return rcnn


def train_rcnn(model, config, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # dataset
    train_loader = DataLoader(dataset, config["batch_size"], shuffle=True)

    model.to(device)
    writer = SummaryWriter(config["experiment_name"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    cls_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.SmoothL1Loss()

    initial_epoch = 0
    global_step = 0

    # preload
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model: {model_filename}")

        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{config['num_epochs']}]")

        for images, (labels, bbox_targets) in batch_iterator:
            images = images.to(device)  # (B, 3, 224, 224)
            labels = labels.to(device)  # (B,) class indices
            bbox_targets = bbox_targets.to(device)  # (B, 4)

            class_logits, bbox_deltas = model(images)

            # Compute losses
            loss_cls = cls_loss_fn(class_logits, labels)
            loss_bbox = bbox_loss_fn(bbox_deltas, bbox_targets)
            loss = loss_cls + config.get("lambda_bbox", 1.0) * loss_bbox

            # Logging
            writer.add_scalar("loss/total", loss.item(), global_step)
            writer.add_scalar("loss/class", loss_cls.item(), global_step)
            writer.add_scalar("loss/bbox", loss_bbox.item(), global_step)
            writer.flush()

            loss.backward()
            if torch.isnan(loss):
                print("NaN detected! Skipping step.")
                continue

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            batch_iterator.set_postfix(loss=f"{loss.item():6.3f}")

        # scheduler.step()
        # Save checkpoint
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

from pathlib import Path

import torch
import torch.nn as nn
from config import get_config, get_weights_file_path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from tqdm import tqdm
from Vision_Transformer import build_vision_transformer


def get_dataset(config):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    train_dataset = CIFAR10(root="./data", train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def get_model(config, num_classes):
    model = build_vision_transformer(
        config["img_size"],
        config["patch_size"],
        config["in_channels"],
        num_classes,
        config["d_model"],
        config["d_ff"],
        config["num_heads"],
        config["num_layers"],
    )
    return model


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # dataset
    train_loader, test_loader = get_dataset(config)

    # model
    model = get_model(config, 10)
    model = model.to(device)

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    loss_fn = nn.CrossEntropyLoss().to(device)

    initial_epoch = 0
    global_step = 0

    # preload
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading model: {model_filename}")

        state = torch.load(model_filename, weights_only=False)

        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    # training
    for epoch in range(initial_epoch, config["num_epochs"]):
        total_loss = 0
        correct = 0
        total = 0

        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{config['num_epochs']}]")
        for images, labels in batch_iterator:
            images, labels = images.to(device), labels.to(device)

            outputs = model.forward(images)
            loss = loss_fn(outputs, labels)

            # Log
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            # Backpropagation
            loss.backward()

            # Update
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            batch_iterator.set_postfix(
                loss=f"{loss.item():6.3f}", accuracy=f"{100 * correct / total:.2f}"
            )
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(
            f"Epoch [{epoch:02d}/{config["num_epochs"]}] loss: {avg_loss:.4f}, accuracy: {accuracy:.2f}"
        )

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


def get_model_state(config):
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        state = torch.load(model_filename, weights_only=False)

    return state


def evaluate_model(model, test_loader, device):
    """
    Evaluates the model on the test dataset.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)

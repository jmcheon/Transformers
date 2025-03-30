import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from config import get_weights_file_path
from FCN_2014.dataset import SegmentationDatset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# pixel labels in the video frames
class_names = [
    "sky",
    "building",
    "column/pole",
    "road",
    "side walk",
    "vegetation",
    "traffic light",
    "fence",
    "vehicle",
    "pedestrian",
    "byciclist",
    "void",
]


def get_dataset_slice_paths(image_dir, label_map_dir):
    """
    Generate the list of image and label map paths

    Args:
        image_dir (str): path to the input image directory
        label_map_dir (str): path to the label map directory

    Returns:
        image_paths (List[str]): paths to each image file
        label_map_paths (List[str]): paths to each label map file
    """

    image_file_lst = os.listdir(image_dir)
    label_map_file_lst = os.listdir(label_map_dir)

    image_paths = [os.path.join(image_dir, fname) for fname in image_file_lst]
    label_map_paths = [os.path.join(label_map_dir, fname) for fname in label_map_file_lst]

    return image_paths, label_map_paths


def get_dataset(config):
    # get the paths to the images
    train_image_paths, train_label_map_paths = get_dataset_slice_paths(
        "/tmp/fcnn/dataset1/images_prepped_train/", "/tmp/fcnn/dataset1/annotations_prepped_train/"
    )
    test_image_paths, test_label_map_paths = get_dataset_slice_paths(
        "/tmp/fcnn/dataset1/images_prepped_test/", "/tmp/fcnn/dataset1/annotations_prepped_test/"
    )

    train_dataset = SegmentationDatset(
        train_image_paths,
        train_label_map_paths,
        class_names,
        height=config["img_size"],
        width=config["img_size"],
    )
    test_dataset = SegmentationDatset(
        test_image_paths,
        test_label_map_paths,
        class_names,
        height=config["img_size"],
        width=config["img_size"],
    )

    # generate the train and test sets
    train_loader = DataLoader(train_dataset, config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, config["batch_size"], shuffle=False)

    return train_loader, test_loader


def train_model(model, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # dataset
    train_loader, test_loader = get_dataset(config)

    # model
    model.to(device)

    writer = SummaryWriter(config["experiment_name"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    # optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)

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
        total_loss = 0
        correct = 0
        total = 0

        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{config['num_epochs']}]")
        for images, labels in batch_iterator:
            images, labels = images.to(device), labels.to(device)

            outputs = model.forward(images)
            loss = criterion(outputs, labels)

            # Log
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            # backward
            loss.backward()

            # update
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            _, targets = labels.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.numel()

            batch_iterator.set_postfix(
                loss=f"{loss.item():6.3f}", accuracy=f"{100 * correct / total:.2f}"
            )
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

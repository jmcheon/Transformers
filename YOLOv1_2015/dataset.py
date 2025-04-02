import os
import xml.etree.ElementTree as ET

import PIL.Image as Image
import torch
import torch.nn as nn
from utils import compute_iou_vectorized

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class VOCDataset(nn.Module):
    def __init__(
        self,
        img_dir: str,
        annotation_dir: str,
        grid_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 20,
        transform=None,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.img_filenames = [f[:-4] for f in os.listdir(annotation_dir) if f.endswith(".xml")]

        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        print(idx)
        filename = self.img_filenames[idx]
        print(filename)
        img_path = os.path.join(self.img_dir, filename + ".jpg")
        annotation_path = os.path.join(self.annotation_dir, filename + ".xml")

        image = Image.open(img_path).convert("RGB")
        boxes, labels = self.parse_annotation(annotation_path)

        if self.transform:
            image = self.transform(image)

        target = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        for box, label in zip(boxes, labels):
            x_center, y_center, w, h = box

            # extract cell-level position
            i = int(y_center * self.S)  # row index of the grid cell
            j = int(x_center * self.S)  # column index of the grid cell

            # x, y coordinates relative to the cell
            x_cell = x_center * self.S - j
            y_cell = y_center * self.S - i

            # ground truth box for IoU comparison
            box_tensor = torch.tensor([x_cell, y_cell, w, h])

            # choose the best box of B to assign this object
            pred_boxes = torch.stack([target[i, j, b * 5 : b * 5 + 4] for b in range(self.B)])

            # compute iou b/w all the predicted boxes and one ground truth box
            ious = compute_iou_vectorized(pred_boxes, box_tensor)
            best_box_idx = torch.argmax(ious).item()

            box_offset = best_box_idx * 5
            target[i, j, box_offset + 0 : box_offset + 4] = box_tensor
            target[i, j, box_offset + 4] = 1.0  # confidence
            target[i, j, self.B * 5 + label] = 1  # one-hot class assignement

        return image, target

    def parse_annotation(self, annotation_path):
        """
        Perse Object's bounding box and it's label(class) in xml format

        Returns:
            boxes (List): normalized bounding box values
            labels (List): class indices corresponding to bounding box's
        """
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = []
        labels = []

        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        for obj in root.findall("object"):
            class_name = obj.find("name").text.lower().strip()
            class_idx = VOC_CLASSES.index(class_name)

            bbox = obj.find("bndbox")
            x_min = int(bbox.find("xmin").text)
            y_min = int(bbox.find("ymin").text)
            x_max = int(bbox.find("xmax").text)
            y_max = int(bbox.find("ymax").text)

            # normalize coordinates [0, 1]
            x_center = ((x_max + x_min) / 2) / img_width
            y_center = ((y_max + y_min) / 2) / img_height
            box_width = (x_max - x_min) / img_width
            box_height = (y_max - y_min) / img_height

            boxes.append([x_center, y_center, box_width, box_height])
            labels.append(class_idx)

        return boxes, labels

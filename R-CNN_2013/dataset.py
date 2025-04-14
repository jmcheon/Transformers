import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import compute_iou_vectorized_xyxy
from YOLOv1_2015.dataset import VOCDataset


class RCNNDataset(Dataset):
    def __init__(
        self, voc_dataset: VOCDataset, transform: None, iou_threshold=0.5, max_proposals=200
    ):
        super().__init__()
        self.voc_dataset = voc_dataset
        self.tranform = transform
        self.iou_threshold = iou_threshold
        self.max_proposals = max_proposals

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx):
        image, _ = self.voc_dataset[idx]
        # boxes denormalized (x_min, y_min, x_max, y_max)
        boxes, labels = self.voc_dataset.parse_annotation(
            self.voc_dataset.annotation_path[idx], box_format="xyxy", normalize=False
        )

        # get region proposals
        img_np = np.array(image)
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(img_np)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()[: self.max_proposals]

        proposals = []
        class_labels = []
        bbox_targets = []

        for x, y, w, h in rects:
            proposal = [x, y, x + w, y + h]
            ious = compute_iou_vectorized_xyxy(
                torch.tensor([proposal]), torch.tensor(boxes, dtype=torch.float32)
            )
            best_idx = torch.argmax(ious)
            best_iou = ious[0, best_idx]

            if best_iou > self.iou_threshold:
                label = labels[best_idx]
                gt_box = boxes[best_idx]
            else:
                label = 20  # background
                gt_box = proposal

            # crop region
            region = image.crop([x, y, x + w, y + h])

            if self.tranform:
                image = self.tranform(image)

            # TODO: normalize bbox
            bbox_targets.append(gt_box)
            class_labels.append(label)
            proposals.append(region)

        # (batch, 3, 224, 224), (batch, 4), (batch,)
        return torch.stack(proposals), torch.tensor(class_labels), torch.stack(bbox_targets)

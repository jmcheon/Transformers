import numpy as np
import torch


def compute_iou(box1, box2, smoothing_factor=1e-6):
    """
    Compute IoU between two boxes

    Args:
        box1: normalized tensor/list of (x, y, w, h)
        box2: normalized tensor/list of (x, y, w, h)
    """
    box1 = torch.tensor(box1, dtype=torch.float32)
    box2 = torch.tensor(box2, dtype=torch.float32)

    b1x1 = box1[0] - box1[2] / 2
    b1y1 = box1[1] - box1[3] / 2
    b1x2 = box1[0] + box1[2] / 2
    b1y2 = box1[1] + box1[3] / 2

    b2x1 = box2[0] - box2[2] / 2
    b2y1 = box2[1] - box2[3] / 2
    b2x2 = box2[0] + box2[2] / 2
    b2y2 = box2[1] + box2[3] / 2

    # intersection coordinates
    inter_x1 = max(b1x1, b2x1)
    inter_y1 = max(b1y1, b2y1)
    inter_x2 = min(b1x2, b2x2)
    inter_y2 = min(b1y2, b2y2)

    # compute intersection area
    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)

    # compute union area
    b1_area = (b1x2 - b1x1) * (b1y2 - b1y1)
    b2_area = (b2x2 - b2x1) * (b2y2 - b2y1)
    union_area = b1_area + b2_area - inter_area + smoothing_factor

    iou = inter_area / union_area
    return iou.item()


def compute_iou_vectorized(pred_boxes, gt_box, smoothing_factor=1e-6):
    """
    Compute IoU b/w several predicted boxes and one ground truth box

    Args:
        pred_boxes (tensor): (B, 4) for each row (x, y, w, h)
        gt_box (tensor): (x, y, w, h)
    """
    pred_boxes = pred_boxes.clone()
    gt_box = gt_box.clone()

    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    gt_x1 = gt_box[0] - gt_box[2] / 2
    gt_y1 = gt_box[1] - gt_box[3] / 2
    gt_x2 = gt_box[0] + gt_box[2] / 2
    gt_y2 = gt_box[1] + gt_box[3] / 2

    # intersection coordinates
    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)

    # compute intersection area
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # compute union area
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union_area = pred_area + gt_area - inter_area + smoothing_factor

    return inter_area / union_area


def decode_predictions(output, S=7, B=2, C=20, conf_threshold=0.3):
    """
    Convert YOLOv1 output (S, S, B * 5 + C) to list of boxes

    Each box is (x1, y1, x2, y2, class_id, confidence)
    """
    boxes = []
    output = output.detach().cpu().numpy()

    for i in range(S):
        for j in range(S):
            cell = output[i, j]
            class_probs = cell[B * 5 :]
            class_id = np.argmax(class_probs)
            class_score = class_probs[class_id]

            for b in range(B):
                x, y, w, h, conf = cell[b * 5 : (b + 1) * 5]
                confidence = conf * class_score

                if confidence < conf_threshold:
                    continue

                # convert to absolut image coordinates
                x_abs = (j + x) / S
                y_abs = (i + y) / S

                x1 = x_abs - w / 2
                y1 = y_abs - h / 2
                x2 = x_abs + w / 2
                y2 = y_abs + h / 2

                boxes.append([x1, y1, x2, y2, class_id, confidence])

    return boxes

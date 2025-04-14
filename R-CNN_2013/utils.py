import torch


def compute_iou_vectorized_xyxy(boxes1, boxes2, eps=1e-6):
    """
    Compute IoU between two sets of bounding boxes in xyxy format.
    Args:
        boxes1 (Tensor[N, 4]): proposals (x1, y1, x2, y2)
        boxes2 (Tensor[M, 4]): ground truth boxes (x1, y1, x2, y2)
        eps (float): Small value to avoid division by zero.
    """

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    # left-top and right-bottom corners of the intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    # width and height of the intersection
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2 - inter
    iou = inter / (union + eps)
    return iou  # (N, M)

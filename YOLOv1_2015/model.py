import torch
import torch.nn as nn


class YOLO(nn.Module):
    def __init__(self, grid_size: int = 7, num_boxes: int = 2, num_classes: int = 20):
        super().__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes

        # feature extractor
        self.features = nn.Sequential(
            self._conv_block(3, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            self._conv_block(64, 192, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            self._conv_block(192, 128, 1),
            self._conv_block(128, 256, 3, padding=1),
            self._conv_block(256, 256, 1),
            self._conv_block(256, 512, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (56 -> 28)
            # 4x conv blocks
            *[
                block
                for _ in range(4)
                for block in [
                    self._conv_block(512, 256, 1),
                    self._conv_block(256, 512, 3, padding=1),
                ]
            ],
            self._conv_block(512, 512, 1),
            self._conv_block(512, 1024, 3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (28 -> 14)
            #
            self._conv_block(1024, 512, 1),
            self._conv_block(512, 1024, 3, padding=1),
            self._conv_block(1024, 512, 1),
            self._conv_block(512, 1024, 3, padding=1),
            self._conv_block(1024, 1024, 3, padding=1),
            self._conv_block(1024, 1024, 3, stride=2, padding=1),  # (14 -> 7)
            #
            self._conv_block(1024, 1024, 3, padding=1),
            self._conv_block(1024, 1024, 3, padding=1),
        )

        # fully-connected layers after flatten
        self.fc = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(512 * 28 * 28, 4096),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C)),
        )

    def _conv_block(self, in_c, out_c, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)

        return x.view(-1, self.S, self.S, self.B * 5 + self.C)


class YOLOLoss(nn.Module):
    def __init__(
        self,
        grid_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 20,
        lambda_coord: int = 5,
        lambda_noobj: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.eps = eps

    def forward(self, predictions, target):
        """
        predictions: (N, S, S, 5B + C)
        target : (N, S, S, 5B + C)
        """
        N = predictions.size(0)

        pred_boxes = predictions[..., : self.B * 5].view(N, self.S, self.S, self.B, 5)
        pred_classes = predictions[..., self.B * 5 :]  # (N, S, S, C)

        target_boxes = target[..., : self.B * 5].view(N, self.S, self.S, self.B, 5)
        target_classes = target[..., self.B * 5 :]  # (N, S, S, C)

        # mask: extract only confidence value from bounding box
        exists_box = target_boxes[..., 4].unsqueeze(-1)  # confidence (N, S, S, B, 1)

        # localization loss
        box_pred_xy = pred_boxes[..., 0:2]
        box_pred_wh = pred_boxes[..., 2:4]
        box_target_xy = target_boxes[..., 0:2]
        box_target_wh = target_boxes[..., 2:4]

        box_pred_wh = torch.sign(box_pred_wh) * torch.sqrt(torch.abs(box_pred_wh + self.eps))
        box_target_wh = torch.sqrt(box_target_wh + self.eps)

        loc_loss = self.mse(exists_box * box_pred_xy, exists_box * box_target_xy)
        loc_loss += self.mse(exists_box * box_target_wh, exists_box * box_target_wh)

        # confidence loss
        box_pred_conf = pred_boxes[..., 4:5]
        box_target_conf = target_boxes[..., 4:5]

        conf_loss_obj = self.mse(exists_box * box_pred_conf, exists_box * box_target_conf)

        noobj_mask = 1 - exists_box
        conf_loss_noobj = self.mse(noobj_mask * box_pred_conf, noobj_mask * box_target_conf)

        # classification loss
        # class predictions: (N, S, S, C)
        object_mask = (target[..., 4::5].sum(dim=-1, keepdim=True) > 0).float()
        class_loss = self.mse(
            object_mask.expand_as(pred_classes) * pred_classes,
            object_mask.expand_as(target_classes) * target_classes,
        )

        # total loss
        total_loss = (
            self.lambda_coord * loc_loss
            + conf_loss_obj
            + self.lambda_noobj * conf_loss_noobj
            + class_loss
        ) / N

        return total_loss

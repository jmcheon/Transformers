import torch.nn as nn


class RCNN(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 21):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(backbone.output_dim, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )

        self.bbox_regressor = nn.Sequential(
            nn.Linear(backbone.output_dim, 256), nn.ReLU(), nn.Linear(256, 4)
        )

    def forward(self, x):
        features = self.backbone(x)  # (batch, output_dim)
        class_logits = self.classifier(features)  # (batch, num_classes)
        bbox_deltas = self.bbox_regressor(features)  # (batch, 4)

        return class_logits, bbox_deltas

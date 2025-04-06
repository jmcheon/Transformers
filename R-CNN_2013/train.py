from model import RCNN
from ResNet_2015.ResNet import ResNet18


def get_rcnn(config):
    resnet18 = ResNet18()
    rcnn = RCNN(backbone=resnet18, num_classes=config["num_classes"])

    return rcnn

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDatset(Dataset):
    def __init__(self, image_paths, label_map_paths, class_names, height=224, width=224):
        self.image_paths = image_paths
        self.label_map_paths = label_map_paths
        self.height = height
        self.width = width
        self.class_names = class_names

        self.image_tranform = transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.ToTensor(),  # [0, 1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
            ]
        )

        self.mask_transform = transforms.Resize((height, width), interpolation=Image.NEAREST)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # load image
        image = Image.open(self.image_paths[index]).convert("RGB")
        image = self.image_tranform(image)

        # load mask
        mask = Image.open(self.label_map_paths[index]).convert("L")
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask)).long()

        # one-hot encode mask
        mask_one_hot = torch.zeros(
            (len(self.class_names), self.height, self.width), dtype=torch.int32
        )
        for c in range(len(self.class_names)):
            mask_one_hot[c] = mask == c

        return image, mask_one_hot

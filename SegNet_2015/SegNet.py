import torch
from model import SegNet

model = SegNet(11)

def get_model():
    # state_dict = torch.load("segnet_camvid_best_model.pkl", map_location="cpu")
    state_dict = torch.load("segnet_camvid.pth", map_location="cpu")

    model.load_state_dict(state_dict)


if __name__ == "__main__":
    get_model()
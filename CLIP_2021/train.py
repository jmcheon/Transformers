from model import CLIP, CLIPTextEncoder
from ResNet_2015.model import ResNet50
from Vision_Transformer_2020.Vision_Transformer import build_vitmodel


def get_model(config):
    resnet_encoder = ResNet50()
    resnet_encoder.output_dim = 2048

    vit_encoder = build_vitmodel(
        config["img_size"],
        config["patch_size"],
        config["in_channels"],
        config["d_model"],
        config["d_ff"],
        config["num_heads"],
        config["num_layers"],
        config["dropout"],
    )

    transformer_encoder = CLIPTextEncoder(vocab_size=30000, max_len=77)

    clip = CLIP(image_encoder=resnet_encoder, text_encoder=transformer_encoder, embed_dim=512)

    return clip

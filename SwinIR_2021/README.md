# SwinIR: Image Restoration Using Swin Transformer

## Architecture

<div align="center">

  <img alt="SwinIR" src="./assets/SwinIR.png" width=800 height=350/>
  <br/>
  <figcaption>Figure 1: SwinIR Architecture</figcaption>

</div>


# Training

- Dataset: [Sony Low-Light RAW Image Dataset](https://www.kaggle.com/datasets/jungmoo/sid-sony-dataset/versions/1) 
- Encoder: Swin Transformer (feature extractor)
- Decoder: U-net Upsampling

# References

- https://arxiv.org/abs/2108.10257
- [Swin Transformer](https://huggingface.co/docs/transformers/en/model_doc/swin)
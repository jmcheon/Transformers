# Fully Convolutional Networks for Semantic Segmentation

## Architectures

### FCN-32

<div align="center">

  <img alt="FCN-32" src="./assets/FCN-32.png" width=800 height=300/>
  <br/>
  <figcaption>Figure 1: FCN-32 Architecture</figcaption>

</div>

### FCN-16

<div align="center">

  <img alt="FCN-16" src="./assets/FCN-16.png" width=800 height=300/>
  <br/>
  <figcaption>Figure 1: FCN-16 Architecture</figcaption>


</div>

### FCN-8

<div align="center">

  <img alt="FCN-8" src="./assets/FCN-8.png" width=800 height=300/>
  <br/>
  <figcaption>Figure 1: FCN-8 Architecture</figcaption>

</div>

### Comparison of different FCNs

<div align="center">

  <img alt="Comparison FCNs" src="./assets/comparison_FCNs.png" width=800 height=300/>
  <br/>

</div>

# Training

- Dataset: The Cambridge Driving (CamVid) (https://github.com/divamgupta/image-segmentation-keras)

- Encoder: pretrained VGG-16 Conv Layers
- Decoder: FCN-8

# References

- https://arxiv.org/abs/1411.4038
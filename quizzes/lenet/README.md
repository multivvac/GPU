# LeNet

A pure CUDA implementation for classical CNN LeNet.

## Architecture

![the architecture of LeNet](https://upload.wikimedia.org/wikipedia/commons/3/35/LeNet-5_architecture.svg)

## Kernels

### Convolution

```cpp
void conv_forward(size_t N, size_t M, size_t C, size_t H, size_t W, size_t K, float* X, float* F, float* Y);
```

| Name | Type | Desc|
| ------------- | -------------- | -------------- |
| `N` | `size_t` | number of images in 1 batch |
| `M` | `size_t` | number of feature maps per image |
| `C` | `size_t` | number of output feature maps |
| `H` | `size_t` | height of input image |
| `W` | `size_t` | width of input image |
| `K` | `size_t` | size of filter |
| `X` | `float*` | input tensor with size `(N, M, H, W)` |
| `F` | `float*` | filter weight `K x K` |
| `Y` | `float*` | output tensor with size `(N, C, H_o, W_o)` |

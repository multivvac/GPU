#ifndef LENET_KERNEL_H
#define LENET_KERNEL_H

#include <torch/torch.h>
#define THREADS_PER_BLOCK 1024
#define TILE_SIZE 32
#define KERNEL_1_SIZE 6
#define KERNEL_1_IN_CHAN 6
#define KERNEL_1_OUT_CHAN 16
torch::Tensor im2col_cuda(torch::Tensor &input, size_t K);
torch::Tensor im2col_optimized_cuda(torch::Tensor &input, size_t K);
torch::Tensor conv2d_im2col_cuda(torch::Tensor &input, torch::Tensor &filter, size_t K); 

#endif

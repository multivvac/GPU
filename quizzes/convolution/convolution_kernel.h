#ifndef CONVOLUTION_KERNEL_H
#define CONVOLUTION_KERNEL_H

#include <torch/torch.h>
#define THREAD_PER_BLOCK 16
#define FILTER_RADIUS 4
#define FILTER_KERNEL_SIZE (2 * FILTER_RADIUS + 1)
torch::Tensor convolution_naive_cuda(torch::Tensor &data,
                                     torch::Tensor &filter_weight, int radius);

torch::Tensor convolution_constant_mem_cuda(torch::Tensor &data,
                                            torch::Tensor &filter_weight,
                                            int radius);

torch::Tensor convolution_2D_tiled_constant_mem_cuda(
    torch::Tensor &data, torch::Tensor &filter_weight, int radius);
#endif

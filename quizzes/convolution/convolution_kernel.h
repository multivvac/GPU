#ifndef CONVOLUTION_KERNEL_H
#define CONVOLUTION_KERNEL_H

#include <torch/torch.h>
#define THREAD_PER_BLOCK 16
torch::Tensor convolution_naive_cuda(torch::Tensor &data, torch::Tensor &filter_weight, int radius);

#endif

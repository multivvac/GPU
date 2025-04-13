#ifndef LENET_KERNEL_H
#define LENET_KERNEL_H

#include <torch/torch.h>
#define THREADS_PER_BLOCK 128
torch::Tensor im2col_cuda(torch::Tensor &input, size_t K);

#endif

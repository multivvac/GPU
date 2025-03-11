#ifndef MATRIX_TRANSPOSE_KERNEL_H
#define MATRIX_TRANSPOSE_KERNEL_H

#include <torch/torch.h>
#define BLOCKDIM 32
torch::Tensor matrix_transpose_cuda(torch::Tensor &data);

#endif

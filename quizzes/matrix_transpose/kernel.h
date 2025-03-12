#ifndef MATRIX_TRANSPOSE_KERNEL_H
#define MATRIX_TRANSPOSE_KERNEL_H

#include <torch/torch.h>
#define BLOCKDIM 32
#define COARSE 4
torch::Tensor matrix_transpose_cuda(torch::Tensor &data);
torch::Tensor matrix_transpose_coalesced_cuda(torch::Tensor &data);
torch::Tensor matrix_transpose_coalesced_coarse_cuda(torch::Tensor &data);

#endif

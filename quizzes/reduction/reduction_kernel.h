#ifndef REDUCTION_KERNEL_H
#define REDUCTION_KERNEL_H

#include <torch/torch.h>
#define MAX_THREAD_BLOCK 1024
#define THREAD_PER_BLOCK 8
torch::Tensor reduction_naive_cuda(torch::Tensor &data);
torch::Tensor reduction_convergent_cuda(torch::Tensor &data);
torch::Tensor reduction_shared_mem_cuda(torch::Tensor &data);

#endif

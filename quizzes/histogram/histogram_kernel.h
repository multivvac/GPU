#ifndef HISTOGRAM_KERNEL_H
#define HISTOGRAM_KERNEL_H

#include <torch/torch.h>
#define NUM_BINS 256
#define THREAD_PER_BLOCK 512
torch::Tensor histogram_cuda(torch::Tensor &data);
torch::Tensor histogram_coarse_cuda(torch::Tensor &data);
torch::Tensor histogram_coarse_contiguous_cuda(torch::Tensor &data);
torch::Tensor histogram_vec_cuda(torch::Tensor &data);
torch::Tensor histogram_aggregation_cuda(torch::Tensor &data);

#endif

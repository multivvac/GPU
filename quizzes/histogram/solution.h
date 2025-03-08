#ifndef HISTORGRAM_SOLUTION_H
#define HISTORGRAM_SOLUTION_H

#include <torch/torch.h>
#define NUM_BINS 256
#define THREAD_PER_BLOCK 128
torch::Tensor histogram_cuda(torch::Tensor &data);
torch::Tensor histogram_vec_cuda(torch::Tensor &data);

#endif

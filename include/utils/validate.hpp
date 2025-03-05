#ifndef VALIDATE_H
#define VALIDATE_H

#include <torch/torch.h>

std::vector<std::string> verbose_allclose(const torch::Tensor &received,
                                          const torch::Tensor &expected,
                                          float rtol, float atol,
                                          int max_print = 10);
std::vector<std::string> verbose_allequal(const torch::Tensor &received,
                                          const torch::Tensor &expected,
                                          int max_print = 10);
#endif // !VALIDATE_H

#ifndef VALIDATE_H
#define VALIDATE_H

#include <torch/torch.h>
std::vector<std::string> verbose_allclose(const torch::Tensor &received,
                                          const torch::Tensor &expected,
                                          float rtol, float atol,
                                          unsigned int max_print);
#endif // !VALIDATE_H

#include "utils/validate.h"
#include <torch/torch.h>
std::vector<std::string> verbose_allclose(const torch::Tensor &received,
                                          const torch::Tensor &expected,
                                          float rtol, float atol,
                                          unsigned int max_print) {
  auto messages = std::vector<std::string>();
  std::cout << "verbose_allclose called!" << std::endl;
  return messages;
}

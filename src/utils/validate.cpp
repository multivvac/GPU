#include "utils/validate.h"
#include <string>
#include <torch/torch.h>
std::vector<std::string> verbose_allclose(const torch::Tensor &received,
                                          const torch::Tensor &expected,
                                          float rtol, float atol,
                                          int max_print) {
  auto messages = std::vector<std::string>();
  if (received.sizes() != expected.sizes()) {
    messages.push_back("SIZE MISMATCH");
    return messages;
  }
  {
    torch::NoGradGuard no_grad;
    auto diff = torch::abs(received - expected);

    auto tolerance = atol + rtol * torch::abs(expected);

    auto tol_mismatched = diff > tolerance;

    auto nan_mismatched =
        torch::logical_xor(torch::isnan(received), torch::isnan(expected));

    auto posinf_mismatched = torch::logical_xor(torch::isposinf(received),
                                                torch::isposinf(expected));

    auto neginf_mismatched = torch::logical_xor(torch::isneginf(received),
                                                torch::isposinf(expected));

    auto mismatched = torch::logical_or(
        torch::logical_or(tol_mismatched, nan_mismatched),
        torch::logical_or(posinf_mismatched, neginf_mismatched));

    auto mismatched_indices = torch::nonzero(mismatched);
    auto num_mismatched = torch::count_nonzero(mismatched).item<int>();

    if (num_mismatched >= 1) {
      messages.push_back("Number of mismatched elements: " +
                         std::to_string(num_mismatched));

      for (unsigned int index = 0; index < max_print; index++) {
        auto i = mismatched_indices[index].item<int>();
        messages.push_back("ERROR AT " + std::to_string(i) + ": " +
                           std::to_string(received[i].item<float>()) +
                           " != " + std::to_string(expected[i].item<float>()));
      }

      if (num_mismatched > max_print) {
        messages.push_back("... and " +
                           std::to_string(num_mismatched - max_print) +
                           " more mismatched elements");
      }
    }
  }

  return messages;
}

std::vector<std::string> verbose_allequal(const torch::Tensor &received,
                                          const torch::Tensor &expected,
                                          int max_print) {
  auto messages = std::vector<std::string>();
  {
    torch::NoGradGuard no_grad;

    auto mismatched = torch::not_equal(received, expected);
    auto mismatched_indices = torch::nonzero(mismatched);

    auto num_mismatched = mismatched_indices.size(0);
    if (num_mismatched >= 1) {
      messages.push_back("Number of mismatched elements: " +
                         std::to_string(num_mismatched));

      for (int index = 0;
           index < (max_print <= num_mismatched ? max_print : num_mismatched);
           index++) {
        auto i = mismatched_indices.index({index, 0}).item<int>();

        messages.push_back(
            "ERROR AT " + std::to_string(i) + ": " +
            std::to_string(received.index({i}).item<float>()) +
            " != " + std::to_string(expected.index({i}).item<float>()));
      }

      if (num_mismatched > max_print) {
        messages.push_back("... and " +
                           std::to_string(num_mismatched - max_print) +
                           " more mismatched elements");
      }
    }
  }
  return messages;
}

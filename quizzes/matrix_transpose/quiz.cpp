#include "kernel.h"
#include "utils/benchmark.hpp"
#include "utils/validate.hpp"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/core/DeviceType.h>
#include <cstdlib>
#include <iostream>
#include <torch/torch.h>
#include <torch/types.h>

namespace matrix_transpose {
torch::Tensor generate_input(int M, int N, int seed) {

  auto gen = torch::make_generator<at::CUDAGeneratorImpl>(seed);
  auto data = torch::rand({M, N}, gen,
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));

  return data;
}
torch::Tensor solution(torch::Tensor &input) {
  return matrix_transpose_cuda(input);
}
torch::Tensor baseline(torch::Tensor &input) {
  return torch::transpose(input, 0, 1);
}
torch::Tensor coalesed_solution(torch::Tensor &input) {
  return matrix_transpose_coalesed_cuda(input);
}
} // namespace matrix_transpose

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <size> <seed>\n";
  }
  // only consider square.
  int size = std::stoi(argv[1]);
  int M = size;
  int N = size;
  int seed = std::stoi(argv[3]);

  auto input = matrix_transpose::generate_input(M, N, seed);
  auto torch_output = matrix_transpose::baseline(input);

  auto naive_output = matrix_transpose::solution(input);
  auto naive_errors = verbose_allequal(torch_output, naive_output);
  if (!naive_errors.empty()) {
    std::cout << "found errors in naive kernel:\n";
    for (auto &error : naive_errors) {
      std::cout << error << "\n";
    }
  }
  if (!naive_errors.empty()) {
    std::cout << "found errors in naive kernel:\n";
    for (auto &error : naive_errors) {
      std::cout << error << "\n";
    }
  }
  auto coalesed_output = matrix_transpose::coalesed_solution(input);
  auto coalesed_errors = verbose_allequal(torch_output, coalesed_output);
  if (!coalesed_errors.empty()) {
    std::cout << "found errors in coalesed kernel:\n";
    for (auto &error : coalesed_errors) {
      std::cout << error << "\n";
    }
  }

  auto torch_benchmark =
      benchmark([&]() { matrix_transpose::baseline(input); });
  auto naive_benchmark =
      benchmark([&]() { matrix_transpose::solution(input); });
  auto coalesed_benchmark =
      benchmark([&]() { matrix_transpose::coalesed_solution(input); });
  std::cout << "[matrix transpose] Naive implmentation duration: "
            << naive_benchmark << " ns." << std::endl;
  std::cout << "[matrix transpose] Coalesed implmentation "
               "duration: "
            << coalesed_benchmark << " ns." << std::endl;
  std::cout << "[matrix transpose] Pytorch implmentation duration: "
            << torch_benchmark << " ns." << std::endl;
}

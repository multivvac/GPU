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
  // seed won't work without manually setting current seed.
  gen.set_current_seed(seed);
  auto data = torch::rand({M, N}, gen,
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));

  return data.contiguous();
}
torch::Tensor solution(torch::Tensor &input) {
  return matrix_transpose_cuda(input);
}
torch::Tensor baseline(torch::Tensor &input) {
  auto o = torch::transpose(input, 0, 1);
  torch::cuda::synchronize();
  return o.contiguous();
}
torch::Tensor coalesced_solution(torch::Tensor &input) {
  return matrix_transpose_coalesced_cuda(input);
}
torch::Tensor coalesced_coarse_solution(torch::Tensor &input) {
  return matrix_transpose_coalesced_coarse_cuda(input);
}
torch::Tensor coalesced_coarse_bank_conflict_solution(torch::Tensor &input) {
  return matrix_transpose_coalesced_coarse_bank_conflict_cuda(input);
}
} // namespace matrix_transpose

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <size> <seed>\n";
  }
  // only consider square.
  int size = std::stoi(argv[1]);
  int M = size;
  int N = size;
  int seed = std::stoi(argv[2]);

  auto input = matrix_transpose::generate_input(M, N, seed);
  auto torch_output = matrix_transpose::baseline(input);

  auto naive_output = matrix_transpose::solution(input);
  auto naive_errors = verbose_allequal(naive_output, torch_output);
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
  auto coalesced_output = matrix_transpose::coalesced_solution(input);
  auto coalesced_errors = verbose_allequal(coalesced_output, torch_output);
  if (!coalesced_errors.empty()) {
    std::cout << "found errors in coalesced kernel:\n";
    for (auto &error : coalesced_errors) {
      std::cout << error << "\n";
    }
  }
  auto coalesced_coarse_bank_conflict_output =
      matrix_transpose::coalesced_coarse_bank_conflict_solution(input);
  auto coalesced_coarse_bank_conflict_errors =
      verbose_allequal(coalesced_coarse_bank_conflict_output, torch_output);
  if (!coalesced_coarse_bank_conflict_errors.empty()) {
    std::cout << "found errors in coalesced coarse bank conflict kernel:\n";
    for (auto &error : coalesced_coarse_bank_conflict_errors) {
      std::cout << error << "\n";
    }
  }

  auto coalesced_coarse_output =
      matrix_transpose::coalesced_coarse_solution(input);
  auto coalesced_coarse_errors =
      verbose_allequal(coalesced_coarse_output, torch_output);
  if (!coalesced_coarse_errors.empty()) {
    std::cout << "found errors in coalesced coarse kernel:\n";
    for (auto &error : coalesced_coarse_errors) {
      std::cout << error << "\n";
    }
  }
  auto torch_benchmark =
      benchmark([&]() { matrix_transpose::baseline(input); });
  auto naive_benchmark =
      benchmark([&]() { matrix_transpose::solution(input); });
  auto coalesced_benchmark =
      benchmark([&]() { matrix_transpose::coalesced_solution(input); });
  auto coalesced_coarse_bank_conflict_benchmark = benchmark([&]() {
    matrix_transpose::coalesced_coarse_bank_conflict_solution(input);
  });
  auto coalesced_coarse_benchmark =
      benchmark([&]() { matrix_transpose::coalesced_coarse_solution(input); });
  std::cout << "[matrix transpose] Naive implmentation duration: "
            << naive_benchmark << " ns." << std::endl;
  std::cout << "[matrix transpose] coalesced implmentation "
               "duration: "
            << coalesced_benchmark << " ns." << std::endl;
  std::cout
      << "[matrix transpose] coalesced coarse bank conflict implmentation "
         "duration: "
      << coalesced_coarse_bank_conflict_benchmark << " ns." << std::endl;
  std::cout << "[matrix transpose] coalesced coarse implmentation "
               "duration: "
            << coalesced_coarse_benchmark << " ns." << std::endl;
  std::cout << "[matrix transpose] Pytorch implmentation duration: "
            << torch_benchmark << " ns." << std::endl;
}

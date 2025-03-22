#include "convolution_kernel.h"
#include "utils/benchmark.hpp"
#include "utils/validate.hpp"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/core/DeviceType.h>
#include <iostream>
#include <torch/cuda.h>
#include <torch/torch.h>
#include <torch/types.h>
namespace F = torch::nn::functional;

namespace convolution {
torch::Tensor generate_input(int size, int seed) {

  auto gen = torch::make_generator<at::CUDAGeneratorImpl>(seed);
  auto data = torch::randn(
      {1, size, size}, gen,
      torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
  return data.contiguous();
}

torch::Tensor generate_filter_weight(int radius, int seed) {

  auto gen = torch::make_generator<at::CUDAGeneratorImpl>(seed);
  auto filter = torch::randn(
      {1, 1, radius * 2 + 1, radius * 2 + 1}, gen,
      torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
  return filter.contiguous();
}
torch::Tensor baseline(torch::Tensor &data, torch::Tensor &filter_weight,
                       int radius) {
  torch::NoGradGuard no_grad;

  auto output = F::conv2d(data, filter_weight,
                          F::Conv2dFuncOptions().stride(1).padding(radius));
  torch::cuda::synchronize();
  return output;
}

torch::Tensor solution(torch::Tensor &data, torch::Tensor &filter_weight,
                       int radius) {
  return convolution_naive_cuda(data, filter_weight, radius);
}
} // namespace convolution

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <size> <filter radius> <seed>\n";
  }
  int size = std::stoi(argv[1]);
  int radius = std::stoi(argv[2]);
  int seed = std::stoi(argv[3]);

  auto input = convolution::generate_input(size, seed);
  auto filter_weight = convolution::generate_filter_weight(radius, seed);
  auto output = convolution::baseline(input, filter_weight, radius).flatten();
  auto naive_output =
      convolution::solution(input, filter_weight, radius).flatten();

  auto errors = verbose_allclose(naive_output, output, 1e-5, 1e-5);

  if (errors.size() > 0) {
    std::cout << "found errors in basic convolution kernel:\n";
    for (auto &error : errors) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }

  // benchmark
  auto baselinetime =
      benchmark([&]() { convolution::baseline(input, filter_weight, radius); });
  auto selftime =
      benchmark([&]() { convolution::solution(input, filter_weight, radius); });

  std::cout << "[convolution] Naive implmentation duration: " << selftime
            << " ns." << std::endl;
  std::cout << "[convolution] Pytorch implmentation duration: " << baselinetime
            << " ns." << std::endl;
}

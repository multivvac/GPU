#include "stencil_kernel.h"
#include "utils/benchmark.hpp"
#include "utils/stat.hpp"
#include "utils/validate.hpp"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <iostream>
#include <torch/torch.h>
namespace F = torch::nn::functional;

namespace stencil {
torch::Tensor generate_input(int size, int seed) {

  auto gen = torch::make_generator<at::CUDAGeneratorImpl>(seed);
  // seed won't work without manually setting current seed.
  gen.set_current_seed(seed);
  auto data = torch::randn(
      {1, size, size, size}, gen,
      torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
  return data.contiguous();
}

torch::Tensor generate_coefficient(int seed) {

  auto gen = torch::make_generator<at::CUDAGeneratorImpl>(seed);
  auto coefficient = torch::zeros(
      {1, 1, C_KERNEL_SIZE, C_KERNEL_SIZE, C_KERNEL_SIZE},
      torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));

  coefficient.index({0, 0, C_CENTER, C_CENTER, C_CENTER}) = 2.0;
  coefficient.index({0, 0, C_CENTER - 1, C_CENTER, C_CENTER}) = 0.0;
  coefficient.index({0, 0, C_CENTER + 1, C_CENTER, C_CENTER}) = 0.0;
  coefficient.index({0, 0, C_CENTER, C_CENTER - 1, C_CENTER}) = 0.0;
  coefficient.index({0, 0, C_CENTER, C_CENTER + 1, C_CENTER}) = 0.0;
  coefficient.index({0, 0, C_CENTER, C_CENTER, C_CENTER - 1}) = 0.0;
  coefficient.index({0, 0, C_CENTER, C_CENTER, C_CENTER + 1}) = 0.0;

  return coefficient.contiguous();
}
torch::Tensor baseline(torch::Tensor &data, torch::Tensor &coefficient) {
  torch::NoGradGuard no_grad;

  auto output =
      F::conv3d(data, coefficient, F::Conv3dFuncOptions().stride(1).padding(0));
  torch::cuda::synchronize();
  output = F::pad(output,
                  F::PadFuncOptions({ORDER, ORDER, ORDER, ORDER, ORDER, ORDER})
                      .mode(torch::kConstant));
  return output;
}

torch::Tensor solution(torch::Tensor &data, torch::Tensor &coefficient) {
  return stencil_naive_cuda(data, coefficient);
}
torch::Tensor shared_mem_tiling_solution(torch::Tensor &data,
                                         torch::Tensor &coefficient) {
  return stencil_shared_mem_tiling_cuda(data, coefficient);
}
} // namespace stencil

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <size> <seed>\n";
  }
  int size = std::stoi(argv[1]);
  int seed = std::stoi(argv[2]);

  auto input = stencil::generate_input(size, seed);
  auto coefficient = stencil::generate_coefficient(seed);
  auto output = stencil::baseline(input, coefficient).flatten();
  auto naive_output = stencil::solution(input, coefficient).flatten();
  auto shared_mem_tiling_output =
      stencil::shared_mem_tiling_solution(input, coefficient).flatten();

  auto errors = verbose_allclose(naive_output, output, 1e-5, 1e-5);

  if (errors.size() > 0) {
    std::cout << "found errors in basic stencil kernel:\n";
    for (auto &error : errors) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }

  auto shared_mem_tiling_errors =
      verbose_allclose(shared_mem_tiling_output, output, 1e-5, 1e-5);
  if (shared_mem_tiling_errors.size() > 0) {
    std::cout << "found errors in shared memory tiling kernel:\n";
    for (auto &error : shared_mem_tiling_errors) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }
  // benchmark
  auto baselinetime =
      benchmark([&]() { stencil::baseline(input, coefficient); });
  auto selftime = benchmark([&]() { stencil::solution(input, coefficient); });
  auto shared_mem_tiling_time = benchmark(
      [&]() { stencil::shared_mem_tiling_solution(input, coefficient); });

  print_table(
      std::vector<FunctionTiming>{
          FunctionTiming{std::string("Naive Implementation(baseline)"),
                         selftime, baselinetime / selftime},
          FunctionTiming{std::string("Shared Memory Tiling Implementation"),
                         shared_mem_tiling_time,
                         baselinetime / shared_mem_tiling_time},
          FunctionTiming{std::string("Pytorch Implementation(hack by conv3d, "
                                     "to check correctness)"),
                         baselinetime, baselinetime / baselinetime}},
      "Stencil Kernel");
}

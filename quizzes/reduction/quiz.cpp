#include "reduction_kernel.h"
#include "utils/benchmark.hpp"
#include "utils/stat.hpp"
#include "utils/validate.hpp"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <iostream>
#include <torch/torch.h>

namespace reduction {
torch::Tensor generate_input(int size, int seed) {

  auto gen = torch::make_generator<at::CUDAGeneratorImpl>(seed);
  // seed won't work without manually setting current seed.
  gen.set_current_seed(seed);
  auto data = torch::randint(
      10, {size}, gen,
      torch::dtype(torch::kInt64).device(torch::kCUDA).requires_grad(false));
  return data.contiguous();
}

torch::Tensor baseline(torch::Tensor &data) {
  torch::NoGradGuard no_grad;

  auto output = data.sum().unsqueeze(0);
  torch::cuda::synchronize();
  return output;
}

torch::Tensor solution(torch::Tensor &data) {
  auto temp = data.clone();
  return reduction_naive_cuda(temp);
}
} // namespace reduction

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <size> <seed>\n";
  }
  int size = std::stoi(argv[1]);
  int seed = std::stoi(argv[2]);

  auto input = reduction::generate_input(size, seed);
  auto output = reduction::baseline(input);
  auto naive_output = reduction::solution(input);

  auto errors = verbose_allequal(naive_output, output);

  if (errors.size() > 0) {
    std::cout << "found errors in basic reduction kernel:\n";
    for (auto &error : errors) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }

  // benchmark
  auto baselinetime = benchmark([&]() { reduction::baseline(input); });
  auto naivetime = benchmark([&]() { reduction::solution(input); });

  print_table(
      std::vector<FunctionTiming>{
          FunctionTiming{std::string("Naive Implementation"), naivetime,
                         baselinetime / naivetime},
          FunctionTiming{std::string("Pytorch Implementation(baseline)"),
                         baselinetime, baselinetime / baselinetime}},
      "Reduction Kernel");
}

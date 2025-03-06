#include "solution.h"
#include "utils/benchmark.hpp"
#include "utils/validate.hpp"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cstdlib>
#include <iostream>
#include <torch/torch.h>
#include <torch/types.h>

namespace histogram {
torch::Tensor generate_input(int size, float contention, int seed) {

  auto gen = torch::make_generator<at::CUDAGeneratorImpl>(seed);
  auto data = torch::randint(0, NUM_BINS, {size}, gen,
                             torch::dtype(torch::kUInt8).device(torch::kCUDA));
  auto evilvalue = torch::randint(
      0, NUM_BINS, {}, gen, torch::dtype(torch::kUInt8).device(torch::kCUDA));
  auto evilloc =
      torch::rand({size}, gen,
                  torch::dtype(torch::kFloat32).device(torch::kCUDA)) <
      (contention / 100.0);
  data.masked_fill_(evilloc, evilvalue);
  return data.contiguous();
}

torch::Tensor baseline(torch::Tensor &data) {
  return torch::bincount(data, {}, NUM_BINS).to(torch::kUInt32);
}

torch::Tensor solution(torch::Tensor &data) { return histogram_cuda(data); }
} // namespace histogram

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <size> <contention> <seed>\n";
  }
  int size = std::stoi(argv[1]);
  int contention = std::stoi(argv[2]);
  int seed = std::stoi(argv[3]);

  auto input = histogram::generate_input(size, contention, seed);
  auto output = histogram::baseline(input);
  auto my_output = histogram::solution(input);

  auto errors = verbose_allequal(my_output, input);

  if (errors.size() > 0) {
    for (auto &error : errors) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }
  // benchmark

  auto selftime = benchmark([&]() { histogram::solution(input); });
  auto baselinetime = benchmark([&]() { histogram::baseline(input); });

  std::cout << "[histogram]self implmentation duration: " << selftime << " us."
            << std::endl;
  std::cout << "[histogram]baseline duration: " << baselinetime << " us."
            << std::endl;
}

#include "solution.h"
#include "utils/timer.hpp"
#include "utils/validate.h"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <iostream>
#include <torch/torch.h>

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
  return torch::bincount(data, {}, NUM_BINS);
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

  auto timer = StopWatch<chrono_alias::us>();
  timer.start();
  auto my_output = histogram::solution(input);
  timer.stop();
  std::cout << "[histogram]self implmentation duration: "
            << timer.getTime().count() << " us." << std::endl;

  timer.reset();

  timer.start();
  auto output = histogram::baseline(input);
  timer.stop();

  std::cout << "[histogram]pytorch duration: " << timer.getTime().count()
            << " us." << std::endl;
  verbose_allclose(input, input, 0.0, 0.0, 1);
}

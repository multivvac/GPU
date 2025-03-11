#include "kernel.h"
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
} // namespace matrix_transpose

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <M> <N> <seed>\n";
  }
  int M = std::stoi(argv[1]);
  int N = std::stoi(argv[2]);
  int seed = std::stoi(argv[3]);

  auto input = matrix_transpose::generate_input(M, N, seed);
  auto output = matrix_transpose::solution(input);

  std::cout << input << std::endl;
  std::cout << output << std::endl;
}

#include "utils/timer.hpp"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <iostream>
#include <torch/torch.h>

int main() {
  auto timer = StopWatch<chrono_alias::ns>();
  timer.start();
  int size = 1;
  int contention = 10;
  auto gen = torch::make_generator<at::CUDAGeneratorImpl>();
  auto data = torch::randint(0, 256, {size}, gen,
                             torch::dtype(torch::kInt8).device(torch::kCUDA));
  auto evilvalue = torch::randint(
      0, 256, {}, gen, torch::dtype(torch::kInt8).device(torch::kCUDA));
  auto evilloc =
      torch::rand({size}, gen,
                  torch::dtype(torch::kFloat32).device(torch::kCUDA)) <
      (contention / 100.0);
  data[evilloc] = evilvalue;
  std::cout << data << std::endl;
  timer.stop();

  std::cout << "duration: " << timer.getTime().count() << std::endl;
}

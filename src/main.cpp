#include "utils/timer.hpp"
#include "utils/validate.hpp"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <iostream>
#include <torch/torch.h>

int main() {
  auto timer = StopWatch<chrono_alias::ns>();
  timer.start();
  int size = 3;
  int contention = 10;
  auto gen = torch::make_generator<at::CUDAGeneratorImpl>();
  auto data = torch::randint(0, 256, {size, 2}, gen,
                             torch::dtype(torch::kUInt8).device(torch::kCUDA));
  auto evilvalue = torch::randint(
      0, 256, {}, gen, torch::dtype(torch::kUInt8).device(torch::kCUDA));
  auto evilloc =
      torch::rand({size, 2}, gen,
                  torch::dtype(torch::kFloat32).device(torch::kCUDA)) <
      (contention / 100.0);
  data.masked_fill_(evilloc, evilvalue);
  std::cout << data.contiguous() << std::endl;
  timer.stop();

  verbose_allclose(data, data, 0.0, 0.0);
  std::cout << "duration: " << timer.getTime().count() << std::endl;
}

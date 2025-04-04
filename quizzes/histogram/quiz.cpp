#include "histogram_kernel.h"
#include "utils/benchmark.hpp"
#include "utils/stat.hpp"
#include "utils/validate.hpp"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cstdlib>
#include <iostream>
#include <torch/torch.h>
#include <torch/types.h>

namespace histogram {
torch::Tensor generate_input(long size, float contention, int seed) {

  auto gen = torch::make_generator<at::CUDAGeneratorImpl>(seed);
  // seed won't work without manually setting current seed.
  gen.set_current_seed(seed);
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
torch::Tensor solution_coarse(torch::Tensor &data) {
  return histogram_coarse_cuda(data);
}
torch::Tensor solution_coarse_contiguous(torch::Tensor &data) {
  return histogram_coarse_contiguous_cuda(data);
}
torch::Tensor solution_vec(torch::Tensor &data) {
  return histogram_vec_cuda(data);
}
torch::Tensor solution_aggregation(torch::Tensor &data) {
  return histogram_aggregation_cuda(data);
}
} // namespace histogram

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <size> <contention> <seed>\n";
  }
  long size = std::stoll(argv[1]);
  int contention = std::stoi(argv[2]);
  int seed = std::stoi(argv[3]);

  auto input = histogram::generate_input(size, contention, seed);
  auto output = histogram::baseline(input);
  auto my_output = histogram::solution(input);
  auto my_coarse_output = histogram::solution_coarse(input);
  auto my_vec_output = histogram::solution_vec(input);
  auto my_coarse_contiguous_output =
      histogram::solution_coarse_contiguous(input);
  auto aggregation_output = histogram::solution_aggregation(input);

  auto errors = verbose_allequal(my_output, output);

  if (errors.size() > 0) {
    std::cout << "found errors in shared memory kernel:\n";
    for (auto &error : errors) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }
  auto errors_coarse = verbose_allequal(my_coarse_output, output);

  if (errors_coarse.size() > 0) {
    std::cout << "found errors in coarsening kernel:\n";
    for (auto &error : errors_coarse) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }

  auto errors_coarse_contiguous =
      verbose_allequal(my_coarse_contiguous_output, output);
  if (errors_coarse_contiguous.size() > 0) {
    std::cout << "found errors in coarsening contiguous kernel:\n";
    for (auto &error : errors_coarse_contiguous) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }
  auto errors_vec = verbose_allequal(my_vec_output, output);

  if (errors_vec.size() > 0) {
    std::cout << "found errors in vectorized kernel:\n";
    for (auto &error : errors_vec) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }

  auto errors_aggregation = verbose_allequal(aggregation_output, output);

  if (errors_aggregation.size() > 0) {
    std::cout << "found errors in aggregation kernel:\n";
    for (auto &error : errors_aggregation) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }
  // benchmark

  auto selftime = benchmark([&]() { histogram::solution(input); });
  auto selfcoarsetime = benchmark([&]() { histogram::solution_coarse(input); });
  auto self_coarse_contiguous_time =
      benchmark([&]() { histogram::solution_coarse_contiguous(input); });
  auto selfvectime = benchmark([&]() { histogram::solution_vec(input); });
  auto aggregation_time =
      benchmark([&]() { histogram::solution_aggregation(input); });
  auto baselinetime = benchmark([&]() { histogram::baseline(input); });

  print_table(
      std::vector<FunctionTiming>{
          FunctionTiming{std::string("Privatization Implementation"), selftime,
                         baselinetime / selftime},
          FunctionTiming{
              std::string(
                  "Coarsening + Privatization Interleaved Implementation"),
              selfcoarsetime, baselinetime / selfcoarsetime},
          FunctionTiming{
              std::string(
                  "Coarsening + Privatization Contiguous Implementation"),
              self_coarse_contiguous_time,
              baselinetime / self_coarse_contiguous_time},
          FunctionTiming{
              std::string("Vectorized + Privatization Implementation"),
              selfvectime, baselinetime / selfvectime},
          FunctionTiming{std::string("Aggregation Implementation"),
                         aggregation_time, baselinetime / aggregation_time},
          FunctionTiming{std::string("Pytorch Implementation(baseline)"),
                         baselinetime, baselinetime / baselinetime}},
      "Histogram Kernel");
}

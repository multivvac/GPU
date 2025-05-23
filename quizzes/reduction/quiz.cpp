#include "reduction_kernel.h"
#include "utils/benchmark.hpp"
#include "utils/helper.hpp"
#include "utils/stat.hpp"
#include "utils/validate.hpp"
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <iostream>
#include <torch/torch.h>

namespace reduction {
torch::Tensor generate_input(size_t size, int seed) {

  auto gen = torch::make_generator<at::CUDAGeneratorImpl>(seed);
  size_t size_pow_of_2 = next_pow_of_2(size);
  // seed won't work without manually setting current seed.
  gen.set_current_seed(seed);
  auto data = torch::randint(
      10, {static_cast<int64_t>(size_pow_of_2)}, gen,
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
  auto tmp = data.clone();
  return reduction_naive_cuda(tmp);
}
torch::Tensor convergent_solution(torch::Tensor &data) {
  auto tmp = data.clone();
  return reduction_convergent_cuda(tmp);
}
torch::Tensor shared_mem_solution(torch::Tensor &data) {
  auto tmp = data.clone();
  return reduction_shared_mem_cuda(tmp);
}
torch::Tensor thread_coarsening_solution(torch::Tensor &data) {
  auto tmp = data.clone();
  return reduction_thread_coarsening_cuda(tmp);
}
} // namespace reduction

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <size> <seed>\n";
  }
  size_t size = std::stoul(argv[1]);
  int seed = std::stoi(argv[2]);

  auto input = reduction::generate_input(size, seed);
  auto output = reduction::baseline(input);
  auto naive_output = reduction::solution(input);
  auto convergent_output = reduction::convergent_solution(input);
  auto shared_mem_output = reduction::shared_mem_solution(input);
  auto thread_coarsening_output = reduction::thread_coarsening_solution(input);

  auto errors = verbose_allequal(naive_output, output);

  if (errors.size() > 0) {
    std::cout << "found errors in basic reduction kernel:\n";
    for (auto &error : errors) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }

  auto convergent_errors = verbose_allequal(convergent_output, output);

  if (convergent_errors.size() > 0) {
    std::cout << "found errors in convergent reduction kernel:\n";
    for (auto &error : convergent_errors) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }

  auto shared_mem_errors = verbose_allequal(shared_mem_output, output);

  if (shared_mem_errors.size() > 0) {
    std::cout << "found errors in shared memory kernel:\n";
    for (auto &error : shared_mem_errors) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }

  auto thread_coarsening_errors =
      verbose_allequal(thread_coarsening_output, output);

  if (thread_coarsening_errors.size() > 0) {
    std::cout << "found errors in thread coarsening kernel:\n";
    for (auto &error : thread_coarsening_errors) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }

  // benchmark
  auto baselinetime = benchmark([&]() { reduction::baseline(input); });
  auto naivetime = benchmark([&]() { reduction::solution(input); });
  auto convergent_time =
      benchmark([&]() { reduction::convergent_solution(input); });
  auto shared_mem_time =
      benchmark([&]() { reduction::shared_mem_solution(input); });
  auto thread_coarsening_time =
      benchmark([&]() { reduction::thread_coarsening_solution(input); });

  print_table(
      std::vector<FunctionTiming>{
          FunctionTiming{std::string("Naive Implementation"), naivetime,
                         baselinetime / naivetime},
          FunctionTiming{std::string("Convergent Implementation"),
                         convergent_time, baselinetime / convergent_time},
          FunctionTiming{std::string("Shared Memory Implementation"),
                         shared_mem_time, baselinetime / shared_mem_time},
          FunctionTiming{std::string("Thread Coarsening Implementation"),
                         thread_coarsening_time,
                         baselinetime / thread_coarsening_time},
          FunctionTiming{std::string("Pytorch Implementation(baseline)"),
                         baselinetime, baselinetime / baselinetime}},
      "Reduction Kernel");
}

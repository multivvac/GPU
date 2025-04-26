
#include "lenet_kernel.h"
#include "utils/benchmark.hpp"
#include "utils/stat.hpp"
#include "utils/validate.hpp"
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <iostream>
#include <torch/nn/functional/fold.h>
#include <torch/nn/options/fold.h>
#include <torch/torch.h>
#include <torch/types.h>
#define KERNEL_SIZE 16
#define IMAGE_HEIGHT 512
#define IMAGE_WIDTH 512
#define CHAN 8
#define BATCH 1

namespace F = torch::nn::functional;
torch::Tensor generate_input(int N, int C, int H, int W, int seed) {

  auto gen = torch::make_generator<at::CUDAGeneratorImpl>(seed);
  // seed won't work without manually setting current seed.
  gen.set_current_seed(seed);
  auto data = torch::rand({N, C, H, W}, gen,
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
  return data.contiguous();
}

torch::Tensor im2col_baseline(torch::Tensor &data, size_t K) {
  auto output = F::unfold(data, F::UnfoldFuncOptions(K).stride(1));
  torch::cuda::synchronize();
  return output;
}

torch::Tensor conv2d_im2col(torch::Tensor &data, torch::Tensor &filter,
                            size_t K) {
  auto output = conv2d_im2col_cuda(data, filter, K);
  return output;
}

torch::Tensor conv2d_baseline(torch::Tensor &data, torch::Tensor &filter,
                              size_t K) {
  auto output =
      F::conv2d(data, filter, F::Conv2dFuncOptions().stride(1).padding(0));
  torch::cuda::synchronize();
  return output;
}
int test_im2col() {
  auto input = generate_input(BATCH, CHAN, IMAGE_HEIGHT, IMAGE_WIDTH, 42);
  auto output = im2col_baseline(input, KERNEL_SIZE);
  auto im2col_output = im2col_cuda(input, KERNEL_SIZE);
  auto im2col_optimized_output = im2col_optimized_cuda(input, KERNEL_SIZE);
  auto errors = verbose_allequal(im2col_output.flatten(), output.flatten());
  auto opt_errors =
      verbose_allequal(im2col_optimized_output.flatten(), output.flatten());

  if (errors.size() > 0) {
    std::cout << "found errors in im2col kernel:\n";
    for (auto &error : errors) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }
  if (opt_errors.size() > 0) {
    std::cout << "found errors in im2col optimized kernel:\n";
    for (auto &error : opt_errors) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }
  return 0;
}

int test_conv2d() {
  auto input =
      generate_input(BATCH, KERNEL_1_IN_CHAN, IMAGE_WIDTH, IMAGE_HEIGHT, 42);
  auto filter = torch::nn::init::kaiming_uniform_(torch::zeros(
      {KERNEL_1_OUT_CHAN, KERNEL_1_IN_CHAN, KERNEL_1_SIZE, KERNEL_1_SIZE},
      torch::dtype(torch::kFloat32).device(torch::kCUDA)));
  auto output = conv2d_im2col(input, filter, KERNEL_1_SIZE);
  auto baseline_output = conv2d_baseline(input, filter, KERNEL_1_SIZE);
  auto errors =
      verbose_allclose(baseline_output.flatten(), output.flatten(), 1e-5, 1e-5);
  if (errors.size() > 0) {
    std::cout << "found errors in conv2d kernel:\n";
    for (auto &error : errors) {
      std::cout << error << "\n";
    }
    return EXIT_FAILURE;
  }
  return 0;
}

int main(int argc, char *argv[]) {

  std::cout << "Running tests...\n";
  int status = 0;

  status |= test_im2col();
  status |= test_conv2d();

  if (status == 0) {
    std::cout << "\n✅ All tests passed successfully!\n";
  } else {
    std::cerr << "\n❌ Some tests failed.\n";
    return status;
  }

  auto input = generate_input(BATCH, CHAN, IMAGE_HEIGHT, IMAGE_HEIGHT, 42);
  auto torch_benchmark =
      benchmark([&]() { im2col_baseline(input, KERNEL_SIZE); });
  auto naive_benchmark = benchmark([&]() { im2col_cuda(input, KERNEL_SIZE); });
  auto opt_benchmark =
      benchmark([&]() { im2col_optimized_cuda(input, KERNEL_SIZE); });
  print_table(
      std::vector<FunctionTiming>{
          FunctionTiming{std::string("Naive Implementation"), naive_benchmark,
                         torch_benchmark / naive_benchmark},
          FunctionTiming{std::string("'Optimized' Implementation"),
                         opt_benchmark, torch_benchmark / opt_benchmark},
          FunctionTiming{std::string("PyTorch Implementation"), torch_benchmark,
                         torch_benchmark / torch_benchmark}},
      "Im2Col Kernel");

  auto conv2d_input =
      generate_input(BATCH, KERNEL_1_IN_CHAN, IMAGE_WIDTH, IMAGE_HEIGHT, 42);
  auto filter = torch::nn::init::kaiming_uniform_(torch::zeros(
      {KERNEL_1_OUT_CHAN, KERNEL_1_IN_CHAN, KERNEL_1_SIZE, KERNEL_1_SIZE},
      torch::dtype(torch::kFloat32).device(torch::kCUDA)));
  auto conv2d_torch_benchmark = benchmark(
      [&]() { conv2d_baseline(conv2d_input, filter, KERNEL_1_SIZE); });
  auto conv2d_im2col_benchmark = benchmark(
      [&]() { conv2d_im2col_cuda(conv2d_input, filter, KERNEL_1_SIZE); });
  print_table(
      std::vector<FunctionTiming>{
          FunctionTiming{std::string("Naive Im2Col Implementation"),
                         conv2d_im2col_benchmark,
                         conv2d_torch_benchmark / conv2d_im2col_benchmark},
          FunctionTiming{std::string("PyTorch Implementation"),
                         conv2d_torch_benchmark,
                         conv2d_torch_benchmark / conv2d_torch_benchmark}},
      "Conv2d Kernel");
}

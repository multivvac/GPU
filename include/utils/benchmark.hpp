#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "utils/timer.hpp"
#include <utility>
template <typename Func, typename... Args>
double benchmark(Func &&func, int warmup_runs = 2, int actual_runs = 5,
                 Args &&...args) {

  // warm up
  for (size_t i = 0; i < warmup_runs; i++) {
    func(std::forward<Args>(args)...);
  }
  auto timer = StopWatch<chrono_alias::ns>();
  for (size_t i = 0; i < actual_runs; i++) {
    timer.start();
    func(std::forward<Args>(args)...);
    timer.stop();
  }
  return timer.getAverageTime().count();
}
#endif // ! BENCHMARK_H

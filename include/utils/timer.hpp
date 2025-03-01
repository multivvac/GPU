#ifndef TIMER_H
#define TIMER_H

#include <chrono>

namespace chrono_alias {
using s = std::chrono::seconds;
using ms = std::chrono::milliseconds;
using us = std::chrono::microseconds;
using ns = std::chrono::nanoseconds;
} // namespace chrono_alias

template <typename T> class StopWatch {
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  T diff_time = T();
  T total_time = T();
  bool running = false;
  int clock_sessions = 0;

public:
  ~StopWatch() {};
  inline void start() {
    start_time = std::chrono::high_resolution_clock::now();
    running = true;
  };
  inline void stop() {
    running = false;
    diff_time = getDiffTime();
    total_time += diff_time;
    clock_sessions++;
  };
  inline void reset() {
    diff_time = T();
    total_time = T();
    clock_sessions = 0;

    if (running) {
      start_time = std::chrono::high_resolution_clock::now();
    }
  };

  inline T getTime() {
    auto retval = total_time;

    if (running) {
      total_time += getDiffTime();
    }

    return retval;
  };
  inline T getAverageTime() {
    return (clock_sessions > 0) ? (total_time / clock_sessions) : T();
  };

private:
  inline T getDiffTime() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<T>(now - start_time);
  };
};

#endif // !TIMER_H

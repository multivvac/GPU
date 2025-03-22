#ifndef STAT_H
#define STAT_H

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

struct FunctionTiming {
  std::string name;
  double duration_ns;
  double speedup;
};

std::string format_speedup(double speedup, int precision = 4) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << speedup;
  if (speedup > 2.0)
    oss << "x ⚡⚡"; // very fast
  else if (speedup > 1.0)
    oss << "x ⚡"; // faster than baseline
  else if (speedup == 1.0)
    oss << "x 💨"; // baseline
  else
    oss << "x 🐢"; // slow

  return oss.str();
}

void print_table(const std::vector<FunctionTiming> &timings,
                 const std::string &name) {
  size_t maxNameLen = 0;
  for (const auto &t : timings) {
    maxNameLen = std::max(maxNameLen, t.name.size());
  }
  maxNameLen = std::max(maxNameLen, name.size());

  const std::string durationHeader = "Duration (ns)";
  const int durationColWidth = 15;
  const std::string speedupHeader = "Speedup";
  const int speedupColWidth = 12;

  size_t totalWidth = maxNameLen + 3 + durationColWidth + 3 + speedupColWidth;
  std::cout << std::string(totalWidth, '-') << "\n";

  std::cout << std::left << std::setw(static_cast<int>(maxNameLen)) << name
            << " | " << std::right << std::setw(durationColWidth)
            << durationHeader << " | " << std::right
            << std::setw(speedupColWidth) << speedupHeader << "\n";

  std::cout << std::string(totalWidth, '-') << "\n";

  for (const auto &t : timings) {
    std::cout << std::left << std::setw(static_cast<int>(maxNameLen)) << t.name
              << " | " << std::right << std::setw(durationColWidth)
              << t.duration_ns << " | " << std::right
              << std::setw(speedupColWidth) << format_speedup(t.speedup)
              << "\n";
  }

  std::cout << std::string(totalWidth, '-') << "\n";
}

#endif // !STAT_H

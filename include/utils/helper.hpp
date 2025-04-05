#ifndef HELPER_H
#define HELPER_H

#include <type_traits>
template <typename T> T next_pow_of_2(T n) {
  static_assert(std::is_unsigned<T>::value,
                "Only works with unsigned integer types");
  if (n == 0)
    return 1;
  n--;

  unsigned int shift = 1;
  // total bits of T
  while (shift < sizeof(T) * 8) {
    n |= n >> shift;
    shift <<= 1;
  }
  return n + 1;
}

#endif // !HELPER_H

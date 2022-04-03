#pragma once
#include <chrono>

using FunctionWrapperForTime = std::function<void()>;
static auto function_wrapper_for_time = [](const FunctionWrapperForTime& i_func, std::size_t i_iterations) {
  const auto start = std::chrono::system_clock::now();
  for (auto i = 0u; i < i_iterations; ++i)
    i_func();
  const auto end = std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
};
#include "Common/ThreadPool.h"
#include "Common/Randomizer.h"

#include <chrono>
#include <gtest/gtest.h>
#include <numeric>

namespace {
  using FunctionWrapperForTime = std::function<void()>;
  std::size_t GetFunctionExecutionTime(const FunctionWrapperForTime& i_func, const std::size_t i_iterations)
  {
    const auto start = std::chrono::system_clock::now();
    for (auto i = 0u; i < i_iterations; ++i)
      i_func();
    const auto end = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  };

}

TEST(ThreadPoolTest, ThreadPoolCommonScenario)
{
  constexpr std::size_t n = 1000000;
  constexpr std::size_t iterations = 1;

  auto sum_func = [](const std::size_t i_start, const std::size_t i_end) {
    long long sum = 0;
    for (auto i = i_start; i < i_end; ++i)
      sum += 2;
    return sum;
  };

  Parallel::ThreadPool* pool = Parallel::ThreadPool::GetInstance();

  std::vector<std::size_t> time_results;
  for (std::size_t k : { 1, 2, 4, 8 }) {
    const std::size_t interval_size = n / k;
    auto time = GetFunctionExecutionTime(
      [&]() {
        std::vector<std::future<long long>> results(k);
        for (std::size_t i = 0; i < k; ++i) {
          results[i] = pool->Enqueue(sum_func, i * interval_size, (i + 1) * interval_size);
        }
        long long sum = 0;
        for (auto& res : results)
          sum += res.get();
      },
      iterations);
    time_results.push_back(time);
  }

  for (std::size_t i = 0; i < time_results.size(); ++i)
    for (std::size_t j = i + 1; j < time_results.size(); ++j)
      EXPECT_TRUE(time_results[i] > time_results[j]);
}

TEST(ThreadPoolTest, ThreadPoolParallelFor)
{
  Randomizer random(123);

  constexpr std::size_t iterations = 1;
  constexpr std::size_t n = 10;

  std::vector<std::vector<int>> matrix(n, std::vector<int>(n));
  for (auto& row : matrix)
    for (auto& elem : row)
      elem = random.Next<int>() % 10;

  Parallel::ThreadPool* pool = Parallel::ThreadPool::GetInstance();

  const auto common_time = GetFunctionExecutionTime(
    [&]() {
      for (const auto& row : matrix)
        for (const auto& elem : row)
          std::this_thread::sleep_for(std::chrono::microseconds(elem));
    },
    iterations);

  const auto parallel_for_time = GetFunctionExecutionTime(
    [&]() {
      pool->ParallelFor(static_cast<std::size_t>(0), n, [&](std::size_t i) {
        for (const auto& elem : matrix[i])
          std::this_thread::sleep_for(std::chrono::microseconds(elem));
      });
    },
    iterations);

  const auto parallel_for_in_parallel_for_time = GetFunctionExecutionTime(
    [&]() {
      pool->ParallelFor(static_cast<std::size_t>(0), n, [&](std::size_t i) {
        pool->ParallelFor(static_cast<std::size_t>(0), n, [&](std::size_t j) {
          std::this_thread::sleep_for(std::chrono::microseconds(matrix[i][j]));
        });
      });
    },
    iterations);
  EXPECT_TRUE(common_time > parallel_for_time);
  EXPECT_TRUE(parallel_for_time > parallel_for_in_parallel_for_time);
}
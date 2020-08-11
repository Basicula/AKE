#include <gtest/gtest.h>
#include <chrono>
#include <numeric>

#include <Common/ThreadPool.h>
#include "Utils.h"

TEST(ThreadPoolTest, ThreadPoolCommonScenario)
  {
  const std::size_t n = 1000000;
  const std::size_t iterations = 1;

  std::vector<int> a(n);
  std::iota(a.begin(), a.end(), 0);
  auto sum_func = [&a](std::size_t i_start, std::size_t i_end)
    {
    long long sum = 0;
    for (auto i = i_start; i < i_end; ++i)
      sum += 2;
    return sum;
    };

  ThreadPool* pool = ThreadPool::GetInstance();

  std::vector<long long> time_results;
  for (auto k : { 1, 2, 4, 8 })
    {
    const std::size_t interval_size = n / k;
    auto time =
      function_wrapper_for_time([&]()
        {
        std::vector<std::future<long long>> results(k);
        for (int i = 0; i < k; ++i)
          {
          results[i] = pool->Enqueue(sum_func, i * interval_size, (i + 1) * interval_size);
          }
        long long sum = 0;
        for (auto& res : results)
          sum += res.get();
        },
        iterations);
    time_results.push_back(time);
    }

  for (int i = 0; i < time_results.size(); ++i)
    for (int j = i + 1; j < time_results.size(); ++j)
      EXPECT_TRUE(time_results[i] > time_results[j]);
  }

TEST(ThreadPoolTest, ThreadPoolParallelFor)
  {
  srand(123);

  const std::size_t iterations = 1;
  const std::size_t n = 10;

  std::vector<std::vector<int>> matrix(n, std::vector<int>(n));
  for (auto& row : matrix)
    for (auto& elem : row)
      elem = rand() % 10;

  ThreadPool* pool = ThreadPool::GetInstance();

  auto common_time =
    function_wrapper_for_time(
      [&]()
      {
      for (const auto& row : matrix)
        for (const auto& elem : row)
          std::this_thread::sleep_for(std::chrono::microseconds(elem));
      },
      iterations);

  auto parallel_for_time =
    function_wrapper_for_time(
      [&]()
      {
      pool->ParallelFor(
        static_cast<std::size_t>(0),
        n,
        [&](std::size_t i)
        {
        for (const auto& elem : matrix[i])
          std::this_thread::sleep_for(std::chrono::microseconds(elem));
        });
      },
      iterations);

  auto parallel_for_in_parallel_for_time =
    function_wrapper_for_time(
      [&]()
      {
      pool->ParallelFor(
        static_cast<std::size_t>(0),
        n,
        [&](std::size_t i)
        {
        pool->ParallelFor(
          static_cast<std::size_t>(0),
          n,
          [&](std::size_t j)
          {
          std::this_thread::sleep_for(std::chrono::microseconds(matrix[i][j]));
          });
        });
      },
      iterations);
  EXPECT_TRUE(common_time > parallel_for_time);
  EXPECT_TRUE(parallel_for_time > parallel_for_in_parallel_for_time);
  }
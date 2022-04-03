#pragma once
#include <algorithm>

using namespace Parallel;

template <class IndexType, class Function>
void ThreadPool::ParallelFor(IndexType i_start, IndexType i_end, const Function& i_function)
{
  const std::size_t num_of_tasks = static_cast<std::size_t>(i_end - i_start + 1);
  auto bucket_size = (num_of_tasks + m_workers.size() - 1) / m_workers.size();

  auto worker_task = [&i_function](IndexType i_start, IndexType i_end) {
    for (auto i = i_start; i < i_end; ++i)
      i_function(i);
  };

  std::vector<std::future<typename std::invoke_result<Function, IndexType>::type>> results;
  IndexType i1 = i_start;
  IndexType i2 = std::min(i_start + bucket_size, i_end);
  for (auto i = 0u; i1 < i_end; ++i) {
    results.push_back(Enqueue(worker_task, i1, i2));
    i1 = i2;
    i2 = std::min(i1 + bucket_size, i_end);
  }
  for (auto& reslult : results)
    reslult.get();
}

template <class Function, class... Args>
inline auto ThreadPool::Enqueue(Function&& i_function, Args&&... i_args)
  -> std::future<typename std::invoke_result<Function, Args...>::type>
{
  using ResultType = typename std::invoke_result<Function, Args...>::type;
  auto task = std::make_shared<std::packaged_task<ResultType()>>(
    std::bind(std::forward<Function>(i_function), std::forward<Args>(i_args)...));
  std::future<ResultType> result = task->get_future();
  if (true) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_tasks.emplace([task]() { (*task)(); });
  }
  m_wait_condition.notify_one();
  return result;
}
#include <ThreadPool.h>
using namespace Parallel;

// static members definition
Parallel::ThreadPool* Parallel::ThreadPool::m_instance = nullptr;
std::size_t Parallel::ThreadPool::m_in_progress_cnt = 0;
const std::size_t Parallel::ThreadPool::m_max_workers = 16;

ThreadPool::ThreadPool()
  : m_stop(false)
  {
  const auto num_of_workers_hint = std::thread::hardware_concurrency();
  const auto num_of_workers = (num_of_workers_hint == 0u ? 8u : num_of_workers_hint);
  for (std::size_t i = 0; i < num_of_workers; ++i)
    _AddWorker();
  }

ThreadPool::~ThreadPool()
  {
  if (true)
    {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_stop = true;
    }
  m_wait_condition.notify_all();
  for (auto& worker : m_workers)
    worker.join();
  }

void Parallel::ThreadPool::_AddWorker()
  {
  m_workers.emplace_back(
    [this]()
    {
    while (true)
      {
      ThreadTask task;

      // critical section waiting for task
      if (true)
        {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_wait_condition.wait(
          lock,
          [this]()
          {
          return m_stop || !m_tasks.empty();
          });
        if (m_stop && m_tasks.empty())
          return;
        task = std::move(m_tasks.front());
        m_tasks.pop();
        }
      // execute task
      ++m_in_progress_cnt;
      task();
      --m_in_progress_cnt;
      }
    });
  }
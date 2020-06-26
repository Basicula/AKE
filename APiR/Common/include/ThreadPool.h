#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <functional>
#include <mutex>
#include <future>

namespace Parallel
  {
  class ThreadPool
    {
    public:
      ~ThreadPool();

      static ThreadPool* GetInstance();

      template<class Function, class... Args>
      auto Enqueue(Function&& i_function, Args&&... i_args)
        ->std::future<typename std::invoke_result<Function, Args...>::type>;

      template<class IndexType, class Function>
      void ParallelFor(
        IndexType i_start,
        IndexType i_end,
        const Function& i_function);

    private:
      using ThreadTask = std::function<void()>;

    private:
      ThreadPool();
      ThreadPool(const ThreadPool& other) = delete;
      ThreadPool(ThreadPool&& other) = delete;

      void _AddWorker();

    private:
      static ThreadPool* m_instance;
      static std::size_t m_in_progress_cnt;
      static const std::size_t m_max_workers;

      std::vector<std::thread> m_workers;
      std::queue<ThreadTask> m_tasks;
      std::mutex m_mutex;
      std::condition_variable m_wait_condition;
      bool m_stop;
    };

  inline ThreadPool* ThreadPool::GetInstance()
    {
    if (!m_instance)
      m_instance = new ThreadPool();
    return m_instance;
    }
  }

#include <ThreadPool_impl.h>
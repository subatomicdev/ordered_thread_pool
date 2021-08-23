// Implements a synchronized thread pool.
//
// This parallelizes jobs of type -
//
//   while (...) {
//      CostlyFn(myArg);
//   }
//
// To parallelize, use the following pattern -
//
//   pool = OrderedThreadPool{10};
//   while (...)
//   {
//     pool.Do(CostlyFn, myArg);
//   }
//
// Properties -
// - All CostlyFn()'s are allowed to run in parallel.
// - Avoids repeated thread creation by reusing threads.
// - If workers are full, this blocks untill next one is free.
// - On destruction blocks untill all pending work is finished.
//
#ifndef ORDERED_THREAD_POOL_H
#define ORDERED_THREAD_POOL_H

#include <condition_variable>
#include <functional>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

template <class ArgType>
class OrderedThreadPool
{
  using JobFnT = std::function<void(ArgType)>;

 public:
  /**
   * Instantiates an ordered queue.
   *
   * @param num_workers Number of workers to spawn. A value of 0 will spawn no
   *   threads, and use the calling thread to perform the work.
   * @param max_pending_jobs If the workers are all occupied, and this many jobs
   *   are in the queue, calling thread will be blocked till a worker is free.
   **/
  OrderedThreadPool(int num_workers, int max_pending_jobs = 1) : max_queue_size_(max_pending_jobs)
  {
    for (int i = 0; i < num_workers; ++i)
    {
      workers_.push_back(std::thread(&OrderedThreadPool::Worker, this));
    }
  }

  // Movable but not copyable.
  OrderedThreadPool(OrderedThreadPool&& other);
  OrderedThreadPool& operator=(OrderedThreadPool&& other);

  /**
   * Starts processing of a new job.
   *
   * The work is logically similar to on_completion(fn()). In fact it is exactly
   * that if threading is disabled with num_workers = 0 during construction.
   *
   * @param fn A function spec that constitutes bulk of the job. This will be
   *   parallelized.
   * @param on_completion A function which will be called with the result of
   *   fn().
   **/
  void Do(JobFnT fn, ArgType arg)
  {

    // Push to the job queue and notify.
    std::unique_lock<std::mutex> lck(fn_queue_mtx_);
    job_removed_.wait(lck, [this]
    {
      return max_queue_size_ == 0 || (int)fn_queue_.size() < max_queue_size_;
    });

    fn_queue_.push( Job{.job_fn = fn, .job_id = job_count_++, .arg = std::move(arg)});

    job_added_.notify_one();
  }

  virtual ~OrderedThreadPool()
  {
    terminate_now_ = true;
    {
      // Notify holding the lock.
      // This prevents missing a notification if this executes inbetween when
      // the wait() checks the predicate to be false and relocks.
      std::lock_guard<std::mutex> lck(fn_queue_mtx_);
      job_added_.notify_all();
    }
    
    for (std::thread& t : workers_)
      t.join();
  }

 private:
  struct Job
  {
    // A function which will be parallelized.
    JobFnT job_fn;
    // Internal ticket number. Used in waiting for previous jobs.
    size_t job_id;
    // Argument for the JobFnT
    ArgType arg;
  };

  // Blocks till a next job is available, or termination signal is received.
  // If termination is requested, returns empty.
  std::optional<Job> NextJob()
  {
    std::unique_lock<std::mutex> lck(fn_queue_mtx_);
    job_added_.wait(lck, [this] { return !fn_queue_.empty() || terminate_now_; });
    // If requested to terminate, finish the entire queue and exit.
    if (terminate_now_ && fn_queue_.empty()) {
      return {};
    }
    Job result = fn_queue_.front();
    fn_queue_.pop();
    job_removed_.notify_one();
    return result;
  }

  void Worker()
  {
    while (true)
    {
      std::optional<Job> job_opt = NextJob();
      if (!job_opt.has_value())
      {
        // This means the workers should terminate.
        return;
      }
      Job job = job_opt.value();

      // This runs parallelly across all threads.
      job.job_fn(job.arg);

      // Wait till our turn comes.
      std::unique_lock<std::mutex> lck(ticket_mtx_);
      ticket_update_.wait(lck, [this, &job] { return ticket_num_ == job.job_id; });
      
      // Update the next ticket and send a signal to other workers in line.
      ++ticket_num_;
      ticket_update_.notify_all();
    }
  }

  // The worker threads are initialized on construction and maintained.
  std::vector<std::thread> workers_;
  // Queue of functions to execute.
  std::queue<Job> fn_queue_;
  std::mutex fn_queue_mtx_;
  std::condition_variable job_added_;
  std::condition_variable job_removed_;
  int max_queue_size_;
  // Incremental job_id passed to each job.
  size_t job_count_ = 0;

  // Ticket system to ensure chronological delivery of jobs. The next job
  // with job_id matching this will proceed
  size_t ticket_num_ = 0;
  // The mutex to lock for second function.
  std::mutex ticket_mtx_;
  std::condition_variable ticket_update_;

  // If true, the worker threads should stop.
  bool terminate_now_ = false;
};

#endif  // ORDERED_THREAD_POOL_H

#pragma once

#include <condition_variable>  // NOLINT(build/c++11)
#include <mutex>               // NOLINT(build/c++11)
#include <queue>               // NOLINT(build/c++11)
#include <thread>              // NOLINT(build/c++11)

namespace mlperf {

template <typename T>

// TODO(chiachenc): replace code with lockless queue here.
class Queue {
 public:
  Queue() {}
  T get() {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty()) {
      cond_.wait(mlock);
    }
    auto item = queue_.front();
    queue_.pop();
    return item;
  }
  void put(const T& item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one();
  }

  bool empty() { return queue_.empty(); }

 private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
};

}  // namespace mlperf
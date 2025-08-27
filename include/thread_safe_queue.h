#pragma once
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include "data_types.h"

template<typename T>
class ThreadSafeQueue {
private:
    mutable std::mutex mtx_;
    std::queue<T> data_queue_;
    std::condition_variable condition_;
    std::atomic<bool> finished_{false};

public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mtx_);
        data_queue_.push(item);
        condition_.notify_one();
    }

    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (data_queue_.empty()) {
            return false;
        }
        item = data_queue_.front();
        data_queue_.pop();
        return true;
    }

    bool wait_and_pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx_);
        while (data_queue_.empty() && !finished_) {
            condition_.wait(lock);
        }
        if (!data_queue_.empty()) {
            item = data_queue_.front();
            data_queue_.pop();
            return true;
        }
        return false;  // 队列已完成且为空
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return data_queue_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return data_queue_.size();
    }

    void set_finished() {
        std::lock_guard<std::mutex> lock(mtx_);
        finished_ = true;
        condition_.notify_all();
    }

    bool is_finished() const {
        return finished_.load();
    }
};

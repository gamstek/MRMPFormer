#ifndef MRMPFORMER_TASK_MANAGER_H
#define MRMPFORMER_TASK_MANAGER_H

#include "json_protocol.h"
#include "mrmpformer.h"
#include "onnx_inference.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace mrmpformer {

class TaskManager {
public:
    explicit TaskManager(const QfConfig& config);
    ~TaskManager();

    TaskManager(const TaskManager&) = delete;
    TaskManager& operator=(const TaskManager&) = delete;

    QfError start(std::string* error);
    void shutdown();

    QfError submit(std::vector<CompoundData> items,
                   int64_t* task_id,
                   std::string* error);
    QfError stop(int64_t task_id, std::string* error);
    QfError wait(int64_t task_id, int32_t timeout_ms, std::string* error);
    QfError query(int64_t task_id,
                  QfTaskStatus* status,
                  float* progress,
                  int32_t* items_done,
                  int32_t* items_total,
                  std::string* error) const;
    QfError set_callback(QfCallback callback, void* user_data);
    QfError result_path(int64_t task_id,
                        std::string* path,
                        std::string* error) const;
    QfError result_json(int64_t task_id,
                        std::string* json,
                        std::string* error) const;
    bool is_gpu_enabled() const;
    bool is_worker_or_callback_context() const noexcept;

private:
    struct OwnedConfig;
    struct Task;

    void worker_loop();
    void process_task(const std::shared_ptr<Task>& task);
    bool cancellation_requested(const std::shared_ptr<Task>& task) const;
    bool processing_timed_out(const std::shared_ptr<Task>& task) const;
    void update_progress(const std::shared_ptr<Task>& task, int32_t items_done);
    void complete_success(const std::shared_ptr<Task>& task,
                          std::string path,
                          std::string json);
    void complete_failure(const std::shared_ptr<Task>& task,
                          std::string error);
    void dispatch_callback(const std::shared_ptr<Task>& task);

    std::unique_ptr<OwnedConfig> config_;
    OnnxInference inference_;

    mutable std::mutex tasks_mutex_;
    std::condition_variable tasks_cv_;
    std::unordered_map<int64_t, std::shared_ptr<Task>> tasks_;
    std::deque<std::shared_ptr<Task>> pending_;
    bool stopping_ = false;
    std::vector<std::thread> workers_;

    mutable std::mutex callback_mutex_;
    QfCallback callback_ = nullptr;
    void* callback_user_data_ = nullptr;
};

}  // namespace mrmpformer

#endif

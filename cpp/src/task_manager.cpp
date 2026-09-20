#include "task_manager.h"

#include "roi_renderer.h"
#include "signal_processing.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <utility>

namespace mrmpformer {
namespace {

namespace fs = std::filesystem;

std::atomic<int64_t> next_task_id{1};
thread_local const TaskManager* worker_context = nullptr;
thread_local const TaskManager* callback_context = nullptr;

class ContextGuard {
public:
    ContextGuard(const TaskManager*& slot, const TaskManager* value)
        : slot_(slot), previous_(slot) {
        slot_ = value;
    }

    ~ContextGuard() {
        slot_ = previous_;
    }

    ContextGuard(const ContextGuard&) = delete;
    ContextGuard& operator=(const ContextGuard&) = delete;

private:
    const TaskManager*& slot_;
    const TaskManager* previous_;
};

bool terminal(QfTaskStatus status) {
    return status == QF_TASK_SUCCESS || status == QF_TASK_FAILED ||
           status == QF_TASK_CANCELLED;
}

double pixel_to_rt(float pixel_x, const RoiImage& image) {
    const double fraction = std::clamp(
        static_cast<double>(pixel_x) / static_cast<double>(image.width),
        0.0,
        1.0);
    return image.rt_min + fraction * (image.rt_max - image.rt_min);
}

std::optional<PeakResult> detection_to_peak(const RoiImage& image,
                                            const Detection& detection) {
    if (image.width <= 0 || !std::isfinite(image.rt_min) ||
        !std::isfinite(image.rt_max) || !(image.rt_min < image.rt_max) ||
        !std::isfinite(detection.x1) || !std::isfinite(detection.x2) ||
        !std::isfinite(detection.score) || !(detection.x1 < detection.x2)) {
        return std::nullopt;
    }

    const double mapped_left = pixel_to_rt(detection.x1, image);
    const double mapped_right = pixel_to_rt(detection.x2, image);
    if (!(mapped_left < mapped_right)) {
        return std::nullopt;
    }
    return PeakResult{mapped_left, mapped_right, detection.score};
}

CompoundResult low_intensity_result(const CompoundData& compound,
                                    const SignalQc& qc) {
    return {
        compound.uid,
        "alert",
        {},
        {{"fail",
          "CHANNEL_LOW_INTENSITY",
          {{"n_points", static_cast<double>(qc.n_points)},
           {"max_intensity", qc.max_intensity}}}},
    };
}

CompoundResult detection_result(const CompoundData& compound,
                                const RoiImage& image,
                                const std::vector<Detection>& detections) {
    CompoundResult result;
    result.uid = compound.uid;
    for (const Detection& detection : detections) {
        if (const auto peak = detection_to_peak(image, detection)) {
            result.peaks.push_back(*peak);
        }
    }
    std::sort(result.peaks.begin(), result.peaks.end(),
              [](const PeakResult& left, const PeakResult& right) {
                  if (left.a != right.a) {
                      return left.a < right.a;
                  }
                  if (left.b != right.b) {
                      return left.b < right.b;
                  }
                  return left.c > right.c;
              });

    if (!result.peaks.empty()) {
        result.status = "ok";
        return result;
    }

    const auto fallback = find_signal_fallback(compound);
    if (fallback && fallback->a < fallback->b && fallback->c > 0.0) {
        result.status = "review";
        result.peaks.push_back(*fallback);
        result.alerts.push_back(
            {"review",
             "SIGNAL_FALLBACK",
             {{"a", fallback->a}, {"b", fallback->b}, {"c", fallback->c}}});
        return result;
    }

    result.status = "alert";
    result.alerts.push_back({"fail", "NO_PEAK_FOUND", {}});
    return result;
}

}  // namespace

struct TaskManager::OwnedConfig {
    explicit OwnedConfig(const QfConfig& source)
        : model_path(source.model_path == nullptr ? "" : source.model_path),
          work_dir(source.work_dir == nullptr || source.work_dir[0] == '\0'
                       ? "./mrmpformer_results/"
                       : source.work_dir),
          log_file(source.log_file == nullptr ? "" : source.log_file),
          values(source) {
        values.model_path = model_path.c_str();
        values.work_dir = work_dir.c_str();
        values.log_file = log_file.empty() ? nullptr : log_file.c_str();
    }

    std::string model_path;
    std::string work_dir;
    std::string log_file;
    QfConfig values{};
};

struct TaskManager::Task {
    int64_t id = 0;
    QfTaskStatus status = QF_TASK_PENDING;
    float progress = 0.0f;
    int32_t items_done = 0;
    int32_t items_total = 0;
    bool cancel_requested = false;
    bool callback_dispatched = false;
    std::vector<CompoundData> items;
    std::string result_path;
    std::string result_json;
    std::string error;
    std::chrono::steady_clock::time_point submitted_at;
};

TaskManager::TaskManager(const QfConfig& config)
    : config_(std::make_unique<OwnedConfig>(config)) {}

TaskManager::~TaskManager() {
    shutdown();
}

QfError TaskManager::start(std::string* error) {
    if (!inference_.load_model(config_->model_path, config_->values)) {
        *error = inference_.last_error();
        return QF_ERR_MODEL_LOAD;
    }

    std::error_code filesystem_error;
    fs::create_directories(config_->work_dir, filesystem_error);
    if (filesystem_error) {
        *error = "failed to create result directory: " +
                 filesystem_error.message();
        return QF_ERR_FILE_WRITE;
    }

    int32_t worker_count = config_->values.max_workers;
    if (worker_count == 0) {
        const unsigned hardware = std::thread::hardware_concurrency();
        worker_count = static_cast<int32_t>(
            std::max(1u, std::min(4u, hardware == 0 ? 1u : hardware)));
    }
    try {
        workers_.reserve(static_cast<std::size_t>(worker_count));
        for (int32_t index = 0; index < worker_count; ++index) {
            workers_.emplace_back(&TaskManager::worker_loop, this);
        }
        return QF_OK;
    } catch (const std::exception& exception) {
        *error = std::string("failed to start worker threads: ") + exception.what();
        shutdown();
        return QF_ERR_NO_WORKER;
    }
}

void TaskManager::shutdown() {
    std::vector<std::shared_ptr<Task>> cancelled;
    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        if (stopping_ && workers_.empty()) {
            return;
        }
        stopping_ = true;
        for (const auto& [id, task] : tasks_) {
            (void)id;
            if (!terminal(task->status)) {
                task->cancel_requested = true;
                task->status = QF_TASK_CANCELLED;
                cancelled.push_back(task);
            }
        }
        tasks_cv_.notify_all();
    }
    for (const auto& task : cancelled) {
        dispatch_callback(task);
    }
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

QfError TaskManager::submit(std::vector<CompoundData> items,
                            int64_t* task_id,
                            std::string* error) {
    auto task = std::make_shared<Task>();
    task->id = next_task_id.fetch_add(1);
    if (task->id <= 0) {
        *error = "task identifier space is exhausted";
        return QF_ERR_INTERNAL;
    }
    task->items_total = static_cast<int32_t>(items.size());
    task->items = std::move(items);
    task->submitted_at = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        if (stopping_) {
            *error = "MRMPFormer is shutting down";
            return QF_ERR_NOT_INITIALIZED;
        }
        tasks_.emplace(task->id, task);
        pending_.push_back(task);
        *task_id = task->id;
        tasks_cv_.notify_one();
    }
    return QF_OK;
}

QfError TaskManager::stop(int64_t task_id, std::string* error) {
    std::shared_ptr<Task> task;
    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        const auto found = tasks_.find(task_id);
        if (found == tasks_.end()) {
            *error = "task not found";
            return QF_ERR_TASK_NOT_FOUND;
        }
        task = found->second;
        if (task->status == QF_TASK_CANCELLED) {
            *error = "task is already cancelled";
            return QF_ERR_TASK_CANCELLED;
        }
        if (terminal(task->status)) {
            *error = "task is already complete";
            return QF_ERR_INVALID_PARAM;
        }
        task->cancel_requested = true;
        task->status = QF_TASK_CANCELLED;
        tasks_cv_.notify_all();
    }
    dispatch_callback(task);
    return QF_OK;
}

QfError TaskManager::wait(int64_t task_id,
                          int32_t timeout_ms,
                          std::string* error) {
    std::unique_lock<std::mutex> lock(tasks_mutex_);
    const auto found = tasks_.find(task_id);
    if (found == tasks_.end()) {
        *error = "task not found";
        return QF_ERR_TASK_NOT_FOUND;
    }
    const auto task = found->second;
    const auto complete = [&] { return terminal(task->status); };
    if (timeout_ms == 0) {
        tasks_cv_.wait(lock, complete);
    } else if (!tasks_cv_.wait_for(
                   lock, std::chrono::milliseconds(timeout_ms), complete)) {
        *error = "timed out waiting for task";
        return QF_ERR_TIMEOUT;
    }
    return QF_OK;
}

QfError TaskManager::query(int64_t task_id,
                           QfTaskStatus* status,
                           float* progress,
                           int32_t* items_done,
                           int32_t* items_total,
                           std::string* error) const {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    const auto found = tasks_.find(task_id);
    if (found == tasks_.end()) {
        *error = "task not found";
        return QF_ERR_TASK_NOT_FOUND;
    }
    const Task& task = *found->second;
    if (status != nullptr) {
        *status = task.status;
    }
    if (progress != nullptr) {
        *progress = task.progress;
    }
    if (items_done != nullptr) {
        *items_done = task.items_done;
    }
    if (items_total != nullptr) {
        *items_total = task.items_total;
    }
    return QF_OK;
}

QfError TaskManager::set_callback(QfCallback callback, void* user_data) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    callback_ = callback;
    callback_user_data_ = user_data;
    return QF_OK;
}

QfError TaskManager::result_path(int64_t task_id,
                                 std::string* path,
                                 std::string* error) const {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    const auto found = tasks_.find(task_id);
    if (found == tasks_.end()) {
        *error = "task not found";
        return QF_ERR_TASK_NOT_FOUND;
    }
    const Task& task = *found->second;
    if (task.status == QF_TASK_CANCELLED) {
        *error = "task was cancelled";
        return QF_ERR_TASK_CANCELLED;
    }
    if (task.status != QF_TASK_SUCCESS) {
        *error = task.status == QF_TASK_FAILED ? task.error : "task is not complete";
        return task.status == QF_TASK_FAILED ? QF_ERR_INTERNAL
                                             : QF_ERR_INVALID_PARAM;
    }
    *path = task.result_path;
    return QF_OK;
}

QfError TaskManager::result_json(int64_t task_id,
                                 std::string* json,
                                 std::string* error) const {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    const auto found = tasks_.find(task_id);
    if (found == tasks_.end()) {
        *error = "task not found";
        return QF_ERR_TASK_NOT_FOUND;
    }
    const Task& task = *found->second;
    if (task.status == QF_TASK_CANCELLED) {
        *error = "task was cancelled";
        return QF_ERR_TASK_CANCELLED;
    }
    if (task.status != QF_TASK_SUCCESS) {
        *error = task.status == QF_TASK_FAILED ? task.error : "task is not complete";
        return task.status == QF_TASK_FAILED ? QF_ERR_INTERNAL
                                             : QF_ERR_INVALID_PARAM;
    }
    *json = task.result_json;
    return QF_OK;
}

bool TaskManager::is_gpu_enabled() const {
    return inference_.is_gpu_enabled();
}

bool TaskManager::is_worker_or_callback_context() const noexcept {
    return worker_context == this || callback_context == this;
}

bool TaskManager::cancellation_requested(
    const std::shared_ptr<Task>& task) const {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    return task->cancel_requested || task->status == QF_TASK_CANCELLED;
}

bool TaskManager::processing_timed_out(
    const std::shared_ptr<Task>& task) const {
    if (config_->values.task_timeout_sec == 0) {
        return false;
    }
    return std::chrono::steady_clock::now() - task->submitted_at >=
           std::chrono::seconds(config_->values.task_timeout_sec);
}

void TaskManager::update_progress(const std::shared_ptr<Task>& task,
                                  int32_t items_done) {
    std::lock_guard<std::mutex> lock(tasks_mutex_);
    if (terminal(task->status)) {
        return;
    }
    task->items_done = items_done;
    task->progress = task->items_total == 0
                         ? 1.0f
                         : static_cast<float>(items_done) /
                               static_cast<float>(task->items_total);
}

void TaskManager::complete_success(const std::shared_ptr<Task>& task,
                                   std::string path,
                                   std::string json) {
    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        if (task->cancel_requested || terminal(task->status)) {
            return;
        }
        task->result_path = std::move(path);
        task->result_json = std::move(json);
        task->items_done = task->items_total;
        task->progress = 1.0f;
        task->status = QF_TASK_SUCCESS;
        tasks_cv_.notify_all();
    }
    dispatch_callback(task);
}

void TaskManager::complete_failure(const std::shared_ptr<Task>& task,
                                   std::string error) {
    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        if (task->cancel_requested || terminal(task->status)) {
            return;
        }
        task->error = std::move(error);
        task->status = QF_TASK_FAILED;
        tasks_cv_.notify_all();
    }
    dispatch_callback(task);
}

void TaskManager::dispatch_callback(const std::shared_ptr<Task>& task) {
    const char* result_path = nullptr;
    QfTaskStatus status = QF_TASK_FAILED;
    {
        std::lock_guard<std::mutex> lock(tasks_mutex_);
        if (task->callback_dispatched || !terminal(task->status)) {
            return;
        }
        task->callback_dispatched = true;
        status = task->status;
        if (status == QF_TASK_SUCCESS) {
            result_path = task->result_path.c_str();
        }
    }

    QfCallback callback = nullptr;
    void* user_data = nullptr;
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        callback = callback_;
        user_data = callback_user_data_;
    }
    if (callback != nullptr) {
        ContextGuard callback_guard(callback_context, this);
        try {
            callback(result_path, task->id, status, user_data);
        } catch (...) {
            // User callbacks are an ABI boundary. Their failures must not
            // escape into task processing or alter the published terminal state.
        }
    }
}

void TaskManager::worker_loop() {
    ContextGuard worker_guard(worker_context, this);
    while (true) {
        std::shared_ptr<Task> task;
        {
            std::unique_lock<std::mutex> lock(tasks_mutex_);
            tasks_cv_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
            if (stopping_ && pending_.empty()) {
                return;
            }
            task = pending_.front();
            pending_.pop_front();
            if (task->status == QF_TASK_CANCELLED) {
                continue;
            }
            task->status = QF_TASK_RUNNING;
        }
        process_task(task);
    }
}

void TaskManager::process_task(const std::shared_ptr<Task>& task) {
    try {
        TaskResult task_result;
        task_result.items.resize(task->items.size());
        std::vector<RoiImage> images;
        std::vector<std::size_t> inferred_indices;
        images.reserve(task->items.size());
        inferred_indices.reserve(task->items.size());

        int32_t completed = 0;
        for (std::size_t index = 0; index < task->items.size(); ++index) {
            if (cancellation_requested(task)) {
                return;
            }
            if (processing_timed_out(task)) {
                complete_failure(task, "task processing timed out");
                return;
            }
            const CompoundData& compound = task->items[index];
            const SignalQc qc = inspect_signal(compound, config_->values);
            if (!qc.accepted) {
                task_result.items[index] = low_intensity_result(compound, qc);
                update_progress(task, ++completed);
                continue;
            }
            const float sigma = compound.smooth_sigma > 0.0f
                                    ? compound.smooth_sigma
                                    : config_->values.smooth_sigma;
            images.push_back(render_roi(compound, sigma));
            inferred_indices.push_back(index);
        }

        if (cancellation_requested(task)) {
            return;
        }
        const auto detections = inference_.infer_batch(images, config_->values.threshold);
        if (detections.size() != images.size()) {
            throw std::runtime_error("inference result count does not match ROI count");
        }
        for (std::size_t inferred = 0; inferred < inferred_indices.size(); ++inferred) {
            if (cancellation_requested(task)) {
                return;
            }
            if (processing_timed_out(task)) {
                complete_failure(task, "task processing timed out");
                return;
            }
            const std::size_t item_index = inferred_indices[inferred];
            task_result.items[item_index] = detection_result(
                task->items[item_index], images[inferred], detections[inferred]);
            update_progress(task, ++completed);
        }

        const std::string output = generate_output_json(task_result);
        const fs::path result_path = fs::path(config_->work_dir) /
                                     ("result_" + std::to_string(task->id) + ".json");
        std::ofstream file(result_path, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            complete_failure(task, "failed to open result file for writing");
            return;
        }
        file.write(output.data(), static_cast<std::streamsize>(output.size()));
        file.close();
        if (!file) {
            complete_failure(task, "failed to write result file");
            return;
        }
        if (cancellation_requested(task)) {
            std::error_code ignored;
            fs::remove(result_path, ignored);
            return;
        }
        complete_success(task, result_path.string(), output);
    } catch (const std::exception& exception) {
        complete_failure(task, exception.what());
    } catch (...) {
        complete_failure(task, "unknown task processing failure");
    }
}

}  // namespace mrmpformer

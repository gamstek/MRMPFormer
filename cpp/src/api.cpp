#include "mrmpformer.h"

#include "json_protocol.h"
#include "task_manager.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

enum class Lifecycle { uninitialized, initialized, shutting_down };

std::mutex lifecycle_mutex;
Lifecycle lifecycle = Lifecycle::uninitialized;
std::shared_ptr<mrmpformer::TaskManager> manager;
thread_local std::string last_error;
thread_local const char* fallback_error = nullptr;

constexpr const char* boundary_error =
    "unhandled C++ exception at MRMPFormer C API boundary";

QfError fail(QfError code, std::string message) {
    last_error = std::move(message);
    fallback_error = nullptr;
    return code;
}

void clear_error() {
    last_error.clear();
    fallback_error = nullptr;
}

void set_boundary_error(const char* function, std::string_view detail) noexcept {
    fallback_error = boundary_error;
    try {
        last_error = function;
        last_error += ": ";
        last_error.append(detail.data(), detail.size());
        fallback_error = nullptr;
    } catch (...) {
        last_error.clear();
    }
}

template <typename Function>
QfError guard_error(const char* function, Function&& operation) noexcept {
    try {
        return operation();
    } catch (const std::exception& exception) {
        set_boundary_error(function, exception.what());
    } catch (...) {
        set_boundary_error(function, "unknown exception");
    }
    return QF_ERR_INTERNAL;
}

template <typename Function>
void guard_void(const char* function, Function&& operation) noexcept {
    try {
        operation();
    } catch (const std::exception& exception) {
        set_boundary_error(function, exception.what());
    } catch (...) {
        set_boundary_error(function, "unknown exception");
    }
}

template <typename Result, typename Function>
Result guard_value(const char* function,
                   Result failure_value,
                   Function&& operation) noexcept {
    try {
        return operation();
    } catch (const std::exception& exception) {
        set_boundary_error(function, exception.what());
    } catch (...) {
        set_boundary_error(function, "unknown exception");
    }
    return failure_value;
}

std::shared_ptr<mrmpformer::TaskManager> current_manager() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex);
    if (lifecycle != Lifecycle::initialized || manager == nullptr) {
        fail(QF_ERR_NOT_INITIALIZED, "MRMPFormer is not initialized");
        return nullptr;
    }
    return manager;
}

QfError manager_result(QfError result, const std::string& error) {
    return result == QF_OK ? QF_OK : fail(result, error);
}

bool path_is_missing(const std::error_code& error) {
    return error == std::errc::no_such_file_or_directory ||
           error == std::errc::not_a_directory;
}

bool valid_config(const QfConfig& config, std::string* error) {
    if (config.model_path == nullptr || config.model_path[0] == '\0') {
        *error = "model_path is required";
        return false;
    }
    if (config.max_workers < 0) {
        *error = "max_workers must be non-negative";
        return false;
    }
    if (!std::isfinite(config.threshold) || config.threshold < 0.0f ||
        config.threshold > 1.0f) {
        *error = "threshold must be finite and in [0, 1]";
        return false;
    }
    if (!std::isfinite(config.smooth_sigma) || config.smooth_sigma < 0.0f) {
        *error = "smooth_sigma must be finite and non-negative";
        return false;
    }
    if (config.task_timeout_sec < 0) {
        *error = "task_timeout_sec must be non-negative";
        return false;
    }
    if (config.use_gpu < -1 || config.use_gpu > 1) {
        *error = "use_gpu must be -1, 0, or 1";
        return false;
    }
    if (config.batch_size <= 0) {
        *error = "batch_size must be positive";
        return false;
    }
    if (config.min_chrom_points < 0) {
        *error = "min_chrom_points must be non-negative";
        return false;
    }
    if (!std::isfinite(config.min_max_intensity) ||
        config.min_max_intensity < 0.0f) {
        *error = "min_max_intensity must be finite and non-negative";
        return false;
    }
    return true;
}

QfError copy_single_input(const QfCompoundInput& input,
                          CompoundData* compound) {
    if (input.uid == nullptr) {
        return fail(QF_ERR_INVALID_PARAM, "input uid is required");
    }
    if (input.channel == nullptr || input.channel[0] == '\0') {
        return fail(QF_ERR_INVALID_PARAM,
                    "input channel is required and must not be empty");
    }
    if (input.rt == nullptr || input.intensity == nullptr) {
        return fail(QF_ERR_INVALID_PARAM,
                    "input retention-time and intensity arrays are required");
    }
    if (input.n_points < 2) {
        return fail(QF_ERR_INVALID_PARAM,
                    "input arrays must contain at least two points");
    }
    if (!std::isfinite(input.mzq1) || !std::isfinite(input.mzq3) ||
        !std::isfinite(input.smooth_sigma)) {
        return fail(QF_ERR_INVALID_PARAM, "input numeric fields must be finite");
    }

    compound->uid = input.uid;
    compound->name = input.name == nullptr ? "" : input.name;
    compound->channel = input.channel;
    compound->mzq1 = input.mzq1;
    compound->mzq3 = input.mzq3;
    compound->smooth_sigma = input.smooth_sigma;
    compound->x.assign(input.rt, input.rt + input.n_points);
    compound->y.assign(input.intensity, input.intensity + input.n_points);
    for (int32_t index = 0; index < input.n_points; ++index) {
        if (!std::isfinite(compound->x[static_cast<std::size_t>(index)]) ||
            !std::isfinite(compound->y[static_cast<std::size_t>(index)])) {
            return fail(QF_ERR_INVALID_PARAM, "input arrays must be finite");
        }
        if (index > 0 && compound->x[static_cast<std::size_t>(index)] <=
                             compound->x[static_cast<std::size_t>(index - 1)]) {
            return fail(QF_ERR_INVALID_PARAM,
                        "input retention times must be strictly increasing");
        }
    }
    return QF_OK;
}

}  // namespace

extern "C" {

static void qf_default_config_impl(QfConfig* config) {
    if (config == nullptr) {
        return;
    }
    *config = {nullptr, "./mrmpformer_results/", nullptr, 0, 0.5f, 0.8f,
               300, -1, 128, 10, 1000.0f};
}

static QfError qf_init_impl(const QfConfig* config) {
    clear_error();
    std::lock_guard<std::mutex> lock(lifecycle_mutex);
    if (lifecycle != Lifecycle::uninitialized) {
        return fail(QF_ERR_ALREADY_INITIALIZED,
                    "MRMPFormer is already initialized");
    }
    if (config == nullptr) {
        return fail(QF_ERR_INVALID_PARAM, "config is null");
    }
    std::string validation_error;
    if (!valid_config(*config, &validation_error)) {
        return fail(QF_ERR_INVALID_PARAM, std::move(validation_error));
    }
    std::error_code filesystem_error;
    const bool regular_model =
        fs::is_regular_file(config->model_path, filesystem_error);
    if (filesystem_error) {
        if (path_is_missing(filesystem_error)) {
            return fail(QF_ERR_FILE_NOT_FOUND, "model file was not found");
        }
        return fail(QF_ERR_FILE_READ,
                    "failed to inspect model file: " + filesystem_error.message());
    }
    if (!regular_model) {
        return fail(QF_ERR_FILE_NOT_FOUND, "model file was not found");
    }
    try {
        auto created = std::make_shared<mrmpformer::TaskManager>(*config);
        std::string start_error;
        const QfError start_result = created->start(&start_error);
        if (start_result != QF_OK) {
            return fail(start_result, std::move(start_error));
        }
        manager = std::move(created);
        lifecycle = Lifecycle::initialized;
        return QF_OK;
    } catch (const std::bad_alloc&) {
        return fail(QF_ERR_INTERNAL, "memory allocation failed during initialization");
    } catch (const std::exception& exception) {
        return fail(QF_ERR_INTERNAL, exception.what());
    }
}

static QfError qf_shutdown_impl(void) {
    clear_error();
    std::shared_ptr<mrmpformer::TaskManager> shutting_down;
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        if (lifecycle != Lifecycle::initialized || manager == nullptr) {
            return fail(QF_ERR_NOT_INITIALIZED, "MRMPFormer is not initialized");
        }
        if (manager->is_worker_or_callback_context()) {
            return fail(
                QF_ERR_INVALID_PARAM,
                "qf_shutdown cannot be called from a worker or callback context");
        }
        lifecycle = Lifecycle::shutting_down;
        shutting_down = manager;
    }
    shutting_down->shutdown();
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        manager.reset();
        lifecycle = Lifecycle::uninitialized;
    }
    return QF_OK;
}

static QfError qf_process_impl(const char* json_path, int64_t* out_task_id) {
    clear_error();
    auto active = current_manager();
    if (active == nullptr) {
        return QF_ERR_NOT_INITIALIZED;
    }
    if (json_path == nullptr || json_path[0] == '\0') {
        return fail(QF_ERR_INVALID_PARAM, "json_path is required");
    }
    if (out_task_id == nullptr) {
        return fail(QF_ERR_INVALID_PARAM, "out_task_id is null");
    }
    std::error_code filesystem_error;
    const bool regular_input = fs::is_regular_file(json_path, filesystem_error);
    if (filesystem_error) {
        if (path_is_missing(filesystem_error)) {
            return fail(QF_ERR_FILE_NOT_FOUND, "input JSON file was not found");
        }
        return fail(QF_ERR_FILE_READ,
                    "failed to inspect input JSON file: " + filesystem_error.message());
    }
    if (!regular_input) {
        return fail(QF_ERR_FILE_NOT_FOUND, "input JSON file was not found");
    }
    std::ifstream input(json_path, std::ios::binary);
    if (!input.is_open()) {
        return fail(QF_ERR_FILE_READ, "failed to open input JSON file");
    }
    const std::string contents{std::istreambuf_iterator<char>(input),
                               std::istreambuf_iterator<char>()};
    if (input.bad()) {
        return fail(QF_ERR_FILE_READ, "failed to read input JSON file");
    }
    ParseResult parsed = parse_input_json(contents);
    if (!parsed.ok) {
        return fail(QF_ERR_JSON_PARSE, std::move(parsed.error));
    }
    std::string error;
    return manager_result(active->submit(std::move(parsed.items), out_task_id, &error),
                          error);
}

static QfError qf_process_single_impl(const QfCompoundInput* input,
                                      int64_t* out_task_id) {
    clear_error();
    auto active = current_manager();
    if (active == nullptr) {
        return QF_ERR_NOT_INITIALIZED;
    }
    if (input == nullptr) {
        return fail(QF_ERR_INVALID_PARAM, "input is null");
    }
    if (out_task_id == nullptr) {
        return fail(QF_ERR_INVALID_PARAM, "out_task_id is null");
    }
    CompoundData compound;
    const QfError copied = copy_single_input(*input, &compound);
    if (copied != QF_OK) {
        return copied;
    }
    std::vector<CompoundData> items;
    items.push_back(std::move(compound));
    std::string error;
    return manager_result(active->submit(std::move(items), out_task_id, &error), error);
}

static QfError qf_stop_impl(int64_t task_id) {
    clear_error();
    auto active = current_manager();
    if (active == nullptr) return QF_ERR_NOT_INITIALIZED;
    std::string error;
    return manager_result(active->stop(task_id, &error), error);
}

static QfError qf_wait_impl(int64_t task_id, int32_t timeout_ms) {
    clear_error();
    auto active = current_manager();
    if (active == nullptr) return QF_ERR_NOT_INITIALIZED;
    if (timeout_ms < 0) {
        return fail(QF_ERR_INVALID_PARAM, "timeout_ms must be non-negative");
    }
    std::string error;
    return manager_result(active->wait(task_id, timeout_ms, &error), error);
}

static QfError qf_query_impl(int64_t task_id,
                             QfTaskStatus* out_status,
                             float* out_progress,
                             int32_t* out_items_done,
                             int32_t* out_items_total) {
    clear_error();
    auto active = current_manager();
    if (active == nullptr) return QF_ERR_NOT_INITIALIZED;
    if (out_status == nullptr && out_progress == nullptr &&
        out_items_done == nullptr && out_items_total == nullptr) {
        return fail(QF_ERR_INVALID_PARAM, "at least one query output is required");
    }
    std::string error;
    return manager_result(active->query(task_id, out_status, out_progress,
                                        out_items_done, out_items_total, &error),
                          error);
}

static QfError qf_set_callback_impl(QfCallback callback, void* user_data) {
    clear_error();
    auto active = current_manager();
    if (active == nullptr) return QF_ERR_NOT_INITIALIZED;
    return active->set_callback(callback, user_data);
}

static QfError qf_get_result_path_impl(int64_t task_id,
                                       char* out_path,
                                       int32_t buf_size) {
    clear_error();
    auto active = current_manager();
    if (active == nullptr) return QF_ERR_NOT_INITIALIZED;
    if (out_path == nullptr || buf_size <= 0) {
        return fail(QF_ERR_INVALID_PARAM, "result path buffer is invalid");
    }
    std::string path;
    std::string error;
    const QfError result = active->result_path(task_id, &path, &error);
    if (result != QF_OK) return fail(result, std::move(error));
    if (path.size() + 1 > static_cast<std::size_t>(buf_size)) {
        return fail(QF_ERR_INVALID_PARAM, "result path buffer is too small");
    }
    std::memcpy(out_path, path.c_str(), path.size() + 1);
    return QF_OK;
}

static QfError qf_get_result_json_impl(int64_t task_id, char** out_json) {
    clear_error();
    auto active = current_manager();
    if (active == nullptr) return QF_ERR_NOT_INITIALIZED;
    if (out_json == nullptr) {
        return fail(QF_ERR_INVALID_PARAM, "out_json is null");
    }
    *out_json = nullptr;
    std::string json;
    std::string error;
    const QfError result = active->result_json(task_id, &json, &error);
    if (result != QF_OK) return fail(result, std::move(error));
    auto* allocation = static_cast<char*>(std::malloc(json.size() + 1));
    if (allocation == nullptr) {
        return fail(QF_ERR_INTERNAL, "failed to allocate result JSON");
    }
    std::memcpy(allocation, json.c_str(), json.size() + 1);
    *out_json = allocation;
    return QF_OK;
}

static int32_t qf_is_gpu_enabled_impl(void) {
    std::lock_guard<std::mutex> lock(lifecycle_mutex);
    return lifecycle == Lifecycle::initialized && manager != nullptr &&
                   manager->is_gpu_enabled()
               ? 1
               : 0;
}

void qf_default_config(QfConfig* config) {
    guard_void("qf_default_config", [&] {
        clear_error();
        qf_default_config_impl(config);
    });
}

QfError qf_init(const QfConfig* config) {
    return guard_error("qf_init", [&] { return qf_init_impl(config); });
}

QfError qf_shutdown(void) {
    return guard_error("qf_shutdown", [] { return qf_shutdown_impl(); });
}

QfError qf_process(const char* json_path, int64_t* out_task_id) {
    return guard_error(
        "qf_process", [&] { return qf_process_impl(json_path, out_task_id); });
}

QfError qf_process_single(const QfCompoundInput* input, int64_t* out_task_id) {
    return guard_error("qf_process_single", [&] {
        return qf_process_single_impl(input, out_task_id);
    });
}

QfError qf_stop(int64_t task_id) {
    return guard_error("qf_stop", [&] { return qf_stop_impl(task_id); });
}

QfError qf_wait(int64_t task_id, int32_t timeout_ms) {
    return guard_error(
        "qf_wait", [&] { return qf_wait_impl(task_id, timeout_ms); });
}

QfError qf_query(int64_t task_id,
                 QfTaskStatus* out_status,
                 float* out_progress,
                 int32_t* out_items_done,
                 int32_t* out_items_total) {
    return guard_error("qf_query", [&] {
        return qf_query_impl(task_id,
                             out_status,
                             out_progress,
                             out_items_done,
                             out_items_total);
    });
}

QfError qf_set_callback(QfCallback callback, void* user_data) {
    return guard_error(
        "qf_set_callback",
        [&] { return qf_set_callback_impl(callback, user_data); });
}

QfError qf_get_result_path(int64_t task_id,
                           char* out_path,
                           int32_t buf_size) {
    return guard_error("qf_get_result_path", [&] {
        return qf_get_result_path_impl(task_id, out_path, buf_size);
    });
}

QfError qf_get_result_json(int64_t task_id, char** out_json) {
    return guard_error("qf_get_result_json", [&] {
        return qf_get_result_json_impl(task_id, out_json);
    });
}

void qf_free(void* ptr) {
    guard_void("qf_free", [&] { std::free(ptr); });
}

const char* qf_get_error(void) {
    return guard_value<const char*>("qf_get_error", boundary_error, [] {
        if (fallback_error != nullptr) {
            return fallback_error;
        }
        return last_error.empty() ? nullptr : last_error.c_str();
    });
}

const char* qf_version(void) {
    return guard_value<const char*>(
        "qf_version", boundary_error, [] { return "0.2.0"; });
}

int32_t qf_is_gpu_enabled(void) {
    return guard_value<int32_t>(
        "qf_is_gpu_enabled", 0, [] { return qf_is_gpu_enabled_impl(); });
}

}  // extern "C"

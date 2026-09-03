#include "mrmpformer.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <json.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef MRMPFORMER_TEST_MODEL_PATH
#error MRMPFORMER_TEST_MODEL_PATH must be defined
#endif

namespace {

namespace fs = std::filesystem;
using nlohmann::json;

struct CallbackRecord {
    std::atomic<int> calls{0};
    std::atomic<int64_t> task_id{0};
    std::atomic<int32_t> status{-1};
    std::atomic<int32_t> query_result{QF_ERR_INTERNAL};
    std::atomic<int32_t> queried_status{QF_TASK_PENDING};
    std::mutex mutex;
    std::string path;
};

struct ShutdownCallbackRecord {
    std::atomic<int> calls{0};
    std::atomic<int32_t> shutdown_result{QF_ERR_INTERNAL};
    std::atomic<bool> exception_escaped{false};
    std::mutex mutex;
    std::string error;
};

struct ThrowingCallbackRecord {
    std::atomic<int> calls{0};
};

void record_callback(const char* path,
                     int64_t task_id,
                     int32_t status,
                     void* user_data) {
    auto* record = static_cast<CallbackRecord*>(user_data);
    {
        std::lock_guard<std::mutex> lock(record->mutex);
        record->path = path == nullptr ? "" : path;
    }
    record->task_id.store(task_id);
    record->status.store(status);
    QfTaskStatus task_status = QF_TASK_PENDING;
    record->query_result.store(qf_query(task_id, &task_status, nullptr, nullptr, nullptr));
    record->queried_status.store(task_status);
    record->calls.fetch_add(1);
}

void shutdown_from_callback(const char*, int64_t, int32_t, void* user_data) {
    auto* record = static_cast<ShutdownCallbackRecord*>(user_data);
    try {
        record->shutdown_result.store(qf_shutdown());
        const char* error = qf_get_error();
        {
            std::lock_guard<std::mutex> lock(record->mutex);
            record->error = error == nullptr ? "" : error;
        }
    } catch (...) {
        record->exception_escaped.store(true);
    }
    record->calls.fetch_add(1);
}

void throwing_callback(const char*, int64_t, int32_t, void* user_data) {
    auto* record = static_cast<ThrowingCallbackRecord*>(user_data);
    record->calls.fetch_add(1);
    throw std::runtime_error("user callback failure");
}

std::vector<double> rt_axis() {
    std::vector<double> values;
    for (int i = 0; i <= 100; ++i) {
        values.push_back(i / 10.0);
    }
    return values;
}

std::vector<double> double_peak_signal() {
    std::vector<double> values;
    for (double x : rt_axis()) {
        values.push_back(
            1000.0 + 8000.0 * std::exp(-0.5 * (x - 3.0) * (x - 3.0) / 0.09) +
            7000.0 * std::exp(-0.5 * (x - 7.0) * (x - 7.0) / 0.09));
    }
    return values;
}

std::vector<double> gaussian_signal() {
    std::vector<double> values;
    for (double x : rt_axis()) {
        values.push_back(
            10000.0 * std::exp(-0.5 * (x - 5.0) * (x - 5.0) / 0.25));
    }
    return values;
}

json compound_json(const std::string& uid,
                   const std::vector<double>& intensity) {
    return {
        {"uid", uid},
        {"name", uid + " name"},
        {"channel", "100.0>50.0"},
        {"mzq1", 100.0},
        {"mzq3", 50.0},
        {"x", rt_axis()},
        {"y", intensity},
    };
}

fs::path write_json(const fs::path& path, const json& value) {
    std::ofstream output(path, std::ios::binary);
    assert(output.is_open());
    output << value.dump(2);
    assert(output.good());
    return path;
}

std::string read_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    assert(input.is_open());
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

void wait_success(int64_t task_id, int32_t expected_total) {
    assert(qf_wait(task_id, 30000) == QF_OK);
    QfTaskStatus status = QF_TASK_PENDING;
    float progress = -1.0f;
    int32_t done = -1;
    int32_t total = -1;
    assert(qf_query(task_id, &status, &progress, &done, &total) == QF_OK);
    assert(status == QF_TASK_SUCCESS);
    assert(progress == 1.0f);
    assert(done == expected_total);
    assert(total == expected_total);
}

json result_json(int64_t task_id, std::string* path_text = nullptr) {
    char path[1024]{};
    assert(qf_get_result_path(task_id, path, sizeof(path)) == QF_OK);
    assert(fs::is_regular_file(path));

    char* result = nullptr;
    assert(qf_get_result_json(task_id, &result) == QF_OK);
    assert(result != nullptr);
    const std::string owned_result(result);
    qf_free(result);

    assert(read_file(path) == owned_result);
    if (path_text != nullptr) {
        *path_text = path;
    }
    return json::parse(owned_result);
}

void test_calls_before_initialization_are_rejected() {
    int64_t task_id = 0;
    QfTaskStatus status = QF_TASK_PENDING;
    float progress = 0.0f;
    int32_t done = 0;
    int32_t total = 0;
    char path[16]{};
    char* output = nullptr;

    assert(qf_shutdown() == QF_ERR_NOT_INITIALIZED);
    assert(qf_process(nullptr, &task_id) == QF_ERR_NOT_INITIALIZED);
    assert(qf_process_single(nullptr, &task_id) == QF_ERR_NOT_INITIALIZED);
    assert(qf_stop(1) == QF_ERR_NOT_INITIALIZED);
    assert(qf_wait(1, 1) == QF_ERR_NOT_INITIALIZED);
    assert(qf_query(1, &status, &progress, &done, &total) ==
           QF_ERR_NOT_INITIALIZED);
    assert(qf_set_callback(nullptr, nullptr) == QF_ERR_NOT_INITIALIZED);
    assert(qf_get_result_path(1, path, sizeof(path)) ==
           QF_ERR_NOT_INITIALIZED);
    assert(qf_get_result_json(1, &output) == QF_ERR_NOT_INITIALIZED);
}

void test_init_validates_and_deep_copies_config(const fs::path& work_dir) {
    assert(qf_init(nullptr) == QF_ERR_INVALID_PARAM);

    std::string missing_path = "missing-mrmpformer-model.onnx";
    QfConfig missing{};
    qf_default_config(&missing);
    missing.model_path = missing_path.c_str();
    assert(qf_init(&missing) == QF_ERR_FILE_NOT_FOUND);

    const fs::path blocked_work_dir = work_dir / "not-a-directory";
    {
        std::ofstream blocker(blocked_work_dir);
        blocker << "file blocks directory creation";
    }
    QfConfig unwritable{};
    qf_default_config(&unwritable);
    unwritable.model_path = MRMPFORMER_TEST_MODEL_PATH;
    const std::string blocked_path = blocked_work_dir.string();
    unwritable.work_dir = blocked_path.c_str();
    unwritable.max_workers = 1;
    unwritable.use_gpu = -1;
    assert(qf_init(&unwritable) == QF_ERR_FILE_WRITE);
    fs::remove(blocked_work_dir);

    std::string model_path = MRMPFORMER_TEST_MODEL_PATH;
    std::string work_path = work_dir.string();
    QfConfig config{};
    qf_default_config(&config);
    config.model_path = model_path.c_str();
    config.work_dir = work_path.c_str();
    config.max_workers = 1;
    config.threshold = 0.6f;
    config.task_timeout_sec = 0;
    config.use_gpu = -1;
    config.batch_size = 8;
    std::atomic<int32_t> first_result{QF_ERR_INTERNAL};
    std::atomic<int32_t> second_result{QF_ERR_INTERNAL};
    std::thread first([&] { first_result.store(qf_init(&config)); });
    std::thread second([&] { second_result.store(qf_init(&config)); });
    first.join();
    second.join();
    assert((first_result.load() == QF_OK &&
            second_result.load() == QF_ERR_ALREADY_INITIALIZED) ||
           (second_result.load() == QF_OK &&
            first_result.load() == QF_ERR_ALREADY_INITIALIZED));
    assert(qf_init(&config) == QF_ERR_ALREADY_INITIALIZED);

    model_path.assign("caller-mutated-model");
    work_path.assign("caller-mutated-work-dir");
}

void test_initialized_argument_and_batch_validation(const fs::path& work_dir) {
    int64_t task_id = 0;
    assert(qf_process(nullptr, &task_id) == QF_ERR_INVALID_PARAM);
    assert(qf_process("missing-input.json", nullptr) == QF_ERR_INVALID_PARAM);
    assert(qf_process("missing-input.json", &task_id) == QF_ERR_FILE_NOT_FOUND);
    assert(qf_process_single(nullptr, &task_id) == QF_ERR_INVALID_PARAM);

    QfCompoundInput input{};
    assert(qf_process_single(&input, nullptr) == QF_ERR_INVALID_PARAM);
    assert(qf_process_single(&input, &task_id) == QF_ERR_INVALID_PARAM);

    assert(qf_query(1, nullptr, nullptr, nullptr, nullptr) ==
           QF_ERR_INVALID_PARAM);
    assert(qf_get_result_path(1, nullptr, 1) == QF_ERR_INVALID_PARAM);
    char result_path[1]{};
    assert(qf_get_result_path(1, result_path, 0) == QF_ERR_INVALID_PARAM);
    assert(qf_get_result_json(1, nullptr) == QF_ERR_INVALID_PARAM);
    assert(qf_wait(1, -1) == QF_ERR_INVALID_PARAM);

    const fs::path malformed = write_json(
        work_dir / "malformed-input.json", json{{"items", {{{"uid", "bad"}}}}});
    task_id = 777;
    assert(qf_process(malformed.string().c_str(), &task_id) == QF_ERR_JSON_PARSE);
    assert(task_id == 777);

    const double bad_rt[] = {1.0, 1.0};
    const double intensity[] = {2000.0, 3000.0};
    input.uid = "bad-single";
    input.channel = "100>50";
    input.mzq1 = 100.0;
    input.mzq3 = 50.0;
    input.rt = bad_rt;
    input.intensity = intensity;
    input.n_points = 2;
    assert(qf_process_single(&input, &task_id) == QF_ERR_INVALID_PARAM);

    input.rt = intensity;
    input.mzq1 = std::numeric_limits<double>::infinity();
    assert(qf_process_single(&input, &task_id) == QF_ERR_INVALID_PARAM);
}

void test_mixed_batch_callback_and_result_ownership(const fs::path& work_dir) {
    const std::vector<double> low_intensity(rt_axis().size(), 100.0);
    const json input = {{"items",
                         {compound_json("normal", double_peak_signal()),
                          compound_json("fallback", gaussian_signal()),
                          compound_json("low", low_intensity)}}};
    const fs::path input_path = write_json(work_dir / "mixed-input.json", input);

    CallbackRecord callback;
    assert(qf_set_callback(record_callback, &callback) == QF_OK);

    int64_t task_id = 0;
    assert(qf_process(input_path.string().c_str(), &task_id) == QF_OK);
    assert(task_id > 0);
    fs::remove(input_path);
    wait_success(task_id, 3);

    for (int attempt = 0; attempt < 100 && callback.calls.load() == 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    assert(callback.calls.load() == 1);
    assert(callback.task_id.load() == task_id);
    assert(callback.status.load() == QF_TASK_SUCCESS);
    assert(callback.query_result.load() == QF_OK);
    assert(callback.queried_status.load() == QF_TASK_SUCCESS);

    std::string result_path;
    const json output = result_json(task_id, &result_path);
    assert(output["items"].size() == 3);
    assert(output["items"][0]["uid"] == "normal");
    assert(output["items"][0]["status"] == "ok");
    assert(!output["items"][0]["peaks"].empty());
    assert(output["items"][0]["alerts"].empty());

    const json& fallback = output["items"][1];
    assert(fallback["uid"] == "fallback");
    assert(fallback["status"] == "review");
    assert(fallback["peaks"].size() == 1);
    assert(fallback["alerts"].size() == 1);
    assert(fallback["alerts"][0]["level"] == "review");
    assert(fallback["alerts"][0]["code"] == "SIGNAL_FALLBACK");
    for (const char* field : {"a", "b", "c"}) {
        assert(fallback["alerts"][0]["detail"][field] ==
               fallback["peaks"][0][field]);
    }

    const json& low = output["items"][2];
    assert(low["uid"] == "low");
    assert(low["status"] == "alert");
    assert(low["peaks"].empty());
    assert(low["alerts"].size() == 1);
    assert(low["alerts"][0]["level"] == "fail");
    assert(low["alerts"][0]["code"] == "CHANNEL_LOW_INTENSITY");
    assert(low["alerts"][0]["detail"]["n_points"] == 101.0);
    assert(low["alerts"][0]["detail"]["max_intensity"] == 100.0);

    {
        std::lock_guard<std::mutex> lock(callback.mutex);
        assert(callback.path == result_path);
    }
    assert(fs::is_regular_file(result_path));
    assert(result_path.find(work_dir.string()) == 0);
    assert(qf_set_callback(nullptr, nullptr) == QF_OK);
}

void test_single_submission_copies_caller_memory_without_temp_json(
    const fs::path& work_dir) {
    std::size_t files_before = 0;
    for (const auto& entry : fs::directory_iterator(work_dir)) {
        if (entry.is_regular_file()) {
            ++files_before;
        }
    }
    std::string uid = "owned-single";
    std::string name = "Owned Single";
    std::string channel = "100>50";
    std::vector<double> rt = rt_axis();
    std::vector<double> intensity = gaussian_signal();

    QfCompoundInput input{};
    input.uid = uid.c_str();
    input.name = name.c_str();
    input.channel = channel.c_str();
    input.mzq1 = 100.0;
    input.mzq3 = 50.0;
    input.rt = rt.data();
    input.intensity = intensity.data();
    input.n_points = static_cast<int32_t>(rt.size());

    int64_t task_id = 0;
    assert(qf_process_single(&input, &task_id) == QF_OK);
    uid.assign("mutated");
    name.clear();
    channel.clear();
    std::fill(rt.begin(), rt.end(), -1.0);
    std::fill(intensity.begin(), intensity.end(), 0.0);

    wait_success(task_id, 1);
    const json output = result_json(task_id);
    assert(output["items"][0]["uid"] == "owned-single");

    std::size_t files_after = 0;
    for (const auto& entry : fs::directory_iterator(work_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        ++files_after;
        const std::string filename = entry.path().filename().string();
        assert(filename.find("input_owned-single") == std::string::npos);
        assert(filename.find("qf_input_owned-single") == std::string::npos);
    }
    assert(files_after == files_before + 1);
}

void test_fallback_failure_uses_no_peak_found() {
    const std::vector<double> rt = rt_axis();
    const std::vector<double> flat(rt.size(), 2000.0);
    QfCompoundInput input{};
    input.uid = "flat";
    input.channel = "100>50";
    input.mzq1 = 100.0;
    input.mzq3 = 50.0;
    input.rt = rt.data();
    input.intensity = flat.data();
    input.n_points = static_cast<int32_t>(rt.size());

    int64_t task_id = 0;
    assert(qf_process_single(&input, &task_id) == QF_OK);
    wait_success(task_id, 1);
    const json output = result_json(task_id);
    const json& item = output["items"][0];
    assert(item["status"] == "alert");
    assert(item["peaks"].empty());
    assert(item["alerts"][0]["level"] == "fail");
    assert(item["alerts"][0]["code"] == "NO_PEAK_FOUND");
}

void test_concurrent_submissions_allocate_unique_increasing_ids() {
    constexpr int task_count = 8;
    const std::vector<double> rt = rt_axis();
    const std::vector<double> low(rt.size(), 100.0);
    std::vector<int64_t> task_ids(task_count, 0);
    std::vector<std::thread> submitters;
    submitters.reserve(task_count);
    for (int index = 0; index < task_count; ++index) {
        submitters.emplace_back([&, index] {
            const std::string uid = "concurrent-" + std::to_string(index);
            QfCompoundInput input{};
            input.uid = uid.c_str();
            input.channel = "100>50";
            input.mzq1 = 100.0;
            input.mzq3 = 50.0;
            input.rt = rt.data();
            input.intensity = low.data();
            input.n_points = static_cast<int32_t>(rt.size());
            assert(qf_process_single(&input, &task_ids[index]) == QF_OK);
        });
    }
    for (std::thread& submitter : submitters) {
        submitter.join();
    }
    for (int64_t task_id : task_ids) {
        assert(task_id > 0);
        wait_success(task_id, 1);
    }
    std::sort(task_ids.begin(), task_ids.end());
    assert(std::adjacent_find(task_ids.begin(), task_ids.end()) == task_ids.end());
    for (std::size_t index = 1; index < task_ids.size(); ++index) {
        assert(task_ids[index] > task_ids[index - 1]);
    }
}

void test_worker_callback_cannot_shutdown_the_library() {
    ShutdownCallbackRecord callback;
    assert(qf_set_callback(shutdown_from_callback, &callback) == QF_OK);

    const std::vector<double> rt = rt_axis();
    const std::vector<double> low(rt.size(), 100.0);
    QfCompoundInput input{};
    input.uid = "callback-shutdown";
    input.channel = "100>50";
    input.mzq1 = 100.0;
    input.mzq3 = 50.0;
    input.rt = rt.data();
    input.intensity = low.data();
    input.n_points = static_cast<int32_t>(rt.size());

    int64_t task_id = 0;
    assert(qf_process_single(&input, &task_id) == QF_OK);
    assert(qf_wait(task_id, 30000) == QF_OK);
    for (int attempt = 0; attempt < 100 && callback.calls.load() == 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    assert(callback.calls.load() == 1);
    assert(!callback.exception_escaped.load());
    assert(callback.shutdown_result.load() == QF_ERR_INVALID_PARAM);
    {
        std::lock_guard<std::mutex> lock(callback.mutex);
        assert(callback.error ==
               "qf_shutdown cannot be called from a worker or callback context");
    }

    QfTaskStatus status = QF_TASK_PENDING;
    assert(qf_query(task_id, &status, nullptr, nullptr, nullptr) == QF_OK);
    assert(status == QF_TASK_SUCCESS);
    assert(qf_set_callback(nullptr, nullptr) == QF_OK);
}

void test_throwing_callback_cannot_change_terminal_state_or_break_worker(
    const fs::path& work_dir) {
    ThrowingCallbackRecord callback;
    assert(qf_set_callback(throwing_callback, &callback) == QF_OK);

    const std::vector<double> rt = rt_axis();
    const std::vector<double> low(rt.size(), 100.0);
    QfCompoundInput input{};
    input.uid = "throwing-callback";
    input.channel = "100>50";
    input.mzq1 = 100.0;
    input.mzq3 = 50.0;
    input.rt = rt.data();
    input.intensity = low.data();
    input.n_points = static_cast<int32_t>(rt.size());

    int64_t task_id = 0;
    assert(qf_process_single(&input, &task_id) == QF_OK);
    assert(qf_wait(task_id, 30000) == QF_OK);
    for (int attempt = 0; attempt < 100 && callback.calls.load() == 0; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    assert(callback.calls.load() == 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    QfTaskStatus status = QF_TASK_PENDING;
    assert(qf_query(task_id, &status, nullptr, nullptr, nullptr) == QF_OK);
    assert(status == QF_TASK_SUCCESS);

    json blocker_items = json::array();
    for (int index = 0; index < 16; ++index) {
        blocker_items.push_back(compound_json(
            "callback-blocker-" + std::to_string(index), double_peak_signal()));
    }
    const fs::path blocker_path = write_json(
        work_dir / "callback-blocker.json",
        json{{"items", std::move(blocker_items)}});
    int64_t blocker_task_id = 0;
    assert(qf_process(blocker_path.string().c_str(), &blocker_task_id) == QF_OK);
    assert(qf_wait(blocker_task_id, 1) == QF_ERR_TIMEOUT);

    const std::vector<double> accepted = double_peak_signal();
    input.uid = "throwing-cancel-callback";
    input.intensity = accepted.data();
    int64_t cancelled_task_id = 0;
    assert(qf_process_single(&input, &cancelled_task_id) == QF_OK);
    bool exception_escaped = false;
    QfError stop_result = QF_ERR_INTERNAL;
    try {
        stop_result = qf_stop(cancelled_task_id);
    } catch (...) {
        exception_escaped = true;
    }
    assert(!exception_escaped);
    assert(stop_result == QF_OK);
    assert(qf_wait(cancelled_task_id, 30000) == QF_OK);
    assert(qf_query(cancelled_task_id, &status, nullptr, nullptr, nullptr) == QF_OK);
    assert(status == QF_TASK_CANCELLED);
    assert(callback.calls.load() >= 2);
    assert(qf_set_callback(nullptr, nullptr) == QF_OK);
    wait_success(blocker_task_id, 16);

    input.uid = "after-throwing-callback";
    input.intensity = low.data();
    int64_t later_task_id = 0;
    assert(qf_process_single(&input, &later_task_id) == QF_OK);
    wait_success(later_task_id, 1);
}

void test_timeout_does_not_cancel_and_stop_reaches_cancelled(const fs::path& work_dir) {
    json slow_items = json::array();
    for (int index = 0; index < 32; ++index) {
        slow_items.push_back(
            compound_json("slow-" + std::to_string(index), double_peak_signal()));
    }
    const fs::path slow_path = write_json(
        work_dir / "slow-input.json", json{{"items", std::move(slow_items)}});

    int64_t slow_id = 0;
    assert(qf_process(slow_path.string().c_str(), &slow_id) == QF_OK);
    assert(qf_wait(slow_id, 1) == QF_ERR_TIMEOUT);
    QfTaskStatus slow_status = QF_TASK_CANCELLED;
    assert(qf_query(slow_id, &slow_status, nullptr, nullptr, nullptr) == QF_OK);
    assert(slow_status == QF_TASK_PENDING || slow_status == QF_TASK_RUNNING);

    int64_t cancelled_id = 0;
    const std::vector<double> rt = rt_axis();
    const std::vector<double> intensity = double_peak_signal();
    QfCompoundInput queued{};
    queued.uid = "cancelled";
    queued.channel = "100>50";
    queued.mzq1 = 100.0;
    queued.mzq3 = 50.0;
    queued.rt = rt.data();
    queued.intensity = intensity.data();
    queued.n_points = static_cast<int32_t>(rt.size());
    assert(qf_process_single(&queued, &cancelled_id) == QF_OK);
    assert(cancelled_id > slow_id);
    assert(qf_stop(cancelled_id) == QF_OK);
    assert(qf_wait(cancelled_id, 1000) == QF_OK);

    QfTaskStatus status = QF_TASK_PENDING;
    float progress = -1.0f;
    int32_t done = -1;
    int32_t total = -1;
    assert(qf_query(cancelled_id, &status, &progress, &done, &total) == QF_OK);
    assert(status == QF_TASK_CANCELLED);
    assert(progress >= 0.0f && progress < 1.0f);
    assert(total == 1);
    char* cancelled_result = nullptr;
    assert(qf_get_result_json(cancelled_id, &cancelled_result) ==
           QF_ERR_TASK_CANCELLED);

    wait_success(slow_id, 32);
}

void test_error_is_thread_local_and_success_clears_it() {
    int64_t unused = 0;
    assert(qf_process("missing-again.json", &unused) == QF_ERR_FILE_NOT_FOUND);
    assert(qf_get_error() != nullptr);
    const std::string main_error = qf_get_error();

    std::string child_initial_error = "unexpected";
    std::string child_own_error;
    std::thread child([&] {
        const char* initial = qf_get_error();
        child_initial_error = initial == nullptr ? "" : initial;
        QfTaskStatus status = QF_TASK_PENDING;
        assert(qf_query(999999, &status, nullptr, nullptr, nullptr) ==
               QF_ERR_TASK_NOT_FOUND);
        child_own_error = qf_get_error() == nullptr ? "" : qf_get_error();
    });
    child.join();

    assert(child_initial_error.empty());
    assert(!child_own_error.empty());
    assert(qf_get_error() != nullptr);
    assert(main_error == qf_get_error());

    assert(qf_set_callback(nullptr, nullptr) == QF_OK);
    assert(qf_get_error() == nullptr);
}

}  // namespace

int main() {
    const fs::path work_dir =
        fs::temp_directory_path() / "mrmpformer-test-api-lifecycle";
    fs::remove_all(work_dir);
    fs::create_directories(work_dir);

    test_calls_before_initialization_are_rejected();
    test_init_validates_and_deep_copies_config(work_dir);
    test_initialized_argument_and_batch_validation(work_dir);
    test_mixed_batch_callback_and_result_ownership(work_dir);
    test_single_submission_copies_caller_memory_without_temp_json(work_dir);
    test_fallback_failure_uses_no_peak_found();
    test_concurrent_submissions_allocate_unique_increasing_ids();
    test_worker_callback_cannot_shutdown_the_library();
    test_throwing_callback_cannot_change_terminal_state_or_break_worker(work_dir);
    test_timeout_does_not_cancel_and_stop_reaches_cancelled(work_dir);
    test_error_is_thread_local_and_success_clears_it();

    assert(qf_shutdown() == QF_OK);
    assert(qf_shutdown() == QF_ERR_NOT_INITIALIZED);
    fs::remove_all(work_dir);
    return 0;
}

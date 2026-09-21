#include "mrmpformer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

static_assert(std::is_standard_layout_v<QfConfig>);
static_assert(std::is_standard_layout_v<QfCompoundInput>);

#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(QfConfig) == 56);
static_assert(offsetof(QfConfig, model_path) == 0);
static_assert(offsetof(QfConfig, work_dir) == 8);
static_assert(offsetof(QfConfig, log_file) == 16);
static_assert(offsetof(QfConfig, max_workers) == 24);
static_assert(offsetof(QfConfig, threshold) == 28);
static_assert(offsetof(QfConfig, smooth_sigma) == 32);
static_assert(offsetof(QfConfig, task_timeout_sec) == 36);
static_assert(offsetof(QfConfig, use_gpu) == 40);
static_assert(offsetof(QfConfig, batch_size) == 44);
static_assert(offsetof(QfConfig, min_chrom_points) == 48);
static_assert(offsetof(QfConfig, min_max_intensity) == 52);

static_assert(sizeof(QfCompoundInput) == 72);
static_assert(offsetof(QfCompoundInput, uid) == 0);
static_assert(offsetof(QfCompoundInput, name) == 8);
static_assert(offsetof(QfCompoundInput, channel) == 16);
static_assert(offsetof(QfCompoundInput, mzq1) == 24);
static_assert(offsetof(QfCompoundInput, mzq3) == 32);
static_assert(offsetof(QfCompoundInput, smooth_sigma) == 40);
static_assert(offsetof(QfCompoundInput, rt) == 48);
static_assert(offsetof(QfCompoundInput, intensity) == 56);
static_assert(offsetof(QfCompoundInput, n_points) == 64);

static_assert(QF_OK == 0);
static_assert(QF_ERR_INVALID_PARAM == -1);
static_assert(QF_ERR_NOT_INITIALIZED == -2);
static_assert(QF_ERR_ALREADY_INITIALIZED == -3);
static_assert(QF_ERR_FILE_NOT_FOUND == -4);
static_assert(QF_ERR_FILE_READ == -5);
static_assert(QF_ERR_FILE_WRITE == -6);
static_assert(QF_ERR_JSON_PARSE == -7);
static_assert(QF_ERR_MODEL_LOAD == -8);
static_assert(QF_ERR_TASK_NOT_FOUND == -9);
static_assert(QF_ERR_TASK_CANCELLED == -10);
static_assert(QF_ERR_TIMEOUT == -11);
static_assert(QF_ERR_NO_WORKER == -12);
static_assert(QF_ERR_INTERNAL == -99);

static_assert(QF_TASK_PENDING == 0);
static_assert(QF_TASK_RUNNING == 1);
static_assert(QF_TASK_SUCCESS == 2);
static_assert(QF_TASK_FAILED == 3);
static_assert(QF_TASK_CANCELLED == 4);
#endif

int main() {
    QfConfig defaults{};
    qf_default_config(&defaults);
    if (defaults.model_path != nullptr ||
        defaults.work_dir == nullptr ||
        std::strcmp(defaults.work_dir, "./mrmpformer_results/") != 0 ||
        defaults.log_file != nullptr ||
        defaults.max_workers != 0 ||
        defaults.threshold != 0.5f ||
        defaults.smooth_sigma != 0.8f ||
        defaults.task_timeout_sec != 300 ||
        defaults.use_gpu != -1 ||
        defaults.batch_size != 128 ||
        defaults.min_chrom_points != 10 ||
        defaults.min_max_intensity != 1000.0f) {
        return 1;
    }

    QfConfig config{};
    config.model_path = "model.onnx";
    config.work_dir = "results";
    config.log_file = "mrmpformer.log";
    config.max_workers = 1;
    config.threshold = 0.5f;
    config.smooth_sigma = 1.0f;
    config.task_timeout_sec = 30;
    config.use_gpu = -1;
    config.batch_size = 4;
    config.min_chrom_points = 10;
    config.min_max_intensity = 1000.0f;

    const double rt[] = {1.0, 2.0};
    const double intensity[] = {10.0, 20.0};
    QfCompoundInput input{};
    input.uid = "compound-1";
    input.name = "Compound 1";
    input.channel = "100.0>50.0";
    input.mzq1 = 100.0;
    input.mzq3 = 50.0;
    input.smooth_sigma = 1.0f;
    input.rt = rt;
    input.intensity = intensity;
    input.n_points = 2;

    void (*default_config)(QfConfig*) = &qf_default_config;
    QfError (*init)(const QfConfig*) = &qf_init;
    QfError (*shutdown)(void) = &qf_shutdown;
    QfError (*process)(const char*, int64_t*) = &qf_process;
    QfError (*submit)(const QfCompoundInput*, int64_t*) = &qf_process_single;
    QfError (*stop)(int64_t) = &qf_stop;
    QfError (*wait)(int64_t, int32_t) = &qf_wait;
    QfError (*query)(int64_t, QfTaskStatus*, float*, int32_t*, int32_t*) = &qf_query;
    QfError (*set_callback)(QfCallback, void*) = &qf_set_callback;
    QfError (*get_result_path)(int64_t, char*, int32_t) = &qf_get_result_path;
    QfError (*get_result_json)(int64_t, char**) = &qf_get_result_json;
    void (*free_result)(void*) = &qf_free;
    const char* (*get_error)(void) = &qf_get_error;
    const char* (*version)(void) = &qf_version;
    int32_t (*is_gpu_enabled)(void) = &qf_is_gpu_enabled;

    (void)config;
    (void)input;
    (void)default_config;
    (void)init;
    (void)shutdown;
    (void)process;
    (void)submit;
    (void)stop;
    (void)wait;
    (void)query;
    (void)set_callback;
    (void)get_result_path;
    (void)get_result_json;
    (void)free_result;
    (void)get_error;
    (void)version;
    (void)is_gpu_enabled;
    return 0;
}

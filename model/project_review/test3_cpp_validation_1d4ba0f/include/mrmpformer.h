#ifndef MRMPFORMER_H
#define MRMPFORMER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(MRMPFORMER_BUILD)
#define QF_API __declspec(dllexport)
#else
#define QF_API __declspec(dllimport)
#endif
#else
#define QF_API __attribute__((visibility("default")))
#endif

typedef enum {
    QF_OK = 0,
    QF_ERR_INVALID_PARAM = -1,
    QF_ERR_NOT_INITIALIZED = -2,
    QF_ERR_ALREADY_INITIALIZED = -3,
    QF_ERR_FILE_NOT_FOUND = -4,
    QF_ERR_FILE_READ = -5,
    QF_ERR_FILE_WRITE = -6,
    QF_ERR_JSON_PARSE = -7,
    QF_ERR_MODEL_LOAD = -8,
    QF_ERR_TASK_NOT_FOUND = -9,
    QF_ERR_TASK_CANCELLED = -10,
    QF_ERR_TIMEOUT = -11,
    QF_ERR_NO_WORKER = -12,
    QF_ERR_INTERNAL = -99,
} QfError;

typedef enum {
    QF_TASK_PENDING = 0,
    QF_TASK_RUNNING = 1,
    QF_TASK_SUCCESS = 2,
    QF_TASK_FAILED = 3,
    QF_TASK_CANCELLED = 4,
} QfTaskStatus;

typedef struct {
    /* Required path to mrmpformerv2.onnx; qf_default_config sets NULL. */
    const char* model_path;
    /* Optional result directory; default ./mrmpformer_results/. */
    const char* work_dir;
    /* Optional log-file path; default NULL disables file logging. */
    const char* log_file;
    /* Default 0 auto-selects workers; 1 is serial. */
    int32_t max_workers;
    /* Model score threshold in [0, 1]; the default is 0.5. */
    float threshold;
    /* Default 0.8 matches Python MassNova inference; 0 disables smoothing. */
    float smooth_sigma;
    /* Default 300 seconds; 0 disables the per-task timeout. */
    int32_t task_timeout_sec;
    /* Default -1 forces CPU; 0 auto-detects and 1 requires GPU inference. */
    int32_t use_gpu;
    /* Number of ROIs per inference batch; the default is 128. */
    int32_t batch_size;
    /* Minimum RT points after which a channel is accepted; default 10. */
    int32_t min_chrom_points;
    /* Minimum smoothed maximum intensity; default 1000. */
    float min_max_intensity;
} QfConfig;

typedef struct {
    const char* uid;
    const char* name;
    const char* channel;
    double mzq1;
    double mzq3;
    float smooth_sigma;
    const double* rt;
    const double* intensity;
    int32_t n_points;
} QfCompoundInput;

/* The callback receives a result path valid until qf_shutdown(). */
typedef void (*QfCallback)(const char* result_json_path,
                           int64_t task_id,
                           int32_t status,
                           void* user_data);

/* Initializes every QfConfig field with the documented default. */
QF_API void qf_default_config(QfConfig* config);

/* The library copies configuration strings during initialization. */
QF_API QfError qf_init(const QfConfig* config);
QF_API QfError qf_shutdown(void);

QF_API QfError qf_process(const char* json_path, int64_t* out_task_id);
/* The library copies all QfCompoundInput strings and arrays before returning. */
QF_API QfError qf_process_single(const QfCompoundInput* input, int64_t* out_task_id);

QF_API QfError qf_stop(int64_t task_id);
QF_API QfError qf_wait(int64_t task_id, int32_t timeout_ms);
QF_API QfError qf_query(int64_t task_id,
                         QfTaskStatus* out_status,
                         float* out_progress,
                         int32_t* out_items_done,
                         int32_t* out_items_total);

/* Replaces the process-global callback; NULL unregisters it. */
QF_API QfError qf_set_callback(QfCallback callback, void* user_data);

QF_API QfError qf_get_result_path(int64_t task_id, char* out_path, int32_t buf_size);
/* qf_get_result_json allocates with malloc; callers release with qf_free(). */
QF_API QfError qf_get_result_json(int64_t task_id, char** out_json);
QF_API void qf_free(void* ptr);

/* Returns the current thread's most recent error message, or NULL when clear. */
QF_API const char* qf_get_error(void);
QF_API const char* qf_version(void);
QF_API int32_t qf_is_gpu_enabled(void);

#ifdef __cplusplus
}
#endif

#endif

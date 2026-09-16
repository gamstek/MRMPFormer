#ifndef MRMPFORMER_H
#define MRMPFORMER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file mrmpformer.h
 * @brief MRMPFormer 推理库的 C 语言公共接口。
 *
 * 本接口采用“初始化 -> 提交异步任务 -> 等待或查询 -> 获取结果 -> 关闭”的
 * 使用流程。除 qf_default_config()、qf_get_error() 和 qf_version() 等特别说明的
 * 函数外，任务相关接口均要求先成功调用 qf_init()。
 *
 * @note 本头文件同时支持 C 和 C++。公开结构体的字段顺序及枚举数值属于 ABI
 *       的一部分，调用方不应自行改变定义。
 * @note 返回 QfError 的函数在失败后会为当前线程设置错误说明；应在同一线程中
 *       立即调用 qf_get_error() 获取详情。
 */

#if defined(_WIN32)
#if defined(MRMPFORMER_BUILD)
#define QF_API __declspec(dllexport)
#else
#define QF_API __declspec(dllimport)
#endif
#else
#define QF_API __attribute__((visibility("default")))
#endif

/** @brief 公共接口返回的错误码；负值表示失败。 */
typedef enum {
    QF_OK = 0,                       /**< 成功。 */
    QF_ERR_INVALID_PARAM = -1,       /**< 参数非法，或当前调用上下文不允许该操作。 */
    QF_ERR_NOT_INITIALIZED = -2,     /**< 库尚未初始化，或已经关闭。 */
    QF_ERR_ALREADY_INITIALIZED = -3, /**< 库已经初始化，不能重复初始化。 */
    QF_ERR_FILE_NOT_FOUND = -4,      /**< 模型文件或输入文件不存在。 */
    QF_ERR_FILE_READ = -5,           /**< 检查、打开或读取文件失败。 */
    QF_ERR_FILE_WRITE = -6,          /**< 创建结果目录或写入结果文件失败。 */
    QF_ERR_JSON_PARSE = -7,          /**< 输入 JSON 语法或数据结构不符合要求。 */
    QF_ERR_MODEL_LOAD = -8,          /**< ONNX 模型或所需推理 Provider 加载失败。 */
    QF_ERR_TASK_NOT_FOUND = -9,      /**< 指定的任务 ID 不存在。 */
    QF_ERR_TASK_CANCELLED = -10,     /**< 指定任务已经取消，无法执行所请求的操作。 */
    QF_ERR_TIMEOUT = -11,            /**< 等待任务完成时超过指定时限。 */
    QF_ERR_NO_WORKER = -12,          /**< 无法创建或启动任务工作线程。 */
    QF_ERR_INTERNAL = -99,           /**< 未预期的内部错误。 */
} QfError;

/** @brief 异步推理任务的生命周期状态。 */
typedef enum {
    QF_TASK_PENDING = 0,   /**< 已进入队列，尚未开始处理。 */
    QF_TASK_RUNNING = 1,   /**< 正在处理。 */
    QF_TASK_SUCCESS = 2,   /**< 处理成功，结果已经可以读取。 */
    QF_TASK_FAILED = 3,    /**< 处理失败。 */
    QF_TASK_CANCELLED = 4, /**< 已被 qf_stop() 取消。 */
} QfTaskStatus;

/**
 * @brief MRMPFormer 全局运行配置。
 *
 * 推荐先调用 qf_default_config() 填充全部字段，再按需覆盖。qf_init() 会复制
 * model_path、work_dir 和 log_file 的字符串内容，因此 qf_init() 返回后调用方
 * 可以释放或修改这些字符串。
 */
typedef struct {
    /**
     * @brief ONNX 模型文件路径。
     * @details 必填，必须指向可读取的 mrmpformerv2.onnx 普通文件。默认值为
     *          NULL，调用 qf_init() 前必须设置。
     */
    const char* model_path;

    /**
     * @brief 结果文件输出目录。
     * @details NULL 或空字符串均使用默认目录 `./mrmpformer_results/`；目录不存在
     *          时初始化过程会尝试创建。每个成功任务会在此生成结果 JSON 文件。
     */
    const char* work_dir;

    /** @brief 日志文件路径；默认值为 NULL，表示不写入日志文件。 */
    const char* log_file;

    /**
     * @brief 并行任务工作线程数。
     * @details 必须大于或等于 0。0 表示根据硬件自动选择（当前实现最多 4 个），
     *          1 表示串行处理任务。默认值为 0。
     */
    int32_t max_workers;

    /**
     * @brief 模型检测置信度阈值。
     * @details 必须是 [0, 1] 范围内的有限数；低于阈值的检测会被过滤。
     *          默认值为 0.99。
     */
    float threshold;

    /**
     * @brief 全局高斯平滑参数 sigma。
     * @details 必须是非负有限数。0 表示不平滑；默认值为 0。单条输入的
     *          QfCompoundInput::smooth_sigma 大于 0 时优先使用单条输入值。
     */
    float smooth_sigma;

    /**
     * @brief 单个任务的处理超时时间，单位为秒。
     * @details 从任务提交时开始计时。必须大于或等于 0；0 表示禁用处理超时，
     *          默认值为 300。该配置不同于 qf_wait() 的等待超时。
     */
    int32_t task_timeout_sec;

    /**
     * @brief 推理设备选择策略。
     * @details -1 强制 CPU；0 尝试 GPU、不可用时回退 CPU；1 要求 GPU、无法
     *          启用时初始化失败。默认值为 -1。
     */
    int32_t use_gpu;

    /**
     * @brief 每个 ONNX 推理批次包含的 ROI 数量上限。
     * @details 必须为正数；默认值为 128。该值影响吞吐量和内存占用。
     */
    int32_t batch_size;

    /**
     * @brief 接受一个色谱通道所需的最少保留时间采样点数。
     * @details 必须大于或等于 0；默认值为 10。低于此值的通道不会进入模型
     *          推理，并会在结果中产生低质量告警。
     */
    int32_t min_chrom_points;

    /**
     * @brief 接受一个色谱通道所需的最小平滑后最大强度。
     * @details 必须是非负有限数；默认值为 1000。低于此值的通道不会进入模型
     *          推理，并会在结果中产生低质量告警。
     */
    float min_max_intensity;
} QfConfig;

/**
 * @brief 直接提交单个化合物/色谱通道时使用的输入数据。
 *
 * qf_process_single() 会在返回前深拷贝所有字符串和数组；调用返回后，调用方
 * 可以立即释放或修改本结构体及其指向的数据。
 */
typedef struct {
    /** @brief 结果关联用的唯一标识；必填且不能为 NULL，可以是空字符串。 */
    const char* uid;
    /** @brief 化合物显示名称；可为 NULL，NULL 会按空字符串处理。 */
    const char* name;
    /** @brief 通道标识（例如 `100.0>50.0`）；必填且不能为空字符串。 */
    const char* channel;
    /** @brief 母离子/第一质量分析器的质荷比；必须是有限数。 */
    double mzq1;
    /** @brief 子离子/第三质量分析器的质荷比；必须是有限数。 */
    double mzq3;
    /**
     * @brief 本通道的高斯平滑参数 sigma。
     * @details 必须是有限数。大于 0 时覆盖 QfConfig::smooth_sigma；小于或等于
     *          0 时使用全局配置。
     */
    float smooth_sigma;
    /**
     * @brief 保留时间数组，元素数量为 n_points。
     * @details 必填；所有元素必须是有限数并严格递增。
     */
    const double* rt;
    /**
     * @brief 与 rt 一一对应的强度数组，元素数量为 n_points。
     * @details 必填；所有元素必须是有限数。
     */
    const double* intensity;
    /** @brief rt 和 intensity 两个数组的共同元素数量，必须至少为 2。 */
    int32_t n_points;
} QfCompoundInput;

/**
 * @brief 任务进入终态时调用的回调函数类型。
 *
 * 每个任务最多回调一次。回调通常在工作线程中同步执行；调用 qf_stop() 取消
 * 排队中的任务时，也可能直接在调用 qf_stop() 的线程中执行。回调应尽快返回，
 * 并自行同步共享数据。
 *
 * @param[in] result_json_path 成功任务的结果 JSON 文件路径；任务失败或取消时为
 *            NULL。指针由库拥有，有效期截至 qf_shutdown()；如需长期保存请复制。
 * @param[in] task_id 进入终态的任务 ID。
 * @param[in] status 终态值，对应 QfTaskStatus 中的 QF_TASK_SUCCESS、
 *            QF_TASK_FAILED 或 QF_TASK_CANCELLED。
 * @param[in] user_data 注册回调时传给 qf_set_callback() 的原样指针，库不会解引用
 *            或释放它。
 * @warning 不得在回调中调用 qf_shutdown()。可以调用 qf_query() 查询当前任务。
 */
typedef void (*QfCallback)(const char* result_json_path,
                           int64_t task_id,
                           int32_t status,
                           void* user_data);

/**
 * @brief 使用推荐默认值初始化配置结构体。
 * @param[out] config 待初始化的配置；为 NULL 时不执行任何操作。
 * @note model_path 的默认值为 NULL，仍须由调用方在 qf_init() 前设置。
 */
QF_API void qf_default_config(QfConfig* config);

/**
 * @brief 初始化全局 MRMPFormer 实例、加载模型并启动工作线程。
 *
 * 同一时刻只能存在一个实例。初始化成功后可提交任务；成功关闭后可用新配置
 * 重新初始化。函数会深拷贝配置中的字符串。
 *
 * @param[in] config 完整配置，不能为 NULL；建议由 qf_default_config() 初始化。
 * @return QF_OK 表示成功；配置非法、模型不存在/加载失败、输出目录不可创建或
 *         重复初始化时返回相应 QfError。
 */
QF_API QfError qf_init(const QfConfig* config);

/**
 * @brief 停止任务系统并释放全局资源。
 *
 * 尚未结束的任务会被取消，工作线程会被回收。成功返回后可再次 qf_init()。
 * 库持有的借用指针随之失效，但已生成的结果文件不会自动删除。
 *
 * @return QF_OK 表示成功；未初始化时返回 QF_ERR_NOT_INITIALIZED。
 * @warning 不得从任务工作线程或 QfCallback 内调用；此时返回
 *          QF_ERR_INVALID_PARAM。
 */
QF_API QfError qf_shutdown(void);

/**
 * @brief 从 JSON 文件读取一批输入并异步提交推理任务。
 *
 * 返回前完成文件读取和 JSON 解析，因此成功返回后可删除或修改输入文件。
 * 成功仅表示任务已入队，不表示推理完成。
 *
 * @param[in] json_path 输入 JSON 文件路径，不能为 NULL 或空字符串。
 * @param[out] out_task_id 成功时写入新任务 ID，不能为 NULL；失败时原值不变。
 * @return QF_OK 表示提交成功；也可能返回未初始化、参数、文件或 JSON 错误。
 */
QF_API QfError qf_process(const char* json_path, int64_t* out_task_id);

/**
 * @brief 直接从内存异步提交一个化合物/色谱通道的推理任务。
 *
 * 返回前深拷贝 QfCompoundInput 的所有字符串和数组。成功仅表示任务已入队。
 *
 * @param[in] input 单条输入，不能为 NULL；字段约束见 QfCompoundInput。
 * @param[out] out_task_id 成功时写入新任务 ID，不能为 NULL；失败时原值不变。
 * @return QF_OK 表示提交成功；也可能返回未初始化或参数非法。
 */
QF_API QfError qf_process_single(const QfCompoundInput* input, int64_t* out_task_id);

/**
 * @brief 请求取消指定的非终态任务。
 *
 * 成功后状态立即变为 QF_TASK_CANCELLED，并触发已注册的终态回调。取消运行中的
 * 任务属于协作式取消，后台处理会在下一个取消检查点停止。
 *
 * @param[in] task_id qf_process() 或 qf_process_single() 返回的任务 ID。
 * @return QF_OK 表示取消成功；任务不存在、已取消或已结束时返回相应错误。
 */
QF_API QfError qf_stop(int64_t task_id);

/**
 * @brief 阻塞当前线程，直到任务进入任一终态或等待超时。
 *
 * 任务成功、失败或取消都会使等待返回 QF_OK；最终状态应通过 qf_query() 获取。
 * 等待超时不会取消任务。
 *
 * @param[in] task_id 要等待的任务 ID。
 * @param[in] timeout_ms 最长等待毫秒数；0 表示无限等待，负数非法。
 * @return QF_OK 表示任务已进入终态；超时返回 QF_ERR_TIMEOUT；任务不存在或参数
 *         非法时返回相应错误。
 */
QF_API QfError qf_wait(int64_t task_id, int32_t timeout_ms);

/**
 * @brief 查询任务当前状态和进度。
 *
 * 四个输出参数均可选，但至少必须提供一个非 NULL 参数。输出是调用时刻的快照。
 *
 * @param[in] task_id 要查询的任务 ID。
 * @param[out] out_status 可选；接收当前 QfTaskStatus。
 * @param[out] out_progress 可选；接收 [0, 1] 范围内的完成比例。
 * @param[out] out_items_done 可选；接收已处理完成的输入条目数。
 * @param[out] out_items_total 可选；接收任务包含的输入条目总数。
 * @return QF_OK 表示成功；所有输出均为 NULL 或任务不存在时返回相应错误。
 */
QF_API QfError qf_query(int64_t task_id,
                         QfTaskStatus* out_status,
                         float* out_progress,
                         int32_t* out_items_done,
                         int32_t* out_items_total);

/**
 * @brief 设置或移除进程内唯一的任务终态回调。
 *
 * 新设置替换旧设置；NULL 回调表示取消注册。不会为注册前已经完成且已派发回调的
 * 任务补发通知。
 *
 * @param[in] callback 回调函数；NULL 表示取消注册。
 * @param[in] user_data 回调时原样传回的用户数据，可为 NULL；生命周期由调用方管理。
 * @return QF_OK 表示成功；未初始化时返回 QF_ERR_NOT_INITIALIZED。
 */
QF_API QfError qf_set_callback(QfCallback callback, void* user_data);

/**
 * @brief 将成功任务的结果 JSON 文件路径复制到调用方缓冲区。
 * @param[in] task_id 已成功完成的任务 ID。
 * @param[out] out_path 接收以 NUL 结尾路径的缓冲区，不能为 NULL。
 * @param[in] buf_size 缓冲区总字节容量，必须大于 0，并包含结尾 NUL 的空间。
 * @return QF_OK 表示成功；缓冲区过小、任务未完成/失败/取消或不存在时返回错误。
 * @note 结果文件不会在 qf_shutdown() 时自动删除。
 */
QF_API QfError qf_get_result_path(int64_t task_id, char* out_path, int32_t buf_size);

/**
 * @brief 获取成功任务的完整结果 JSON 文本。
 * @param[in] task_id 已成功完成的任务 ID。
 * @param[out] out_json 成功时接收库分配的 NUL 结尾字符串；不能为 NULL。调用开始
 *             时会先被置为 NULL。
 * @return QF_OK 表示成功；任务未完成/失败/取消、不存在或分配失败时返回错误。
 * @note 成功取得的字符串必须且只能使用 qf_free() 释放。
 */
QF_API QfError qf_get_result_json(int64_t task_id, char** out_json);

/**
 * @brief 释放 MRMPFormer 接口返回的动态内存。
 * @param[in] ptr qf_get_result_json() 返回的指针；允许为 NULL。
 * @warning 不要释放调用方自行分配的内存，也不要重复释放同一指针。
 */
QF_API void qf_free(void* ptr);

/**
 * @brief 获取当前线程最近一次 API 调用失败的详细说明。
 * @return 存在错误时返回库持有的 NUL 结尾字符串；没有错误时返回 NULL。
 * @note 返回指针仅供借用，无需且不得释放；同一线程后续 API 调用可能使其失效。
 *       错误状态按线程保存，必须在发生错误的同一线程中读取。
 */
QF_API const char* qf_get_error(void);

/**
 * @brief 获取 MRMPFormer C API 的版本字符串。
 * @return 库持有的 NUL 结尾版本字符串，例如 `0.1.0`；调用方不得释放。
 */
QF_API const char* qf_version(void);

/**
 * @brief 查询当前实例是否实际启用了 GPU 推理。
 * @return 已初始化且实际启用 GPU 时返回 1；未启用或尚未初始化时返回 0。
 * @note QfConfig::use_gpu 为 0 时，可用本函数判断最终是否回退到 CPU。
 */
QF_API int32_t qf_is_gpu_enabled(void);

#ifdef __cplusplus
}
#endif

#endif

#include "mrmpformer.h"

#include <inttypes.h>
#include <stdio.h>

static int check(QfError error, const char* operation) {
    if (error == QF_OK) {
        return 1;
    }
    fprintf(stderr, "%s failed (%d): %s\n", operation, (int)error,
            qf_get_error() == NULL ? "no error detail" : qf_get_error());
    return 0;
}

int main(int argc, char** argv) {
    static const double rt[] = {0.00, 0.10, 0.20, 0.30, 0.40, 0.50,
                                0.60, 0.70, 0.80, 0.90, 1.00};
    static const double intensity[] = {100.0,  800.0,  3000.0, 7000.0,
                                       12000.0, 7000.0, 3000.0, 800.0,
                                       100.0,  80.0,   60.0};
    QfConfig config;
    QfCompoundInput input = {
        "example-001",             /* uid */
        "Example compound",        /* name */
        "100.0>50.0",              /* channel */
        100.0,                      /* mzq1 */
        50.0,                       /* mzq3 */
        0.0f,                       /* smooth_sigma */
        rt,                         /* rt */
        intensity,                  /* intensity */
        (int32_t)(sizeof(rt) / sizeof(rt[0])), /* n_points */
    };
    int64_t task_id = 0;
    char* result_json = NULL;
    int exit_code = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <mrmpformerv2.onnx>\n", argv[0]);
        return 2;
    }

    qf_default_config(&config);
    config.model_path = argv[1];
    config.use_gpu = -1; /* force CPU for this example */
    config.max_workers = 1;

    if (!check(qf_init(&config), "qf_init")) {
        return exit_code;
    }
    if (!check(qf_process_single(&input, &task_id), "qf_process_single") ||
        !check(qf_wait(task_id, 30000), "qf_wait") ||
        !check(qf_get_result_json(task_id, &result_json), "qf_get_result_json")) {
        goto cleanup;
    }

    printf("task %" PRId64 " result:\n%s\n", task_id, result_json);
    exit_code = 0;

cleanup:
    qf_free(result_json);
    if (!check(qf_shutdown(), "qf_shutdown")) {
        exit_code = 1;
    }
    return exit_code;
}

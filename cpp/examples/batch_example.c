#include "mrmpformer.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static int check(QfError error, const char* operation) {
    if (error == QF_OK) {
        return 1;
    }
    fprintf(stderr, "%s failed (%d): %s\n", operation, (int)error,
            qf_get_error() == NULL ? "no error detail" : qf_get_error());
    return 0;
}

int main(int argc, char** argv) {
    QfConfig config;
    int64_t task_id = 0;
    char* result_json = NULL;
    int exit_code = 1;

    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: %s <mrmpformerv2.onnx> <input.json> [use_gpu: -1=CPU, 0=auto, 1=GPU]\n", argv[0]);
        return 2;
    }

    qf_default_config(&config);
    config.model_path = argv[1];
    config.use_gpu = argc == 4 ? atoi(argv[3]) : 0;
    config.max_workers = 1;

    if (!check(qf_init(&config), "qf_init")) {
        return exit_code;
    }
    if (!check(qf_process(argv[2], &task_id), "qf_process") ||
        !check(qf_wait(task_id, 30000), "qf_wait") ||
        !check(qf_get_result_json(task_id, &result_json), "qf_get_result_json")) {
        goto cleanup;
    }

    printf("task %" PRId64 " result:\n%s\n", task_id, result_json);
    exit_code = 0;

cleanup:
    qf_free(result_json); /* qf_free(NULL) is permitted. */
    if (!check(qf_shutdown(), "qf_shutdown")) {
        exit_code = 1;
    }
    return exit_code;
}

// Test driver: exercise the public C ABI in a separate, private-runtime process.
#include "mrmpformer.h"
#include <json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::json;

static void check(QfError code) {
    if (code != QF_OK) {
        const char* detail = qf_get_error();
        throw std::runtime_error(std::to_string(code) + ": " + (detail ? detail : "unknown"));
    }
}

static json result(int64_t task) {
    check(qf_wait(task, 300000));
    char* text = nullptr;
    check(qf_get_result_json(task, &text));
    std::string owned(text);
    qf_free(text);
    return json::parse(owned);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: inference_parity_runner model.onnx suite.json output.json\n";
        return 2;
    }
    bool initialized = false;
    try {
        std::ifstream input(argv[2]);
        const json suite = json::parse(input);
        json output = json::array();
        for (const auto& group : suite) {
            QfConfig config;
            qf_default_config(&config);
            config.model_path = argv[1];
            config.work_dir = "results";
            config.use_gpu = group.at("config").value("use_gpu", -1);
            config.max_workers = 1;
            const auto& values = group.at("config");
            config.threshold = values.at("threshold").get<float>();
            config.smooth_sigma = values.at("smooth_sigma").get<float>();
            config.batch_size = values.at("batch_size").get<int>();
            config.min_chrom_points = values.at("min_chrom_points").get<int>();
            config.min_max_intensity = values.at("min_max_intensity").get<float>();
            check(qf_init(&config));
            initialized = true;
            {
                std::ofstream batch("batch.json");
                batch << json{{"items", group.at("items")}};
                if (!batch) { throw std::runtime_error("Cannot write batch.json"); }
            }
            int64_t task = 0;
            check(qf_process("batch.json", &task));
            json entry{{"batch", result(task)}, {"single", json::array()}, {"gpu_enabled", qf_is_gpu_enabled() != 0}};
            for (const auto& item : group.at("items")) {
                const auto uid = item.at("uid").get<std::string>();
                const auto channel = item.at("channel").get<std::string>();
                const auto rt = item.at("x").get<std::vector<double>>();
                const auto intensity = item.at("y").get<std::vector<double>>();
                QfCompoundInput one{};
                one.uid = uid.c_str();
                one.channel = channel.c_str();
                one.mzq1 = item.at("mzq1").get<double>();
                one.mzq3 = item.at("mzq3").get<double>();
                one.smooth_sigma = item.value("smooth_sigma", 0.0f);
                one.rt = rt.data();
                one.intensity = intensity.data();
                one.n_points = static_cast<int32_t>(rt.size());
                check(qf_process_single(&one, &task));
                const auto response = result(task).at("items");
                if (response.size() != 1) { throw std::runtime_error("Single result count mismatch"); }
                entry["single"].push_back(response.at(0));
            }
            output.push_back(entry);
            check(qf_shutdown());
            initialized = false;
        }
        std::ofstream file(argv[3]);
        file << output.dump(2);
        if (!file) { throw std::runtime_error("Cannot write output JSON"); }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (initialized) { qf_shutdown(); }
        return 1;
    }
}

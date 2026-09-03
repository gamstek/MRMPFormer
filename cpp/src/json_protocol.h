#ifndef MRMPFORMER_JSON_PROTOCOL_H
#define MRMPFORMER_JSON_PROTOCOL_H

#include <map>
#include <string>
#include <string_view>
#include <vector>

struct CompoundData {
    std::string uid;
    std::string name;
    std::string channel;
    double mzq1 = 0.0;
    double mzq3 = 0.0;
    float smooth_sigma = 0.0f;
    std::vector<double> x;
    std::vector<double> y;
};

struct PeakResult {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
};

struct Alert {
    std::string level;
    std::string code;
    std::map<std::string, double> detail;
};

struct CompoundResult {
    std::string uid;
    std::string status;
    std::vector<PeakResult> peaks;
    std::vector<Alert> alerts;
};

struct TaskResult {
    std::vector<CompoundResult> items;
};

struct ParseResult {
    bool ok = false;
    std::vector<CompoundData> items;
    std::string error;
};

ParseResult parse_input_json(std::string_view input);
std::string generate_output_json(const TaskResult& result);

#endif

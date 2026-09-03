#ifndef MRMPFORMER_SIGNAL_PROCESSING_H
#define MRMPFORMER_SIGNAL_PROCESSING_H

#include "json_protocol.h"
#include "mrmpformer.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

struct SignalQc {
    bool accepted = false;
    std::string code;
    std::size_t n_points = 0;
    double max_intensity = 0.0;
};

std::vector<double> smooth_signal(const std::vector<double>& values, float sigma);
SignalQc inspect_signal(const CompoundData& compound, const QfConfig& config);
std::optional<PeakResult> find_signal_fallback(const CompoundData& compound);

#endif

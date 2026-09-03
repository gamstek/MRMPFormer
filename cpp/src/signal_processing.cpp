#include "signal_processing.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr double k_epsilon = 1e-12;

float effective_sigma(const CompoundData& compound, const QfConfig& config) {
    return compound.smooth_sigma > 0.0f ? compound.smooth_sigma : config.smooth_sigma;
}

bool is_local_minimum(const std::vector<double>& values, std::size_t index) {
    return values[index] <= values[index - 1] && values[index] <= values[index + 1];
}

}  // namespace

std::vector<double> smooth_signal(const std::vector<double>& values, float sigma) {
    if (values.empty() || !(sigma > 0.0f) || !std::isfinite(sigma)) {
        return values;
    }

    const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    std::vector<double> kernel(static_cast<std::size_t>(2 * radius + 1));
    double kernel_sum = 0.0;
    for (int offset = -radius; offset <= radius; ++offset) {
        const double scaled = static_cast<double>(offset) / sigma;
        const double weight = std::exp(-0.5 * scaled * scaled);
        kernel[static_cast<std::size_t>(offset + radius)] = weight;
        kernel_sum += weight;
    }

    for (double& weight : kernel) {
        weight /= kernel_sum;
    }

    std::vector<double> smoothed(values.size(), 0.0);
    const int last_index = static_cast<int>(values.size()) - 1;
    for (std::size_t index = 0; index < values.size(); ++index) {
        double value = 0.0;
        for (int offset = -radius; offset <= radius; ++offset) {
            const int source_index = std::clamp(static_cast<int>(index) + offset, 0, last_index);
            value += kernel[static_cast<std::size_t>(offset + radius)] *
                     values[static_cast<std::size_t>(source_index)];
        }
        smoothed[index] = value;
    }
    return smoothed;
}

SignalQc inspect_signal(const CompoundData& compound, const QfConfig& config) {
    SignalQc qc;
    qc.n_points = compound.y.size();

    const std::vector<double> smoothed = smooth_signal(compound.y, effective_sigma(compound, config));
    if (!smoothed.empty()) {
        qc.max_intensity = *std::max_element(smoothed.begin(), smoothed.end());
    }

    qc.accepted = qc.n_points >= static_cast<std::size_t>(std::max(config.min_chrom_points, 0)) &&
                  qc.max_intensity >= static_cast<double>(config.min_max_intensity);
    if (!qc.accepted) {
        qc.code = "CHANNEL_LOW_INTENSITY";
    }
    return qc;
}

std::optional<PeakResult> find_signal_fallback(const CompoundData& compound) {
    if (compound.x.size() != compound.y.size() || compound.x.size() < 2) {
        return std::nullopt;
    }

    const auto apex_it = std::max_element(compound.y.begin(), compound.y.end());
    const std::size_t apex_index = static_cast<std::size_t>(std::distance(compound.y.begin(), apex_it));

    std::size_t left_index = 0;
    for (std::size_t index = apex_index; index > 1; --index) {
        const std::size_t candidate = index - 1;
        if (is_local_minimum(compound.y, candidate)) {
            left_index = candidate;
            break;
        }
    }

    std::size_t right_index = compound.y.size() - 1;
    for (std::size_t index = apex_index + 1; index + 1 < compound.y.size(); ++index) {
        if (is_local_minimum(compound.y, index)) {
            right_index = index;
            break;
        }
    }

    const double a = compound.x[left_index];
    const double b = compound.x[right_index];
    if (!(a < b)) {
        return std::nullopt;
    }

    const double apex = compound.y[apex_index];
    const double baseline = (compound.y[left_index] + compound.y[right_index]) / 2.0;
    const double score = std::clamp((apex - baseline) / std::max(apex, k_epsilon), 0.0, 1.0);
    return PeakResult{a, b, score};
}

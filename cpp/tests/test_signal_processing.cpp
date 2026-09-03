#include "roi_renderer.h"
#include "signal_processing.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

CompoundData make_compound(std::vector<double> x, std::vector<double> y) {
    CompoundData compound;
    compound.uid = "synthetic";
    compound.channel = "quant";
    compound.x = std::move(x);
    compound.y = std::move(y);
    return compound;
}

QfConfig make_config() {
    QfConfig config{};
    config.min_chrom_points = 10;
    config.min_max_intensity = 1000.0f;
    return config;
}

void test_qc_rejects_too_few_points() {
    const CompoundData compound = make_compound(
        {0, 1, 2, 3, 4, 5, 6, 7, 8},
        {1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000});

    const SignalQc qc = inspect_signal(compound, make_config());

    assert(!qc.accepted);
    assert(qc.code == "CHANNEL_LOW_INTENSITY");
    assert(qc.n_points == 9);
    assert(qc.max_intensity == 1000.0);
}

void test_qc_uses_the_intensity_threshold_inclusively() {
    const CompoundData below = make_compound(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 999.9});
    const CompoundData at_threshold = make_compound(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1000.0});

    const SignalQc below_qc = inspect_signal(below, make_config());
    const SignalQc threshold_qc = inspect_signal(at_threshold, make_config());

    assert(!below_qc.accepted);
    assert(below_qc.code == "CHANNEL_LOW_INTENSITY");
    assert(threshold_qc.accepted);
    assert(threshold_qc.code.empty());
    assert(threshold_qc.max_intensity == 1000.0);
}

void test_qc_uses_gaussian_smoothing_before_measuring_the_maximum() {
    CompoundData compound = make_compound(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 0, 0, 0, 0, 1200, 0, 0, 0, 0});
    compound.smooth_sigma = 1.0f;

    const SignalQc qc = inspect_signal(compound, make_config());

    assert(!qc.accepted);
    assert(qc.max_intensity < 1000.0);
}

void test_fallback_encloses_the_dominant_apex() {
    const CompoundData compound = make_compound(
        {0, 1, 2, 3, 4, 5, 6},
        {0, 1, 4, 9, 4, 1, 0});

    const auto fallback = find_signal_fallback(compound);

    assert(fallback.has_value());
    assert(fallback->a < 3.0);
    assert(fallback->b > 3.0);
    assert(fallback->a < fallback->b);
    assert(fallback->c >= 0.0);
    assert(fallback->c <= 1.0);
}

void test_fallback_rejects_a_zero_width_rt_interval() {
    const CompoundData compound = make_compound({1, 1}, {0, 9});

    const auto fallback = find_signal_fallback(compound);

    assert(!fallback.has_value());
}

void test_roi_is_raw_rgb_with_deterministic_dimensions_and_preserves_input() {
    const CompoundData compound = make_compound(
        {10, 11, 12, 13, 14},
        {0, 2, 8, 2, 0});
    const std::vector<double> original_x = compound.x;
    const std::vector<double> original_y = compound.y;

    const RoiImage roi = render_roi(compound, 0.0f);

    assert(roi.width == 400);
    assert(roi.height == 300);
    assert(roi.rgb.size() == static_cast<std::size_t>(roi.width) * roi.height * 3);
    assert(roi.rt_min == 10.0);
    assert(roi.rt_max == 14.0);
    assert(compound.x == original_x);
    assert(compound.y == original_y);

    bool has_blue_polyline_pixel = false;
    for (std::size_t i = 0; i < roi.rgb.size(); i += 3) {
        if (roi.rgb[i] == 0 && roi.rgb[i + 1] == 0 && roi.rgb[i + 2] == 255) {
            has_blue_polyline_pixel = true;
            break;
        }
    }
    assert(has_blue_polyline_pixel);
}

}  // namespace

int main() {
    test_qc_rejects_too_few_points();
    test_qc_uses_the_intensity_threshold_inclusively();
    test_qc_uses_gaussian_smoothing_before_measuring_the_maximum();
    test_fallback_encloses_the_dominant_apex();
    test_fallback_rejects_a_zero_width_rt_interval();
    test_roi_is_raw_rgb_with_deterministic_dimensions_and_preserves_input();
    return 0;
}

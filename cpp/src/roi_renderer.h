#ifndef MRMPFORMER_ROI_RENDERER_H
#define MRMPFORMER_ROI_RENDERER_H

#include "json_protocol.h"

#include <cstdint>
#include <vector>

struct RoiImage {
    int width = 400;
    int height = 300;
    double rt_min = 0.0;
    double rt_max = 0.0;
    std::vector<std::uint8_t> rgb;
};

RoiImage render_roi(const CompoundData& compound, float sigma);

#endif

#include "roi_renderer.h"

#include "signal_processing.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {

constexpr int k_left_margin = 40;
constexpr int k_right_margin = 20;
constexpr int k_top_margin = 20;
constexpr int k_bottom_margin = 30;

void set_blue_pixel(RoiImage& image, int x, int y) {
    if (x < 0 || x >= image.width || y < 0 || y >= image.height) {
        return;
    }
    const std::size_t offset = (static_cast<std::size_t>(y) * image.width + x) * 3;
    image.rgb[offset] = 0;
    image.rgb[offset + 1] = 0;
    image.rgb[offset + 2] = 255;
}

void draw_blue_line(RoiImage& image, int x0, int y0, int x1, int y1) {
    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;

    while (true) {
        set_blue_pixel(image, x0, y0);
        if (x0 == x1 && y0 == y1) {
            return;
        }
        const int double_error = 2 * error;
        if (double_error >= dy) {
            error += dy;
            x0 += sx;
        }
        if (double_error <= dx) {
            error += dx;
            y0 += sy;
        }
    }
}

}  // namespace

RoiImage render_roi(const CompoundData& compound, float sigma) {
    RoiImage image;
    image.rgb.assign(static_cast<std::size_t>(image.width) * image.height * 3, 255);
    if (compound.x.empty() || compound.x.size() != compound.y.size()) {
        return image;
    }

    image.rt_min = compound.x.front();
    image.rt_max = compound.x.back();
    const std::vector<double> y = smooth_signal(compound.y, sigma);
    const auto [minimum_it, maximum_it] = std::minmax_element(y.begin(), y.end());
    const double y_min = *minimum_it;
    const double y_max = *maximum_it;
    const double x_span = image.rt_max - image.rt_min;
    const double y_span = y_max - y_min;
    const int plot_width = image.width - k_left_margin - k_right_margin;
    const int plot_height = image.height - k_top_margin - k_bottom_margin;

    auto x_coordinate = [&](double value) {
        const double normalized = x_span > 0.0 ? (value - image.rt_min) / x_span : 0.5;
        return k_left_margin + static_cast<int>(std::lround(normalized * plot_width));
    };
    auto y_coordinate = [&](double value) {
        const double normalized = y_span > 0.0 ? (value - y_min) / y_span : 0.5;
        return image.height - k_bottom_margin - static_cast<int>(std::lround(normalized * plot_height));
    };

    int previous_x = x_coordinate(compound.x.front());
    int previous_y = y_coordinate(y.front());
    set_blue_pixel(image, previous_x, previous_y);
    for (std::size_t index = 1; index < compound.x.size(); ++index) {
        const int current_x = x_coordinate(compound.x[index]);
        const int current_y = y_coordinate(y[index]);
        draw_blue_line(image, previous_x, previous_y, current_x, current_y);
        previous_x = current_x;
        previous_y = current_y;
    }
    return image;
}

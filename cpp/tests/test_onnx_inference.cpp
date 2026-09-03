#include "onnx_inference.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MRMPFORMER_TEST_MODEL_PATH
#error MRMPFORMER_TEST_MODEL_PATH must be defined
#endif

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string tensor_name(const Ort::Session& session, bool input, std::size_t index) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto name = input ? session.GetInputNameAllocated(index, allocator)
                      : session.GetOutputNameAllocated(index, allocator);
    return name.get();
}

void verify_model_metadata(const std::string& model_path) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mrmpformer-metadata-test");
    Ort::SessionOptions options;
#ifdef _WIN32
    const std::wstring wide_path(model_path.begin(), model_path.end());
    Ort::Session session(env, wide_path.c_str(), options);
#else
    Ort::Session session(env, model_path.c_str(), options);
#endif
    require(session.GetInputCount() == 2, "expected two model inputs");
    require(session.GetOutputCount() == 3, "expected three model outputs");
    require(tensor_name(session, true, 0) == "image", "missing image input");
    require(tensor_name(session, true, 1) == "img_size", "missing img_size input");
    require(tensor_name(session, false, 0) == "scores", "missing scores output");
    require(tensor_name(session, false, 1) == "boxes_xyxy", "missing boxes_xyxy output");
    require(tensor_name(session, false, 2) == "boxes_norm", "missing boxes_norm output");

    const std::array<std::size_t, 2> input_ranks{4, 1};
    for (std::size_t i = 0; i < input_ranks.size(); ++i) {
        const auto info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        require(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "input must be float32");
        require(info.GetShape().size() == input_ranks[i], "input rank differs");
    }
    const auto image_shape = session.GetInputTypeInfo(0)
                                 .GetTensorTypeAndShapeInfo().GetShape();
    const auto size_shape = session.GetInputTypeInfo(1)
                                .GetTensorTypeAndShapeInfo().GetShape();
    require(image_shape[1] == 3, "static image channel dimension must be three");
    require(size_shape[0] == 2, "img_size shape must be [2]");
    const std::array<std::size_t, 3> output_ranks{2, 3, 3};
    for (std::size_t i = 0; i < output_ranks.size(); ++i) {
        const auto info = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        require(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                "output must be float32");
        require(info.GetShape().size() == output_ranks[i], "output rank differs");
    }
    const auto scores_shape = session.GetOutputTypeInfo(0)
                                  .GetTensorTypeAndShapeInfo().GetShape();
    const auto pixel_shape = session.GetOutputTypeInfo(1)
                                 .GetTensorTypeAndShapeInfo().GetShape();
    const auto norm_shape = session.GetOutputTypeInfo(2)
                                .GetTensorTypeAndShapeInfo().GetShape();
    require(pixel_shape[2] == 4 && norm_shape[2] == 4,
            "box outputs must end in four coordinates");
    require(scores_shape[1] == pixel_shape[1] &&
                scores_shape[1] == norm_shape[1],
            "static output query dimensions must agree");
}

RoiImage generated_roi() {
    RoiImage image;
    image.width = 400;
    image.height = 300;
    image.rgb.resize(static_cast<std::size_t>(image.width * image.height * 3));
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const auto offset = static_cast<std::size_t>((y * image.width + x) * 3);
            image.rgb[offset] = static_cast<std::uint8_t>(x % 256);
            image.rgb[offset + 1] = static_cast<std::uint8_t>(y % 256);
            image.rgb[offset + 2] = static_cast<std::uint8_t>((x + y) % 256);
        }
    }
    return image;
}

}  // namespace

int main() {
    try {
    const std::string model_path = MRMPFORMER_TEST_MODEL_PATH;
    verify_model_metadata(model_path);

    QfConfig config{};
    config.use_gpu = -1;
    config.batch_size = 2;

    mrmpformer::OnnxInference inference;
    require(inference.load_model(model_path, config), inference.last_error().c_str());
    require(inference.is_loaded(), "CPU model should be loaded");
    require(!inference.is_gpu_enabled(), "forced CPU enabled GPU unexpectedly");

    QfConfig automatic = config;
    automatic.use_gpu = 0;
    mrmpformer::OnnxInference automatic_inference;
    require(automatic_inference.load_model(model_path, automatic),
            automatic_inference.last_error().c_str());

    const RoiImage image = generated_roi();
    RoiImage second_image = image;
    std::fill(second_image.rgb.begin(), second_image.rgb.end(), 0);
    RoiImage third_image = image;
    std::fill(third_image.rgb.begin(), third_image.rgb.end(), 255);
    const auto batches = inference.infer_batch(
        {image, second_image, third_image}, 0.0f);
    require(batches.size() == 3, "inference result batch count differs");
    require(batches[0].size() == 3 && batches[1].size() == 3 &&
                batches[2].size() == 3,
            "chunked batched inference must split three queries per image");
    require(std::fabs(batches[0][0].score - batches[1][0].score) > 1.0e-6f,
            "distinct images must retain separate batched outputs");
    for (const auto& detection : batches.front()) {
        require(std::isfinite(detection.score), "score must be finite");
        require(std::isfinite(detection.x1) && std::isfinite(detection.y1) &&
                    std::isfinite(detection.x2) && std::isfinite(detection.y2),
                "pixel box must be finite");
        require(std::fabs(detection.x1 - std::clamp(
                         (detection.cx - detection.width / 2.0f) * image.width,
                         0.0f, static_cast<float>(image.width))) <
                    1.0e-3f, "normalized x1 disagrees with pixel x1");
        require(std::fabs(detection.y1 - std::clamp(
                         (detection.cy - detection.height / 2.0f) * image.height,
                         0.0f, static_cast<float>(image.height))) <
                    1.0e-3f, "normalized y1 disagrees with pixel y1");
        require(std::fabs(detection.x2 - std::clamp(
                         (detection.cx + detection.width / 2.0f) * image.width,
                         0.0f, static_cast<float>(image.width))) <
                    1.0e-3f, "normalized x2 disagrees with pixel x2");
        require(std::fabs(detection.y2 - std::clamp(
                         (detection.cy + detection.height / 2.0f) * image.height,
                         0.0f, static_cast<float>(image.height))) <
                    1.0e-3f, "normalized y2 disagrees with pixel y2");
    }

    RoiImage different_size = image;
    different_size.width = 200;
    different_size.rgb.resize(
        static_cast<std::size_t>(different_size.width * different_size.height * 3));
    bool rejected_mixed_sizes = false;
    try {
        (void)inference.infer_batch({image, different_size}, 0.0f);
    } catch (const std::invalid_argument&) {
        rejected_mixed_sizes = true;
    }
    require(rejected_mixed_sizes,
            "one batched invocation must reject mixed ROI dimensions");

    QfConfig forced_gpu = config;
    forced_gpu.use_gpu = 1;
    mrmpformer::OnnxInference gpu_inference;
    if (!gpu_inference.load_model(model_path, forced_gpu)) {
        require(!gpu_inference.last_error().empty(), "forced CUDA failure needs detail");
        require(!gpu_inference.is_gpu_enabled(), "failed CUDA load reports enabled");
    }
    return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}

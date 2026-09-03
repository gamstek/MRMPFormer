#include "onnx_inference.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mrmpformer {
namespace {

constexpr std::array<const char*, 2> kInputNames{"image", "img_size"};
constexpr std::array<const char*, 3> kOutputNames{
    "scores", "boxes_xyxy", "boxes_norm"};

std::string status_message(OrtStatus* status) {
    if (status == nullptr) {
        return {};
    }
    const std::string message = Ort::GetApi().GetErrorMessage(status);
    Ort::GetApi().ReleaseStatus(status);
    return message;
}

void append_cuda_provider(Ort::SessionOptions& options) {
    const auto providers = Ort::GetAvailableProviders();
    if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") ==
        providers.end()) {
        throw std::runtime_error("CUDAExecutionProvider is not available");
    }
    OrtCUDAProviderOptionsV2* cuda_options = nullptr;
    std::string error = status_message(
        Ort::GetApi().CreateCUDAProviderOptions(&cuda_options));
    if (!error.empty()) {
        throw std::runtime_error(error);
    }
    struct Releaser {
        void operator()(OrtCUDAProviderOptionsV2* value) const {
            Ort::GetApi().ReleaseCUDAProviderOptions(value);
        }
    };
    std::unique_ptr<OrtCUDAProviderOptionsV2, Releaser> holder(cuda_options);
    error = status_message(
        Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(
            options, cuda_options));
    if (!error.empty()) {
        throw std::runtime_error(error);
    }
}

std::string value_name(const Ort::Session& session, bool input, std::size_t index) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto name = input ? session.GetInputNameAllocated(index, allocator)
                      : session.GetOutputNameAllocated(index, allocator);
    return name.get();
}

void validate_tensor(const Ort::Session& session,
                     bool input,
                     std::size_t index,
                     const char* expected_name,
                     std::size_t expected_rank,
                     std::int64_t expected_last_dimension = 0) {
    const std::string actual_name = value_name(session, input, index);
    if (actual_name != expected_name) {
        throw std::runtime_error(std::string(input ? "input" : "output") +
                                 " tensor " + std::to_string(index) +
                                 " must be named '" + expected_name +
                                 "', got '" + actual_name + "'");
    }
    const auto type_info = input ? session.GetInputTypeInfo(index)
                                 : session.GetOutputTypeInfo(index);
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("tensor '" + actual_name + "' must be float32");
    }
    const auto shape = tensor_info.GetShape();
    if (shape.size() != expected_rank) {
        throw std::runtime_error("tensor '" + actual_name + "' has invalid rank");
    }
    if (expected_last_dimension != 0 && shape.back() > 0 &&
        shape.back() != expected_last_dimension) {
        throw std::runtime_error("tensor '" + actual_name +
                                 "' has invalid final dimension");
    }
}

void validate_contract(const Ort::Session& session) {
    if (session.GetInputCount() != kInputNames.size() ||
        session.GetOutputCount() != kOutputNames.size()) {
        throw std::runtime_error(
            "model must expose exactly 2 inputs and 3 outputs");
    }
    validate_tensor(session, true, 0, kInputNames[0], 4);
    validate_tensor(session, true, 1, kInputNames[1], 1, 2);
    validate_tensor(session, false, 0, kOutputNames[0], 2);
    validate_tensor(session, false, 1, kOutputNames[1], 3, 4);
    validate_tensor(session, false, 2, kOutputNames[2], 3, 4);

    const auto image_shape = session.GetInputTypeInfo(0)
                                 .GetTensorTypeAndShapeInfo().GetShape();
    const auto size_shape = session.GetInputTypeInfo(1)
                                .GetTensorTypeAndShapeInfo().GetShape();
    const auto scores_shape = session.GetOutputTypeInfo(0)
                                  .GetTensorTypeAndShapeInfo().GetShape();
    const auto pixel_shape = session.GetOutputTypeInfo(1)
                                 .GetTensorTypeAndShapeInfo().GetShape();
    const auto norm_shape = session.GetOutputTypeInfo(2)
                                .GetTensorTypeAndShapeInfo().GetShape();
    if (image_shape[1] > 0 && image_shape[1] != 3) {
        throw std::runtime_error("tensor 'image' must have three channels");
    }
    if (size_shape[0] > 0 && size_shape[0] != 2) {
        throw std::runtime_error("tensor 'img_size' must have shape [2]");
    }
    const std::int64_t score_queries = scores_shape[1];
    const std::int64_t pixel_queries = pixel_shape[1];
    const std::int64_t norm_queries = norm_shape[1];
    if ((score_queries > 0 && pixel_queries > 0 &&
         score_queries != pixel_queries) ||
        (score_queries > 0 && norm_queries > 0 &&
         score_queries != norm_queries) ||
        (pixel_queries > 0 && norm_queries > 0 &&
         pixel_queries != norm_queries)) {
        throw std::runtime_error("model output query dimensions are incompatible");
    }
}

Ort::SessionOptions session_options(bool use_cuda) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    if (use_cuda) {
        append_cuda_provider(options);
    }
    return options;
}

std::unique_ptr<Ort::Session> create_session(Ort::Env& env,
                                             const std::string& path,
                                             bool use_cuda) {
    auto options = session_options(use_cuda);
    const std::filesystem::path model_path = std::filesystem::u8path(path);
    auto session = std::make_unique<Ort::Session>(
        env, model_path.c_str(), options);
    validate_contract(*session);
    return session;
}

std::vector<float> to_nchw(const RoiImage& image) {
    if (image.width <= 0 || image.height <= 0) {
        throw std::invalid_argument("ROI dimensions must be positive");
    }
    const auto pixels = static_cast<std::size_t>(image.width) *
                        static_cast<std::size_t>(image.height);
    if (image.rgb.size() != pixels * 3) {
        throw std::invalid_argument(
            "ROI RGB buffer size does not match width and height");
    }
    std::vector<float> nchw(pixels * 3);
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        nchw[pixel] = static_cast<float>(image.rgb[pixel * 3]);
        nchw[pixels + pixel] = static_cast<float>(image.rgb[pixel * 3 + 1]);
        nchw[pixels * 2 + pixel] = static_cast<float>(image.rgb[pixel * 3 + 2]);
    }
    return nchw;
}

}  // namespace

struct OnnxInference::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mrmpformer"};
    Ort::MemoryInfo memory_info{
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::unique_ptr<Ort::Session> session;
    mutable std::mutex mutex;
    bool gpu_enabled = false;
    std::size_t batch_size = 1;
    std::string error;
};

OnnxInference::OnnxInference() : impl_(std::make_unique<Impl>()) {}
OnnxInference::~OnnxInference() = default;

bool OnnxInference::load_model(const std::string& path, const QfConfig& config) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->session.reset();
    impl_->gpu_enabled = false;
    impl_->error.clear();
    if (path.empty()) {
        impl_->error = "model path is empty";
        return false;
    }
    if (config.use_gpu < -1 || config.use_gpu > 1) {
        impl_->error = "use_gpu must be -1 (CPU), 0 (auto), or 1 (CUDA)";
        return false;
    }
    impl_->batch_size = config.batch_size > 0
                            ? static_cast<std::size_t>(config.batch_size)
                            : 1;

    if (config.use_gpu >= 0) {
        try {
            impl_->session = create_session(impl_->env, path, true);
            impl_->gpu_enabled = true;
            return true;
        } catch (const std::exception& exception) {
            if (config.use_gpu == 1) {
                impl_->error = std::string("CUDA execution was required but unavailable: ") +
                               exception.what();
                return false;
            }
        }
    }

    try {
        impl_->session = create_session(impl_->env, path, false);
        return true;
    } catch (const std::exception& exception) {
        impl_->error = std::string("failed to load ONNX model: ") + exception.what();
        return false;
    }
}

std::vector<std::vector<Detection>> OnnxInference::infer_batch(
    const std::vector<RoiImage>& images, float threshold) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->session) {
        throw std::runtime_error("ONNX model is not loaded");
    }
    if (!std::isfinite(threshold)) {
        throw std::invalid_argument("threshold must be finite");
    }

    std::vector<std::vector<Detection>> result(images.size());
    if (images.empty()) {
        return result;
    }
    const int width = images.front().width;
    const int height = images.front().height;
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("ROI dimensions must be positive");
    }
    for (const RoiImage& image : images) {
        if (image.width != width || image.height != height) {
            throw std::invalid_argument(
                "all ROIs in an inference batch must have identical dimensions");
        }
    }

    const std::size_t values_per_image =
        static_cast<std::size_t>(3) * static_cast<std::size_t>(width) *
        static_cast<std::size_t>(height);
    for (std::size_t chunk_begin = 0; chunk_begin < images.size();
         chunk_begin += impl_->batch_size) {
        const std::size_t chunk_size =
            std::min(impl_->batch_size, images.size() - chunk_begin);
        std::vector<float> input_data(chunk_size * values_per_image);
        for (std::size_t batch_index = 0; batch_index < chunk_size;
             ++batch_index) {
            std::vector<float> image_data =
                to_nchw(images[chunk_begin + batch_index]);
            std::copy(image_data.begin(), image_data.end(),
                      input_data.begin() + batch_index * values_per_image);
        }
        std::array<float, 2> image_size{
            static_cast<float>(width), static_cast<float>(height)};
        const std::array<std::int64_t, 4> input_shape{
            static_cast<std::int64_t>(chunk_size), 3, height, width};
        const std::array<std::int64_t, 1> size_shape{2};
        std::array<Ort::Value, 2> inputs{
            Ort::Value::CreateTensor<float>(
                impl_->memory_info, input_data.data(), input_data.size(),
                input_shape.data(), input_shape.size()),
            Ort::Value::CreateTensor<float>(
                impl_->memory_info, image_size.data(), image_size.size(),
                size_shape.data(), size_shape.size())};
        auto outputs = impl_->session->Run(
            Ort::RunOptions{nullptr}, kInputNames.data(), inputs.data(), inputs.size(),
            kOutputNames.data(), kOutputNames.size());

        const auto score_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const auto pixel_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        const auto norm_shape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
        if (score_shape.size() != 2 || pixel_shape.size() != 3 ||
            norm_shape.size() != 3 ||
            score_shape[0] != static_cast<std::int64_t>(chunk_size) ||
            pixel_shape[0] != static_cast<std::int64_t>(chunk_size) ||
            norm_shape[0] != static_cast<std::int64_t>(chunk_size) ||
            pixel_shape[1] != score_shape[1] || norm_shape[1] != score_shape[1] ||
            pixel_shape[2] != 4 || norm_shape[2] != 4) {
            throw std::runtime_error("model returned incompatible output shapes");
        }

        const float* scores = outputs[0].GetTensorData<float>();
        const float* boxes = outputs[1].GetTensorData<float>();
        const float* normalized = outputs[2].GetTensorData<float>();
        const std::size_t query_count = static_cast<std::size_t>(score_shape[1]);
        for (std::size_t batch_index = 0; batch_index < chunk_size;
             ++batch_index) {
            std::vector<Detection>& detections =
                result[chunk_begin + batch_index];
            detections.reserve(query_count);
            for (std::size_t query = 0; query < query_count; ++query) {
                const std::size_t score_offset = batch_index * query_count + query;
                const std::size_t box_offset = score_offset * 4;
                const float score = scores[score_offset];
                const float raw_x1 = boxes[box_offset];
                const float raw_y1 = boxes[box_offset + 1];
                const float raw_x2 = boxes[box_offset + 2];
                const float raw_y2 = boxes[box_offset + 3];
                if (!std::isfinite(score) || score < threshold ||
                    !std::isfinite(raw_x1) || !std::isfinite(raw_y1) ||
                    !std::isfinite(raw_x2) || !std::isfinite(raw_y2) ||
                    raw_x2 <= raw_x1 || raw_y2 <= raw_y1 || raw_x2 <= 0.0f ||
                    raw_y2 <= 0.0f || raw_x1 >= width || raw_y1 >= height) {
                    continue;
                }
                const float* norm = normalized + box_offset;
                if (!std::all_of(norm, norm + 4,
                                 [](float value) { return std::isfinite(value); })) {
                    continue;
                }
                detections.push_back({
                    score,
                    std::clamp(raw_x1, 0.0f, static_cast<float>(width)),
                    std::clamp(raw_y1, 0.0f, static_cast<float>(height)),
                    std::clamp(raw_x2, 0.0f, static_cast<float>(width)),
                    std::clamp(raw_y2, 0.0f, static_cast<float>(height)),
                    norm[0], norm[1], norm[2], norm[3]});
            }
        }
    }
    return result;
}

bool OnnxInference::is_gpu_enabled() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->gpu_enabled;
}

bool OnnxInference::is_loaded() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->session != nullptr;
}

std::string OnnxInference::last_error() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->error;
}

}  // namespace mrmpformer

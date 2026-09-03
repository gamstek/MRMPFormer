#ifndef MRMPFORMER_ONNX_INFERENCE_H
#define MRMPFORMER_ONNX_INFERENCE_H

#include "mrmpformer.h"
#include "roi_renderer.h"

#include <memory>
#include <string>
#include <vector>

namespace mrmpformer {

struct Detection {
    float score = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
};

class OnnxInference {
public:
    OnnxInference();
    ~OnnxInference();
    OnnxInference(const OnnxInference&) = delete;
    OnnxInference& operator=(const OnnxInference&) = delete;

    bool load_model(const std::string& path, const QfConfig& config);
    std::vector<std::vector<Detection>> infer_batch(
        const std::vector<RoiImage>& images, float threshold);
    bool is_gpu_enabled() const;
    bool is_loaded() const;
    std::string last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mrmpformer

#endif

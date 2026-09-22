#ifndef MRMPFORMER_PYTHON_BRIDGE_H
#define MRMPFORMER_PYTHON_BRIDGE_H

#include "json_protocol.h"
#include "mrmpformer.h"

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

struct _object;

namespace mrmpformer {

class PythonBridge {
public:
    PythonBridge() = default;
    ~PythonBridge();

    PythonBridge(const PythonBridge&) = delete;
    PythonBridge& operator=(const PythonBridge&) = delete;

    bool initialize(const std::string& model_path,
                    const QfConfig& config,
                    std::string* error);
    void shutdown() noexcept;
    bool process(const std::vector<CompoundData>& items,
                 std::string* output_json,
                 std::string* error);
    bool is_gpu_enabled() const noexcept;

private:
    _object* module_ = nullptr;
    bool initialized_ = false;
    std::atomic<bool> gpu_enabled_{false};
    mutable std::mutex call_mutex_;
};

}  // namespace mrmpformer

#endif

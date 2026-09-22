#include "python_bridge.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <json.hpp>

#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace mrmpformer {
namespace {

std::once_flag python_once;
bool python_ready = false;
std::string python_startup_error;

std::filesystem::path library_directory() {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&python_once), &module)) {
        return {};
    }
    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(
        module, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) {
        return {};
    }
    buffer.resize(length);
    return std::filesystem::path(buffer).parent_path();
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&python_once), &info) == 0 ||
        info.dli_fname == nullptr) {
        return {};
    }
    return std::filesystem::path(info.dli_fname).parent_path();
#endif
}

std::string python_error_text() {
    if (!PyErr_Occurred()) {
        return "unknown Python error";
    }
    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);
    PyObject* text = value == nullptr ? nullptr : PyObject_Str(value);
    const char* utf8 = text == nullptr ? nullptr : PyUnicode_AsUTF8(text);
    std::string result = utf8 == nullptr ? "Python call failed" : utf8;
    Py_XDECREF(text);
    Py_XDECREF(traceback);
    Py_XDECREF(value);
    Py_XDECREF(type);
    PyErr_Clear();
    return result;
}

void add_python_path(const char* path) {
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    PyObject* sys_path = PySys_GetObject("path");  // borrowed
    PyObject* value = PyUnicode_DecodeFSDefault(path);
    if (sys_path == nullptr || value == nullptr || PyList_Insert(sys_path, 0, value) != 0) {
        Py_XDECREF(value);
        throw std::runtime_error("failed to add MRMPFormer Python module path: " +
                                 python_error_text());
    }
    Py_DECREF(value);
}

void add_python_path(const std::filesystem::path& path) {
#if defined(_WIN32)
    const std::wstring native_path = path.wstring();
    PyObject* value = PyUnicode_FromWideChar(native_path.c_str(), native_path.size());
    PyObject* sys_path = PySys_GetObject("path");
    if (value == nullptr || sys_path == nullptr || PyList_Insert(sys_path, 0, value) != 0) {
        Py_XDECREF(value);
        throw std::runtime_error("failed to add MRMPFormer Python module path: " +
                                 python_error_text());
    }
    Py_DECREF(value);
#else
    add_python_path(path.c_str());
#endif
}

void start_python_once() {
    try {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            throw std::runtime_error("Py_Initialize did not initialize CPython");
        }
        const std::filesystem::path dll_dir = library_directory();
        if (!dll_dir.empty()) {
            add_python_path(dll_dir);
            add_python_path(dll_dir / "python");
            add_python_path(dll_dir / "python" / "Lib" / "site-packages");
        }
        if (const char* configured_path = std::getenv("MRMPFORMER_PYTHON_PATH")) {
            add_python_path(configured_path);
        }
        // CPython remains alive for the DLL process lifetime.  qf_shutdown only
        // releases the MassNova module/session so qf_init can safely run again.
        PyEval_SaveThread();
        python_ready = true;
    } catch (const std::exception& exception) {
        python_startup_error = exception.what();
    }
}

class GilGuard {
public:
    GilGuard() : state_(PyGILState_Ensure()) {}
    ~GilGuard() { PyGILState_Release(state_); }

private:
    PyGILState_STATE state_;
};

PyObject* required_callable(PyObject* module, const char* name) {
    PyObject* callable = PyObject_GetAttrString(module, name);
    if (callable == nullptr || !PyCallable_Check(callable)) {
        Py_XDECREF(callable);
        throw std::runtime_error(std::string("Python bridge function is unavailable: ") + name);
    }
    return callable;
}

std::string config_json(const QfConfig& config) {
    nlohmann::ordered_json value;
    value["threshold"] = config.threshold;
    value["smooth_sigma"] = config.smooth_sigma;
    value["use_gpu"] = config.use_gpu;
    value["batch_size"] = config.batch_size;
    value["min_chrom_points"] = config.min_chrom_points;
    value["min_max_intensity"] = config.min_max_intensity;
    return value.dump();
}

std::string input_json(const std::vector<CompoundData>& items) {
    nlohmann::ordered_json root;
    root["items"] = nlohmann::ordered_json::array();
    for (const CompoundData& item : items) {
        nlohmann::ordered_json value;
        value["uid"] = item.uid;
        value["name"] = item.name;
        value["channel"] = item.channel;
        value["mzq1"] = item.mzq1;
        value["mzq3"] = item.mzq3;
        value["smooth_sigma"] = item.smooth_sigma;
        value["x"] = item.x;
        value["y"] = item.y;
        root["items"].push_back(std::move(value));
    }
    return root.dump();
}

}  // namespace

PythonBridge::~PythonBridge() {
    shutdown();
}

bool PythonBridge::initialize(const std::string& model_path,
                              const QfConfig& config,
                              std::string* error) {
    std::lock_guard<std::mutex> lock(call_mutex_);
    std::call_once(python_once, start_python_once);
    if (!python_ready) {
        *error = python_startup_error.empty() ? "failed to initialize CPython" : python_startup_error;
        return false;
    }

    GilGuard gil;
    PyObject* module = PyImport_ImportModule("inference.massnova_bridge_native");
    if (module == nullptr) {
        // Development fallback.  Release packages include the compiled .pyd.
        PyErr_Clear();
        module = PyImport_ImportModule("inference.massnova_bridge");
    }
    if (module == nullptr) {
        *error = "failed to import MassNova bridge: " + python_error_text();
        return false;
    }

    bool runtime_initialized = false;
    try {
        PyObject* function = required_callable(module, "initialize");
        const std::string serialized_config = config_json(config);
        PyObject* result = PyObject_CallFunction(
            function, "ss", model_path.c_str(), serialized_config.c_str());
        Py_DECREF(function);
        if (result == nullptr) {
            throw std::runtime_error("MassNova initialize failed: " + python_error_text());
        }
        Py_DECREF(result);
        runtime_initialized = true;

        PyObject* gpu_function = required_callable(module, "is_gpu_enabled");
        PyObject* gpu_result = PyObject_CallNoArgs(gpu_function);
        Py_DECREF(gpu_function);
        if (gpu_result == nullptr) {
            throw std::runtime_error("failed to query Python GPU provider: " + python_error_text());
        }
        const int gpu_truth = PyObject_IsTrue(gpu_result);
        Py_DECREF(gpu_result);
        if (gpu_truth < 0) {
            throw std::runtime_error("failed to read Python GPU provider state: " +
                                     python_error_text());
        }
        gpu_enabled_.store(gpu_truth == 1);
        module_ = module;
        initialized_ = true;
        return true;
    } catch (const std::exception& exception) {
        PyErr_Clear();
        if (runtime_initialized) {
            PyObject* shutdown_function = PyObject_GetAttrString(module, "shutdown");
            if (shutdown_function != nullptr && PyCallable_Check(shutdown_function)) {
                PyObject* shutdown_result = PyObject_CallNoArgs(shutdown_function);
                Py_XDECREF(shutdown_result);
            }
            Py_XDECREF(shutdown_function);
            PyErr_Clear();
        }
        Py_DECREF(module);
        *error = exception.what();
        return false;
    }
}

void PythonBridge::shutdown() noexcept {
    std::lock_guard<std::mutex> lock(call_mutex_);
    if (!initialized_ || module_ == nullptr || !python_ready) {
        return;
    }
    GilGuard gil;
    PyObject* function = PyObject_GetAttrString(module_, "shutdown");
    if (function != nullptr && PyCallable_Check(function)) {
        PyObject* result = PyObject_CallNoArgs(function);
        Py_XDECREF(result);
    }
    Py_XDECREF(function);
    PyErr_Clear();
    Py_DECREF(module_);
    module_ = nullptr;
    initialized_ = false;
    gpu_enabled_.store(false);
}

bool PythonBridge::process(const std::vector<CompoundData>& items,
                           std::string* output_json,
                           std::string* error) {
    std::lock_guard<std::mutex> lock(call_mutex_);
    if (!initialized_ || module_ == nullptr) {
        *error = "MassNova Python bridge is not initialized";
        return false;
    }
    GilGuard gil;
    try {
        PyObject* function = required_callable(module_, "process_json");
        const std::string serialized_input = input_json(items);
        PyObject* result = PyObject_CallFunction(function, "s", serialized_input.c_str());
        Py_DECREF(function);
        if (result == nullptr) {
            throw std::runtime_error("MassNova processing failed: " + python_error_text());
        }
        if (!PyUnicode_Check(result)) {
            Py_DECREF(result);
            throw std::runtime_error("MassNova bridge returned a non-string result");
        }
        Py_ssize_t length = 0;
        const char* utf8 = PyUnicode_AsUTF8AndSize(result, &length);
        if (utf8 == nullptr) {
            Py_DECREF(result);
            throw std::runtime_error("failed to encode MassNova result: " + python_error_text());
        }
        output_json->assign(utf8, static_cast<std::size_t>(length));
        Py_DECREF(result);
        return true;
    } catch (const std::exception& exception) {
        PyErr_Clear();
        *error = exception.what();
        return false;
    }
}

bool PythonBridge::is_gpu_enabled() const noexcept {
    return gpu_enabled_.load();
}

}  // namespace mrmpformer

if(NOT DEFINED ONNXRUNTIME_ROOT OR ONNXRUNTIME_ROOT STREQUAL "")
    set(_bundled_onnxruntime_root
        "${CMAKE_CURRENT_LIST_DIR}/../third_party/onnxruntime")
    if(WIN32 AND EXISTS
       "${_bundled_onnxruntime_root}/include/onnxruntime_cxx_api.h")
        set(ONNXRUNTIME_ROOT "${_bundled_onnxruntime_root}")
    else()
        message(FATAL_ERROR
            "ONNXRUNTIME_ROOT is required. Set it to an ONNX Runtime directory containing "
            "include/onnxruntime_cxx_api.h and lib/onnxruntime.lib (Windows) or lib/libonnxruntime.so (Linux).")
    endif()
endif()

get_filename_component(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" ABSOLUTE)
set(_onnxruntime_include_path "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")

if(WIN32)
    set(_onnxruntime_library_path "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    set(_onnxruntime_runtime_path "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
else()
    set(_onnxruntime_library_path "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
endif()

set(_onnxruntime_missing_paths)
if(NOT EXISTS "${_onnxruntime_include_path}")
    list(APPEND _onnxruntime_missing_paths "${_onnxruntime_include_path}")
endif()
if(NOT EXISTS "${_onnxruntime_library_path}")
    list(APPEND _onnxruntime_missing_paths "${_onnxruntime_library_path}")
endif()
if(WIN32 AND NOT EXISTS "${_onnxruntime_runtime_path}")
    list(APPEND _onnxruntime_missing_paths "${_onnxruntime_runtime_path}")
endif()

if(_onnxruntime_missing_paths)
    string(JOIN "\n  " _onnxruntime_missing_message ${_onnxruntime_missing_paths})
    message(FATAL_ERROR "ONNX Runtime is incomplete under ${ONNXRUNTIME_ROOT}. Missing required path(s):\n  ${_onnxruntime_missing_message}")
endif()

if(WIN32)
    set(ONNXRUNTIME_RUNTIME_DLLS "${_onnxruntime_runtime_path}")
    set(_onnxruntime_shared_provider
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll")
    set(_onnxruntime_cuda_provider
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll")
    if(EXISTS "${_onnxruntime_shared_provider}" AND
       EXISTS "${_onnxruntime_cuda_provider}")
        list(APPEND ONNXRUNTIME_RUNTIME_DLLS
            "${_onnxruntime_shared_provider}"
            "${_onnxruntime_cuda_provider}")
    elseif(EXISTS "${_onnxruntime_shared_provider}" OR
           EXISTS "${_onnxruntime_cuda_provider}")
        message(FATAL_ERROR
            "ONNX Runtime GPU providers are incomplete under ${ONNXRUNTIME_ROOT}/lib. "
            "onnxruntime_providers_shared.dll and onnxruntime_providers_cuda.dll "
            "must both be present.")
    endif()
endif()

if(NOT TARGET OnnxRuntime::OnnxRuntime)
    add_library(OnnxRuntime::OnnxRuntime UNKNOWN IMPORTED)
    set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
        IMPORTED_LOCATION "${_onnxruntime_library_path}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")
endif()

set(OnnxRuntime_FOUND TRUE)

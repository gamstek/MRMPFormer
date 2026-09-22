if(NOT DEFINED SOURCE_DIR OR NOT DEFINED TARGET_DIR)
    message(FATAL_ERROR "SOURCE_DIR and TARGET_DIR are required")
endif()

set(_runtime_dlls onnxruntime.dll)
foreach(_provider IN ITEMS shared cuda)
    set(_provider_dll "onnxruntime_providers_${_provider}.dll")
    if(EXISTS "${SOURCE_DIR}/${_provider_dll}")
        list(APPEND _runtime_dlls "${_provider_dll}")
    endif()
endforeach()

foreach(_runtime_dll IN LISTS _runtime_dlls)
    if(NOT EXISTS "${TARGET_DIR}/${_runtime_dll}")
        message(FATAL_ERROR
            "Required ONNX Runtime DLL was not deployed: ${TARGET_DIR}/${_runtime_dll}")
    endif()
endforeach()

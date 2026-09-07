cmake_minimum_required(VERSION 3.20)
project(MRMPFormer VERSION 0.1.0 LANGUAGES C CXX)

option(BUILD_TESTS "Build MRMPFormer test executables" ON)
set(MRMPFORMER_JSON_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/json" CACHE PATH
    "Directory containing the nlohmann json.hpp header")

if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(OnnxRuntime REQUIRED)

if(NOT EXISTS "${MRMPFORMER_JSON_INCLUDE_DIR}/json.hpp")
    message(FATAL_ERROR
        "MRMPFORMER_JSON_INCLUDE_DIR must name a directory containing json.hpp")
endif()

add_library(mrmpformer SHARED
    src/api.cpp
    src/json_protocol.cpp
    src/onnx_inference.cpp
    src/signal_processing.cpp
    src/roi_renderer.cpp
    src/task_manager.cpp)
target_compile_features(mrmpformer PUBLIC cxx_std_20)
target_compile_definitions(mrmpformer PRIVATE MRMPFORMER_BUILD)
target_include_directories(mrmpformer
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${MRMPFORMER_JSON_INCLUDE_DIR}")
target_link_libraries(mrmpformer PRIVATE OnnxRuntime::OnnxRuntime)

if(WIN32)
    add_custom_command(TARGET mrmpformer POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            ${ONNXRUNTIME_RUNTIME_DLLS}
            "$<TARGET_FILE_DIR:mrmpformer>"
        COMMAND_EXPAND_LISTS)
endif()

if(MSVC)
    target_compile_options(mrmpformer PRIVATE /utf-8)
endif()

add_executable(batch_example examples/batch_example.c)
target_compile_features(batch_example PRIVATE c_std_11)
target_link_libraries(batch_example PRIVATE mrmpformer)
# The shared library is C++ even though the public ABI and example are C.
set_target_properties(batch_example PROPERTIES LINKER_LANGUAGE CXX)

add_executable(single_example examples/single_example.c)
target_compile_features(single_example PRIVATE c_std_11)
target_link_libraries(single_example PRIVATE mrmpformer)
set_target_properties(single_example PROPERTIES LINKER_LANGUAGE CXX)

if(BUILD_TESTS)
    include(CTest)
    enable_testing()

    add_executable(test_header_abi tests/test_header_abi.cpp)
    target_compile_features(test_header_abi PRIVATE cxx_std_20)
    target_link_libraries(test_header_abi PRIVATE mrmpformer)
    add_test(NAME header_abi COMMAND test_header_abi)

    add_executable(test_json_protocol tests/test_json_protocol.cpp src/json_protocol.cpp)
    target_compile_features(test_json_protocol PRIVATE cxx_std_20)
    target_include_directories(test_json_protocol PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${MRMPFORMER_JSON_INCLUDE_DIR}")
    add_test(NAME json_protocol COMMAND test_json_protocol)

    add_executable(test_signal_processing
        tests/test_signal_processing.cpp
        src/signal_processing.cpp
        src/roi_renderer.cpp)
    target_compile_features(test_signal_processing PRIVATE cxx_std_20)
    target_include_directories(test_signal_processing PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
    add_test(NAME signal_processing COMMAND test_signal_processing)

    add_executable(test_onnx_inference
        tests/test_onnx_inference.cpp
        src/onnx_inference.cpp)
    target_compile_features(test_onnx_inference PRIVATE cxx_std_20)
    target_include_directories(test_onnx_inference PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(test_onnx_inference PRIVATE OnnxRuntime::OnnxRuntime)
    target_compile_definitions(test_onnx_inference PRIVATE
        MRMPFORMER_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../model/checkpoint/mrmpformerv2.onnx")
    add_test(NAME onnx_inference COMMAND test_onnx_inference)

    add_executable(test_api tests/test_api.cpp)
    target_compile_features(test_api PRIVATE cxx_std_20)
    target_include_directories(test_api PRIVATE
        "${MRMPFORMER_JSON_INCLUDE_DIR}")
    target_link_libraries(test_api PRIVATE mrmpformer)
    target_compile_definitions(test_api PRIVATE
        MRMPFORMER_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../model/checkpoint/mrmpformerv2.onnx")
    add_test(NAME api COMMAND test_api)

    if(WIN32)
        add_test(NAME runtime_dlls
            COMMAND "${CMAKE_COMMAND}"
                "-DSOURCE_DIR=${ONNXRUNTIME_ROOT}/lib"
                "-DTARGET_DIR=$<TARGET_FILE_DIR:mrmpformer>"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyRuntimeDlls.cmake")
    endif()

endif()

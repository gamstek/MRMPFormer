cmake_minimum_required(VERSION 3.20)
project(MRMPFormerParity LANGUAGES CXX)
set(PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build/windows" CACHE PATH "Packaged SDK directory")
add_library(packaged_mrmpformer SHARED IMPORTED)
set_target_properties(packaged_mrmpformer PROPERTIES
    IMPORTED_IMPLIB "${PACKAGE_DIR}/mrmpformer.lib"
    IMPORTED_LOCATION "${PACKAGE_DIR}/mrmpformer.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${PACKAGE_DIR}")
add_executable(inference_parity_runner inference_parity_runner.cpp)
target_compile_features(inference_parity_runner PRIVATE cxx_std_20)
target_include_directories(inference_parity_runner PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../cpp/third_party/json")
target_link_libraries(inference_parity_runner PRIVATE packaged_mrmpformer)

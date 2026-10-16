cmake_minimum_required(VERSION 3.10)
project(yolo_tensorrt)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_compile_options(/utf-8)
endif ()

get_filename_component(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

# 1. 先创建库 target
add_library(${PROJECT_NAME} SHARED yolo_tensorrt.cpp)

# 2. 设置 include 目录
target_include_directories(${PROJECT_NAME} PRIVATE
        ${PROJECT_ROOT_DIR}/include
)

# 3. 定义导出宏
target_compile_definitions(${PROJECT_NAME} PRIVATE LIBFASTYOLO_EXPORTS)

# 4. 设置输出目录
set_target_properties(${PROJECT_NAME} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# --- 查找 TensorRT ---
find_path(TENSORRT_INCLUDE_DIR NvInferRuntime.h
        PATHS
        /usr/src/tensorrt/include
        /usr/include/x86_64-linux-gnu
        /usr/include
        "C:/Program Files/NVIDIA/TensorRT/include"
        REQUIRED)

find_library(NVINFER nvinfer
        PATHS
        /usr/src/tensorrt/lib
        /usr/lib/x86_64-linux-gnu
        "C:/Program Files/NVIDIA/TensorRT/lib"
        REQUIRED)

find_library(NVINFER_PLUGIN nvinfer_plugin
        PATHS
        /usr/src/tensorrt/lib
        /usr/lib/x86_64-linux-gnu
        "C:/Program Files/NVIDIA/TensorRT/lib"
        REQUIRED)

# --- 查找 CUDA ---
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
        PATHS
        /usr/local/cuda/include
        /usr/local/cuda-12*/include
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/include"
        REQUIRED)

find_library(CUDART cudart
        PATHS
        /usr/local/cuda/lib64
        /usr/local/cuda/targets/x86_64-linux/lib
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/lib/x64"
        REQUIRED)

find_library(CUBLAS cublas
        PATHS
        /usr/local/cuda/lib64
        /usr/local/cuda/targets/x86_64-linux/lib
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/*/lib/x64"
        REQUIRED)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${TENSORRT_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE
        ${NVINFER}
        ${NVINFER_PLUGIN}
        ${CUDART}
        ${CUBLAS}
        pthread
        dl
)

# --- 拷贝生成的库到 bin 目录 ---
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:${PROJECT_NAME}>
        ${CMAKE_BINARY_DIR}/bin
)

if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(${PROJECT_NAME} PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_RPATH "$ORIGIN"
    )
endif ()

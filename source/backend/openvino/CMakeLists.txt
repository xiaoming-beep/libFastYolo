cmake_minimum_required(VERSION 3.10)
project(yolo_openvino)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_compile_options(/utf-8)
endif ()

get_filename_component(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

# OpenVINO 安装根目录（pip 包的 site-packages/openvino，例如
#   Windows: D:\venv\Lib\site-packages\openvino
#   Linux:   /tmp/venv/lib/python3.12/site-packages/openvino
# 其中包含 cmake/OpenVINOConfig.cmake、include/、libs/（运行库）
set(OPENVINO_ROOT "" CACHE PATH "Path to OpenVINO install root (e.g. pip site-packages/openvino)")
if (NOT OPENVINO_ROOT)
    message(FATAL_ERROR
        "OPENVINO_ROOT is not set. Please specify it, e.g.\n"
        "  -DOPENVINO_ROOT=D:/venv/Lib/site-packages/openvino")
endif ()

# 1. 创建库 target
add_library(${PROJECT_NAME} SHARED yolo_openvino.cpp)

# 2. include 目录（OpenVINO 头文件 + 公共头文件）
target_include_directories(${PROJECT_NAME} PRIVATE
        ${OPENVINO_ROOT}/include
        ${PROJECT_ROOT_DIR}/include
)

# 3. 定义导出宏
target_compile_definitions(${PROJECT_NAME} PRIVATE LIBFASTYOLO_EXPORTS)

# 4. 输出目录
set_target_properties(${PROJECT_NAME} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# 5. 链接 OpenVINO 运行时
find_package(OpenVINO REQUIRED PATHS ${OPENVINO_ROOT} NO_DEFAULT_PATH)
target_link_libraries(${PROJECT_NAME} PRIVATE openvino::runtime)

# 6. 构建后自动部署 OpenVINO 运行库到输出目录，运行时无需手动设置
#    PATH / LD_LIBRARY_PATH。
#    OpenVINO 是插件式运行时（CPU/GPU/NPU 插件 + 格式前端 + TBB 均运行时 dlopen），
#    无法像 ONNX 后端那样静态链接成单个 so；这里只拷贝本项目实际用到的库：
#      - libopenvino 核心 + IR/ONNX 前端（yolo_openvino.cpp 从内存读 .onnx/.xml）
#      - CPU/GPU/NPU/AUTO 插件（对应 InferDevice::CPU/OPENCL/NPU/AUTO）
#      - TBB + hwloc（运行时依赖）
#    若需要 pytorch/tensorflow 等前端或全部插件，设 -DOPENVINO_COPY_ALL_RUNTIME=ON
option(OPENVINO_COPY_ALL_RUNTIME "Copy the full OpenVINO runtime libs dir instead of the minimal set" OFF)
# TBB 在 wheel 间有命名差异（旧版 tbb.dll / 新版 tbb12.dll + tbbbind），glob 兼容
file(GLOB OV_TBB_FILES
        ${OPENVINO_ROOT}/libs/tbb.dll
        ${OPENVINO_ROOT}/libs/tbb12.dll
        ${OPENVINO_ROOT}/libs/tbbbind*.dll
        ${OPENVINO_ROOT}/libs/tbbmalloc*.dll
)
# NPU 插件运行时依赖（openvino_intel_npu_plugin.dll 需要 compiler/VM runtime）
file(GLOB OV_NPU_FILES
        ${OPENVINO_ROOT}/libs/openvino_intel_npu_compiler*.dll
        ${OPENVINO_ROOT}/libs/openvino_intel_npu_vm_runtime.dll
)
if (WIN32)
    if (OPENVINO_COPY_ALL_RUNTIME)
        file(GLOB OV_RUNTIME_FILES ${OPENVINO_ROOT}/libs/*.dll)
    else ()
        file(GLOB OV_RUNTIME_FILES
                ${OPENVINO_ROOT}/libs/openvino.dll
                ${OPENVINO_ROOT}/libs/openvino_c.dll
                ${OPENVINO_ROOT}/libs/openvino_ir_frontend.dll
                ${OPENVINO_ROOT}/libs/openvino_onnx_frontend.dll
                ${OPENVINO_ROOT}/libs/openvino_intel_cpu_plugin.dll
                ${OPENVINO_ROOT}/libs/openvino_intel_gpu_plugin.dll
                ${OPENVINO_ROOT}/libs/openvino_intel_npu_plugin.dll
                ${OPENVINO_ROOT}/libs/openvino_auto_plugin.dll
                ${OPENVINO_ROOT}/libs/openvino_hetero_plugin.dll
                ${OV_TBB_FILES}
                ${OV_NPU_FILES}
        )
    endif ()
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OV_RUNTIME_FILES}
            ${CMAKE_BINARY_DIR}/bin
    )
elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    if (OPENVINO_COPY_ALL_RUNTIME)
        file(GLOB OV_RUNTIME_FILES ${OPENVINO_ROOT}/libs/lib*.so*)
    else ()
        file(GLOB OV_RUNTIME_FILES
                ${OPENVINO_ROOT}/libs/libopenvino.so*
                ${OPENVINO_ROOT}/libs/libopenvino_ir_frontend.so*
                ${OPENVINO_ROOT}/libs/libopenvino_onnx_frontend.so*
                ${OPENVINO_ROOT}/libs/libopenvino_intel_cpu_plugin.so*
                ${OPENVINO_ROOT}/libs/libopenvino_intel_gpu_plugin.so*
                ${OPENVINO_ROOT}/libs/libopenvino_intel_npu_plugin.so*
                ${OPENVINO_ROOT}/libs/libopenvino_auto_plugin.so*
                ${OPENVINO_ROOT}/libs/libopenvino_hetero_plugin.so*
                ${OPENVINO_ROOT}/libs/libtbb.so*
                ${OPENVINO_ROOT}/libs/libtbbmalloc.so*
                ${OPENVINO_ROOT}/libs/libtbbmalloc_proxy.so*
                ${OPENVINO_ROOT}/libs/libtbbbind*.so*
                ${OPENVINO_ROOT}/libs/libhwloc.so*
        )
    endif ()
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OV_RUNTIME_FILES}
            ${CMAKE_BINARY_DIR}/lib
    )
endif ()

# 7. RPATH：只保留 $ORIGIN，从 so 所在目录加载 OpenVINO 运行库。
#    不用 BUILD_RPATH（CMake 会自动追加链接目录的绝对路径，导致换机器/换位置失效），
#    用 BUILD_WITH_INSTALL_RPATH 让构建产物直接使用 INSTALL_RPATH。
if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(${PROJECT_NAME} PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH TRUE
    )
endif ()

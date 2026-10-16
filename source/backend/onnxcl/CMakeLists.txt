# onnxcl 后端：OpenCL 直读 ONNX 推理
#
# 特点：完全自包含。不依赖 MNN / TensorRT / ONNX Runtime，也【不依赖 OpenCL SDK】——
# OpenCL 走运行时加载（engine/include/onnxcl/cl_api.h），只要有显卡驱动自带的
# OpenCL 运行时（Windows 的 OpenCL.dll / Linux 的 libOpenCL.so.1）就能编译和运行。
# 所以 MinGW / MSVC / GCC 都能编，不需要导入库（OpenCL.lib）。
#
# 启用: cmake -DONNXCL_BACKEND=ON ..
cmake_minimum_required(VERSION 3.16)
project(yolo_onnxcl)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    # 源码是 UTF-8 中文注释
    add_compile_options(/utf-8)
endif ()

# 项目根（source/），公共头在 source/include
get_filename_component(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

set(ONNXCL_ENGINE ${CMAKE_CURRENT_SOURCE_DIR}/engine)
set(ONNXCL_KERNELS ${ONNXCL_ENGINE}/src/kernels)

# ---------------------------------------------------------------------------
# 1. 把 .cl 内核源码嵌进二进制
#    产物 kernels_generated.cpp 提供 onnxcl::embeddedKernelSource(name)，
#    这样运行时不需要旁边放 kernels 目录，单个 dll 就是完整的。
#    生成到 binary dir，不污染源码树。
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(ONNXCL_GEN_CPP ${CMAKE_CURRENT_BINARY_DIR}/kernels_generated.cpp)
file(GLOB ONNXCL_CL_FILES ${ONNXCL_KERNELS}/*.cl)
list(LENGTH ONNXCL_CL_FILES ONNXCL_CL_COUNT)
message(STATUS "onnxcl: 发现 ${ONNXCL_CL_COUNT} 个内核文件")

add_custom_command(
        OUTPUT ${ONNXCL_GEN_CPP}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_kernels.py
                ${ONNXCL_KERNELS} ${ONNXCL_GEN_CPP}
        DEPENDS ${ONNXCL_CL_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/gen_kernels.py
        COMMENT "嵌入 OpenCL 内核源码 -> kernels_generated.cpp"
        VERBATIM)
add_custom_target(onnxcl_kernels DEPENDS ${ONNXCL_GEN_CPP})

# ---------------------------------------------------------------------------
# 2. 后端库
#    适配层（yolo_onnxcl.cpp，实现 yolo_infer_* ABI）+ 引擎本体（engine/）
# ---------------------------------------------------------------------------
add_library(${PROJECT_NAME} SHARED
        # 适配层：把引擎接成 libFastYolo 的后端 ABI
        yolo_onnxcl.cpp
        # 引擎：OpenCL 运行时加载器
        ${ONNXCL_ENGINE}/src/cl/cl_api.cpp
        ${ONNXCL_ENGINE}/src/cl/cl_runtime.cpp
        # 引擎：ONNX 解析（手写 protobuf，零第三方依赖）
        ${ONNXCL_ENGINE}/src/onnx/onnx_loader.cpp
        # 引擎：图 IR + 形状推导 + 融合 pass
        ${ONNXCL_ENGINE}/src/graph/graph_build.cpp
        ${ONNXCL_ENGINE}/src/graph/shape_infer.cpp
        ${ONNXCL_ENGINE}/src/graph/fusion.cpp
        # 引擎：执行器
        ${ONNXCL_ENGINE}/src/exec/executor.cpp
        ${ONNXCL_ENGINE}/src/exec/cpu_ops.cpp
        # 生成物：内嵌的 OpenCL 内核源码
        ${ONNXCL_GEN_CPP}
)
add_dependencies(${PROJECT_NAME} onnxcl_kernels)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${ONNXCL_ENGINE}/include                  # 引擎对外头
        ${ONNXCL_ENGINE}/third_party/OpenCLHeaders  # OpenCL 头（自带，不需要 SDK）
        ${PROJECT_ROOT_DIR}/include               # 公共头（fast_yolo/yolo.h）
)

# 导出宏（和 onnx/MNN 后端保持一致）
target_compile_definitions(${PROJECT_NAME} PRIVATE LIBFASTYOLO_EXPORTS)

# 四个 yolo_infer_* ABI 函数的导出方式：
#   MSVC  : 顶层 CMakeLists 里 WIN32 分支已经设了 CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON，
#           会自动生成 .def 导出全部符号，这里不用管
#   MinGW : 没有"自动导出"这回事，必须显式开 --export-all-symbols，
#           否则 Java 侧 System.loadLibrary("FastYolo") 会 UnsatisfiedLinkError
#   Linux : 默认导出全部符号，不用管
if (MINGW)
    target_link_options(${PROJECT_NAME} PRIVATE -Wl,--export-all-symbols)
endif ()

set_target_properties(${PROJECT_NAME} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Windows 上去掉 MinGW 默认的 lib 前缀（顶层 FastYolo 已同样处理）。
# 理由：FastYolo.dll 是按文件名 import 后端库的，两者名字风格应该一致；
# 而且 Windows 本来不用 lib 前缀，统一掉可以避免部署时对不上文件名。
# MSVC 本来就不加前缀，不受影响。
if (WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
endif ()

# ---------------------------------------------------------------------------
# 3. 平台相关
#    OpenCL 只做运行时加载，不需要链接任何 OpenCL 导入库；
#    但 POSIX 上 dlopen 需要 libdl（Linux 2.34 之前是独立库）
# ---------------------------------------------------------------------------
if (UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})
endif ()

# 产物拷到 bin 目录（和 onnx 后端行为一致）
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:${PROJECT_NAME}>
        ${CMAKE_BINARY_DIR}/bin
)

# 可重定位 RPATH：libyolo_onnxcl.so 从自身所在目录加载依赖，
# 整个 lib 目录可整体复制/移动而不失效
if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(${PROJECT_NAME} PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_RPATH "$ORIGIN"
    )
endif ()

# ---------------------------------------------------------------------------
# 4. 集成验证程序（可选）
#    验证「模型字节 -> 后端 create/init -> Letterbox -> 原始输出」整条通路，
#    并把中间结果 dump 成文件，交给 tools/verify_backend.py 和 onnxruntime 对拍。
#    引擎的数值正确性由 Python 侧独立判定，本程序不做任何判断。
#    启用: cmake -DONNXCL_TEST=ON ..
# ---------------------------------------------------------------------------
option(ONNXCL_TEST "构建 onnxcl 集成验证程序" OFF)
if (ONNXCL_TEST)
    add_executable(test_onnxcl test_onnxcl.cpp)
    target_include_directories(test_onnxcl PRIVATE ${PROJECT_ROOT_DIR}/include)
    # 需要两个库：
    #   ${PROJECT_NAME}    = yolo_onnxcl，提供 yolo_infer_* 后端 ABI
    #   ${CMAKE_PROJECT_NAME} = FastYolo，提供 yolo::Letterbox（前处理在本程序里真实跑一遍）
    target_link_libraries(test_onnxcl PRIVATE ${PROJECT_NAME} ${CMAKE_PROJECT_NAME})
    set_target_properties(test_onnxcl PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif ()

# 板端(116 等 Ubuntu 20.04, cmake 3.16)也能编，无 3.16+ 独有语法，勿随意抬回 3.21
cmake_minimum_required(VERSION 3.16)
project(FastYolo)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-deprecated-enum-enum-conversion)
endif()
set(CMAKE_CXX_STANDARD 20)
#属性设置
if (WIN32)
    # 启用所有符号导出（自动生成.lib导入库）
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    # 或者自定义导出宏
    add_definitions(-DLIBFASTYOLO_EXPORTS)
endif ()

option(ONNX_BACKEND "onnx infer" OFF)
option(TENSORRT_BACKEND "tensorrt infer" OFF)
option(MNN_BACKEND "MNN infer" OFF)
option(OPENVINO_BACKEND "openvino infer" OFF)
option(RKNN_BACKEND "rknn infer (Rockchip NPU, aarch64 板端)" OFF)
option(ONNXCL_BACKEND "onnxcl: OpenCL 直读 ONNX 推理（自包含，无需 OpenCL SDK）" OFF)
option(JNI_BACKEND "Build with JNI bindings for Java" OFF)
option(PYTHON_BACKEND "Build with pure C API for Python SDK (ctypes)" OFF)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/source/include)
add_library(${PROJECT_NAME} SHARED
        source/utils/img_util.cpp
        source/core/nms.cpp
        source/core/yolo.cpp
        source/core/postprocess.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Windows 上去掉 MinGW 默认的 lib 前缀。
# Java 侧是 System.loadLibrary("FastYolo")，Windows 下会去找 FastYolo.dll；
# MinGW 默认产出 libFastYolo.dll，Java 就找不到（UnsatisfiedLinkError）。
# Windows 本来也不用 lib 前缀，所以这里统一去掉（MSVC 不受影响）。
if (WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
endif ()

# 可重定位 RPATH：libFastYolo.so 从自身所在目录加载后端库（libyolo_openvino.so 等），
# 整个 lib 目录可整体复制/移动而不会失效（不再依赖 CMake 追加的绝对路径）
if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(${PROJECT_NAME} PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH TRUE
    )
endif ()

# 在Windows上需要设置DEBUG_POSTFIX
if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /utf-8)
    set_target_properties(${PROJECT_NAME} PROPERTIES
            DEBUG_POSTFIX "d"
    )
endif ()

if (MNN_BACKEND)
    add_subdirectory(source/backend/MNN)
    target_link_libraries(${PROJECT_NAME} PRIVATE yolo_mnn)
elseif (TENSORRT_BACKEND)
    add_subdirectory(source/backend/TensorRT)
    target_link_libraries(${PROJECT_NAME} PRIVATE yolo_tensorrt)
elseif (OPENVINO_BACKEND)
    add_subdirectory(source/backend/openvino)
    target_link_libraries(${PROJECT_NAME} PRIVATE yolo_openvino)
elseif (RKNN_BACKEND)
    # 板端用法: cmake -DRKNN_BACKEND=ON -DRKNN_TEST=ON -DRKNN_SDK_DIR=<rknpu2根目录> ..
    add_subdirectory(source/backend/RKNN)
    target_link_libraries(${PROJECT_NAME} PRIVATE yolo_rknn)
elseif (ONNXCL_BACKEND)
    # onnx 由 OpenCL 直接执行（不要转 mnn/ort 格式）：
    #   cmake -DONNXCL_BACKEND=ON ..
    #   运行时只需显卡驱动自带的 OpenCL 运行时（OpenCL.dll / libOpenCL.so.1）
    add_subdirectory(source/backend/onnxcl)
    target_link_libraries(${PROJECT_NAME} PRIVATE yolo_onnxcl)
else ()
    add_subdirectory(source/backend/onnx)
    target_link_libraries(${PROJECT_NAME} PRIVATE yolo_onnx)
endif ()

# ByteTrack 多目标跟踪（集成在 YOLOInference::track() 中）
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/source/track/include)
target_sources(${PROJECT_NAME} PRIVATE
        source/track/ByteTrack/BYTETracker.cpp
        source/track/ByteTrack/KalmanFilter.cpp
        source/track/ByteTrack/lapjv.cpp
        source/track/ByteTrack/Object.cpp
        source/track/ByteTrack/Rect.cpp
        source/track/ByteTrack/STrack.cpp
)

# JNI 绑定（Java 调用 libFastYolo 的桥梁）
# 启用: cmake -DJNI_BACKEND=ON ..
# Java 项目位于 java/fast-yolo-jni/，头文件由 javac -h 生成
if (JNI_BACKEND)
    # 优先使用 -DJAVA_HOME= 传入的值，其次使用环境变量
    if (NOT JAVA_HOME)
        set(JAVA_HOME $ENV{JAVA_HOME})
    endif()
    if (NOT JAVA_HOME)
        message(FATAL_ERROR "JAVA_HOME not set. Use -DJAVA_HOME=/path/to/jdk or export JAVA_HOME")
    endif()

    # jni.h 里平台相关的部分（jni_md.h）在哪个子目录 —— 三个平台各不相同。
    # 之前这里写死成 include/linux，所以 Windows 上直接编不过。
    if (WIN32)
        set(_jni_md_subdir win32)
    elseif (APPLE)
        set(_jni_md_subdir darwin)
    else ()
        set(_jni_md_subdir linux)
    endif()

    set(JNI_INCLUDE_DIRS
        ${JAVA_HOME}/include
        ${JAVA_HOME}/include/${_jni_md_subdir}
    )

    # jvm 库位置也各平台不同：
    #   Windows: %JAVA_HOME%\lib\jvm.lib（JDK 自带导入库，MSVC 直接用）
    #            MinGW 下如果发行版没发导入库，可以直接链 bin\server\jvm.dll
    #   Linux:   $JAVA_HOME/lib/server/libjvm.so（部分发行版在 $JAVA_HOME/lib/<arch>/server）
    #   macOS:   $JAVA_HOME/lib/server/libjvm.dylib
    if (WIN32)
        if (EXISTS "${JAVA_HOME}/lib/jvm.lib")
            set(JVM_LIBRARY "${JAVA_HOME}/lib/jvm.lib")
        else ()
            find_file(JVM_LIBRARY NAMES jvm.dll
                PATHS ${JAVA_HOME}/bin/server ${JAVA_HOME}/jre/bin/server
                NO_DEFAULT_PATH)
        endif ()
    else ()
        file(GLOB _jvm_server_dirs ${JAVA_HOME}/lib/*/server)
        find_library(JVM_LIBRARY jvm
            PATHS ${JAVA_HOME}/lib/server ${JAVA_HOME}/lib ${_jvm_server_dirs}
            NO_DEFAULT_PATH)
    endif ()
    if (NOT JVM_LIBRARY)
        message(FATAL_ERROR "jvm 库未找到，JAVA_HOME=${JAVA_HOME}。"
                "Windows 应该能找到 lib/jvm.lib；Linux 应该在 lib/server/ 下有 libjvm.so")
    endif()
    set(JNI_LIBRARIES ${JVM_LIBRARY})

    target_include_directories(${PROJECT_NAME} PRIVATE ${JNI_INCLUDE_DIRS})
    target_sources(${PROJECT_NAME} PRIVATE source/jni/yolo_jni.cpp)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${JNI_LIBRARIES})
    message(STATUS "JNI 已启用, include: ${JNI_INCLUDE_DIRS}, jvm: ${JVM_LIBRARY}")
endif ()

# Python 绑定（纯 C 标量 API，供 python/fastyolo SDK 经 ctypes 调用）
# 启用: cmake -DPYTHON_BACKEND=ON ..
# 不依赖任何 Python 头文件/库，产物与 Python 版本无关
if (PYTHON_BACKEND)
    target_sources(${PROJECT_NAME} PRIVATE source/python/yolo_c_api.cpp)
    message(STATUS "Python C API enabled (fastyolo_c.h)")
endif ()

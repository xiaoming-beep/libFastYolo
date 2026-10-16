cmake_minimum_required(VERSION 3.10)
project(yolo_rknn)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_compile_options(/utf-8)
endif ()

get_filename_component(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

# RKNN SDK 位置（include/rknn_api.h + librknnrt.so）
# 板端一般传: -DRKNN_SDK_DIR=/home/hzhy/rknn_yolo/3rdparty/rknpu2 -DRKNN_LIBRARY=.../Linux/aarch64/librknnrt.so
set(RKNN_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib" CACHE PATH "rknn toolkit2 SDK 根目录")
set(RKNN_INCLUDE_DIR "${RKNN_SDK_DIR}/include" CACHE PATH "rknn_api.h 所在目录")
set(RKNN_LIBRARY "${RKNN_SDK_DIR}/lib/librknnrt.so" CACHE FILEPATH "librknnrt.so 路径")

# 1. 创建库 target
add_library(${PROJECT_NAME} SHARED yolo_rknn.cpp)

# 2. include 目录（RKNN 头文件 + 公共头文件）
target_include_directories(${PROJECT_NAME} PRIVATE
        ${RKNN_INCLUDE_DIR}
        ${PROJECT_ROOT_DIR}/include
)

# 3. 定义导出宏
target_compile_definitions(${PROJECT_NAME} PRIVATE LIBFASTYOLO_EXPORTS)

# 4. 输出目录（Windows: .dll->bin；Linux: .so->lib）
set_target_properties(${PROJECT_NAME} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if (WIN32)
    message(FATAL_ERROR "RKNN 后端仅支持 Linux(aarch64) 板端，Windows 不支持")
endif ()

# 5. 链接 RKNN 运行时
target_link_libraries(${PROJECT_NAME} PRIVATE
        ${RKNN_LIBRARY}
        pthread
        dl
)

# 6. 拷贝 librknnrt.so 到输出目录，确保运行时可加载
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${RKNN_LIBRARY}
        ${CMAKE_BINARY_DIR}/lib/librknnrt.so
)

# 7. RPATH：libyolo_rknn.so 与 librknnrt.so 同目录（$ORIGIN）
set_target_properties(${PROJECT_NAME} PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_RPATH "$ORIGIN"
)

# 8. 板端自测工具（可选）：cmake -DRKNN_TEST=ON ..
# 用法: test_rknn <model.rknn> <image.jpg> [帧数]
option(RKNN_TEST "Build rknn backend self-test tool" OFF)
if (RKNN_TEST)
    add_executable(test_rknn test_rknn.cpp)
    target_compile_definitions(test_rknn PRIVATE FASTYOLO_RKNN_TEST_STB_IMPL)
    target_include_directories(test_rknn PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_ROOT_DIR}/include
            ${RKNN_INCLUDE_DIR}
    )
    set_target_properties(test_rknn PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            INSTALL_RPATH "$ORIGIN/../lib"
            BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    )
    target_link_libraries(test_rknn PRIVATE FastYolo pthread dl)
endif ()

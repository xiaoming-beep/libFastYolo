cmake_minimum_required(VERSION 3.10)
project(yolo_mnn)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_compile_options(/utf-8)
endif ()

get_filename_component(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

# 1. 创建库 target
add_library(${PROJECT_NAME} SHARED yolo_mnn.cpp)

# 2. include 目录（本地 MNN 头文件 + 公共头文件）
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PROJECT_ROOT_DIR}/include
)

# 3. 定义导出宏
target_compile_definitions(${PROJECT_NAME} PRIVATE LIBFASTYOLO_EXPORTS)

# 4. 输出目录（Windows: .dll->bin, .lib->lib；Linux: .so->lib）
set_target_properties(${PROJECT_NAME} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if (WIN32)
    # Windows: 链接 MNN 导入库(MNN.lib)，运行时需 MNN.dll 与产物同目录
    set(MNN_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/lib/MNN.lib)
    set(MNN_DLL ${CMAKE_CURRENT_SOURCE_DIR}/lib/MNN.dll)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${MNN_LIBRARY})
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MNN_DLL}
            ${CMAKE_BINARY_DIR}/bin/MNN.dll
    )
else ()
    # 5. 链接 MNN 库（Linux）
    set(MNN_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/lib/libMNN.so)
    target_link_libraries(${PROJECT_NAME} PRIVATE
            ${MNN_LIBRARY}
            pthread
            dl
    )

    # 6. 拷贝 libMNN.so 到输出目录，确保运行时加载
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MNN_LIBRARY}
            ${CMAKE_BINARY_DIR}/lib/libMNN.so
    )

    # 7. RPATH
    set_target_properties(${PROJECT_NAME} PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_RPATH "$ORIGIN"
    )
endif ()

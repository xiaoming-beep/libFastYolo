cmake_minimum_required(VERSION 3.16)
project(vehicle_collector)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_compile_options(/utf-8)
endif ()

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio)

# libFastYolo 位置（板端默认布局：与本工程同级）
set(FY_ROOT "/home/hzhy/projects/libFastYolo" CACHE PATH "libFastYolo 仓库根目录")
set(FY_LIB_DIR "${FY_ROOT}/build/lib" CACHE PATH "libFastYolo.so 所在目录")

add_executable(vehicle_collector vehicle_collector.cpp)

target_include_directories(vehicle_collector PRIVATE
        ${FY_ROOT}/source/include
        ${OpenCV_INCLUDE_DIRS}
)

# OpenCV 动态库路径（/usr/local）
target_link_directories(vehicle_collector PRIVATE ${FY_LIB_DIR} ${OpenCV_LIB_DIR})

target_link_directories(vehicle_collector PRIVATE ${FY_LIB_DIR})

target_link_libraries(vehicle_collector PRIVATE
        FastYolo
        ${OpenCV_LIBS}
        pthread
)

set_target_properties(vehicle_collector PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        INSTALL_RPATH "${FY_LIB_DIR}"
        BUILD_RPATH "${FY_LIB_DIR}"
)

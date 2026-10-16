cmake_minimum_required(VERSION 3.10)
project(yolo_onnx)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_compile_options(/utf-8)
endif ()

# 获取项目根目录
get_filename_component(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)

# 1. 先创建库 target
add_library(${PROJECT_NAME} SHARED yolo_onnx.cpp)

# 2. 设置 include 目录
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include          # onnx 内部头文件
        ${PROJECT_ROOT_DIR}/include                  # 公共头文件
)

# 3. 定义导出宏
target_compile_definitions(${PROJECT_NAME} PRIVATE LIBFASTYOLO_EXPORTS)

# 4. 设置输出目录
set_target_properties(${PROJECT_NAME} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# --- 跨平台库路径配置 ---
if (WIN32)
    # Windows 使用动态库
    set(ONNXRUNTIME_LIB "${CMAKE_CURRENT_SOURCE_DIR}/lib/dll/onnxruntime.lib")
    set(ONNXRUNTIME_DLL "${CMAKE_CURRENT_SOURCE_DIR}/lib/dll/onnxruntime.dll")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_LIB})

    # 复制 DLL
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ONNXRUNTIME_DLL}
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )

elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    # 选择优先级：
    #   1. lib/linux-gpu 存在 → 链接 ONNX Runtime GPU 共享库（CUDA EP，运行时按 config->device 启用，
    #      无 GPU/缺 cuDNN 时自动回退 CPU）。运行时依赖：CUDA 12 的 libcublas/cublasLt/cudart + libcudnn.so.9
    #   2. lib/linux/libonnxruntime.a 存在 → 链接 CPU 静态库（原有行为）
    #   3. -DONNXRUNTIME_SHARED_LIB=/path/libonnxruntime.so → 指定共享库（如 pip 安装版）
    set(ONNXRUNTIME_GPU_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/linux-gpu")
    set(ONNXRUNTIME_STATIC_LIB "${CMAKE_CURRENT_SOURCE_DIR}/lib/linux/libonnxruntime.a")
    option(ONNX_GPU "Linux: use ONNX Runtime GPU shared libs (CUDA EP) from lib/linux-gpu when present" ON)
if (ONNX_GPU AND EXISTS "${ONNXRUNTIME_GPU_LIB_DIR}/libonnxruntime.so")
        file(GLOB ONNXRUNTIME_GPU_REAL_LIB "${ONNXRUNTIME_GPU_LIB_DIR}/libonnxruntime.so.*.*")
        get_filename_component(ONNXRUNTIME_GPU_REAL_LIB_NAME ${ONNXRUNTIME_GPU_REAL_LIB} NAME)
        target_link_libraries(${PROJECT_NAME} PRIVATE "${ONNXRUNTIME_GPU_LIB_DIR}/libonnxruntime.so")
        target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_ORT_CUDA_EP=1)
        # 产物运行时经 $ORIGIN 加载：libonnxruntime.so.1、providers_cuda/shared 随 libyolo_onnx.so 一起分发
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_GPU_REAL_LIB} ${ONNXRUNTIME_GPU_LIB_DIR}/libonnxruntime_providers_cuda.so
                ${ONNXRUNTIME_GPU_LIB_DIR}/libonnxruntime_providers_shared.so
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
                COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${ONNXRUNTIME_GPU_REAL_LIB_NAME} $<TARGET_FILE_DIR:${PROJECT_NAME}>/libonnxruntime.so.1
        )
        # providers_cuda.so 原始构建没有 RUNPATH，其依赖 libcudnn.so.9 只按系统路径搜索。
        # 打 $ORIGIN 后 cuDNN 的 .so 与本库同目录放置即可零环境变量自包含部署
        # （patchelf 缺失时跳过，改用 apt 安装 libcudnn9 或 LD_LIBRARY_PATH 指向 cuDNN 目录）。
        find_program(PATCHELF_EXE patchelf)
        if (PATCHELF_EXE)
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND}
                    -DPATCHELF_EXE=${PATCHELF_EXE}
                    -DLIBFILE=$<TARGET_FILE_DIR:${PROJECT_NAME}>/libonnxruntime_providers_cuda.so
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/patch_providers_rpath.cmake
            )
        else ()
            message(WARNING "patchelf not found: 产物目录的 libonnxruntime_providers_cuda.so 未打 \$ORIGIN，"
                            "cuDNN 需经系统路径或 LD_LIBRARY_PATH 提供")
        endif ()
        # cuDNN 9 是 providers_cuda.so 的运行时依赖，缺了会静默回退 CPU。
        # 注意必须选 CUDA 12 变体的 cuDNN 9：cuDNN 子库（engines_precompiled 等）内部
        # 按硬编码 soname dlopen cuBLAS —— cu13 变体要 libcublasLt.so.13、cuda11 变体要
        # .so.11，与本 CUDA 12 构建不匹配（表现为 "Invalid handle. Cannot load symbol
        # cublasLtCreate" 后 abort）。另外 cuDNN >= 9.11 已剔除 Pascal(sm_61) 内核，
        # GTX 10 系显卡需 cuDNN <= 9.10.x（如 nvidia-cudnn-cu12==9.10.2.21）。
        # 顺序：-DONNX_CUDNN_LIB_DIR=... → lib/linux-gpu 自身 → lib/linux-cudnn-cu12
        #       → 仓库自带的 python/fastyolo/libs → pip/CUDA toolkit/apt 等常见位置
        set(ONNX_CUDNN_LIB_DIR "" CACHE PATH "Directory containing libcudnn.so.9 (empty = auto-detect)")
        find_program(CMAKE_GREP_EXE grep)
        # 校验目录里的 cuDNN 是 CUDA 12 变体（engines_precompiled 内含 libcublasLt.so.12）
        function(_ort_cudnn_is_cuda12 _dir _out)
                set(${_out} ON PARENT_SCOPE)
                set(_ep "${_dir}/libcudnn_engines_precompiled.so.9")
                if (EXISTS "${_ep}" AND CMAKE_GREP_EXE)
                        execute_process(COMMAND ${CMAKE_GREP_EXE} -a -q "libcublasLt.so.12" "${_ep}"
                                RESULT_VARIABLE _r OUTPUT_QUIET ERROR_QUIET)
                        set(${_out} OFF PARENT_SCOPE)
                        if (_r EQUAL 0)
                                set(${_out} ON PARENT_SCOPE)
                        endif ()
                endif ()
        endfunction()
        if (NOT ONNX_CUDNN_LIB_DIR)
            foreach (_cudnn_cand "${ONNXRUNTIME_GPU_LIB_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/lib/linux-cudnn-cu12" "${CMAKE_CURRENT_SOURCE_DIR}/../../../python/fastyolo/libs")
                if (EXISTS "${_cudnn_cand}/libcudnn.so.9")
                    set(ONNX_CUDNN_LIB_DIR "${_cudnn_cand}")
                    break()
                endif ()
            endforeach()
        endif ()
        if (NOT ONNX_CUDNN_LIB_DIR)
            file(GLOB _cudnn_hits
                    "/usr/lib/x86_64-linux-gnu/libcudnn.so.9"
                    "/usr/local/cuda*/lib64/libcudnn.so.9"
                    "$ENV{HOME}/.local/lib/python*/site-packages/nvidia/cudnn/lib/libcudnn.so.9"
            )
            foreach (_cudnn_first ${_cudnn_hits})
                get_filename_component(_cudnn_dir "${_cudnn_first}" DIRECTORY)
                _ort_cudnn_is_cuda12("${_cudnn_dir}" _cudnn_ok)
                if (_cudnn_ok)
                    set(ONNX_CUDNN_LIB_DIR "${_cudnn_dir}")
                    break()
                endif ()
            endforeach()
        endif ()
        if (ONNX_CUDNN_LIB_DIR)
            _ort_cudnn_is_cuda12("${ONNX_CUDNN_LIB_DIR}" _cudnn_ok)
            if (NOT _cudnn_ok)
                message(FATAL_ERROR "ONNX_CUDNN_LIB_DIR=${ONNX_CUDNN_LIB_DIR} 不是 CUDA 12 变体的 cuDNN 9"
                        "（engines_precompiled 不含 libcublasLt.so.12）。请用 nvidia-cudnn-cu12 版 cuDNN 9"
                        "（Pascal 显卡需 <= 9.10.x），或改用 -DONNX_CUDNN_LIB_DIR 指定正确目录")
            endif ()
        endif ()
        if (ONNX_CUDNN_LIB_DIR)
            # 通配 libcudnn 相关 .so.9[.x]，展开为实际文件列表后整体拷入产物目录
            file(GLOB _cudnn_libs "${ONNX_CUDNN_LIB_DIR}/libcudnn*.so.9" "${ONNX_CUDNN_LIB_DIR}/libcudnn*.so.9.*")
            list(LENGTH _cudnn_libs _cudnn_count)
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${_cudnn_libs}
                    $<TARGET_FILE_DIR:${PROJECT_NAME}>
            )
            message(STATUS "Copying cuDNN 9 (${_cudnn_count} libs) from: ${ONNX_CUDNN_LIB_DIR}")
        else ()
            message(WARNING "libcudnn.so.9 not found: CUDA EP will fall back to CPU at runtime. "
                            "Install cuDNN 9 or set -DONNX_CUDNN_LIB_DIR=/path/to/dir")
        endif ()
        # providers_cuda / cuDNN 的运行时依赖还包括 CUDA 12 的 cublasLt/cublas/cudart/
        # nvJitLink/curand/cufft。cuDNN 子库经 dlopen("libcublasLt.so.12") 加载 cuBLAS，
        # dlopen 会先搜 exe/调用方的 RUNPATH，因此一并拷入产物目录即可自包含部署。
        file(GLOB _cuda_rt_dirs "/usr/local/cuda*/targets/x86_64-linux/lib" "/usr/local/cuda*/lib64")
        set(_cuda_rt_dir "")
        foreach (_d ${_cuda_rt_dirs})
            if (EXISTS "${_d}/libcublasLt.so.12")
                get_filename_component(_cuda_rt_dir "${_d}" ABSOLUTE)
                break()
            endif ()
        endforeach ()
        if (_cuda_rt_dir)
            foreach (_so libcublasLt.so.12 libcublas.so.12 libcudart.so.12 libnvJitLink.so.12 libcurand.so.10 libcufft.so.11)
                file(GLOB _rt_matches "${_cuda_rt_dir}/${_so}.*")
                set(_rt_real "")
                foreach (_f ${_rt_matches})
                    get_filename_component(_fn "${_f}" NAME)
                    if (NOT _fn STREQUAL _so)
                        set(_rt_real "${_f}")
                        break()
                    endif ()
                endforeach ()
                if (_rt_real)
                    get_filename_component(_rt_fn "${_rt_real}" NAME)
                    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_rt_real}" $<TARGET_FILE_DIR:${PROJECT_NAME}>/
                            COMMAND ${CMAKE_COMMAND} -E create_symlink "${_rt_fn}" $<TARGET_FILE_DIR:${PROJECT_NAME}>/${_so}
                    )
                else ()
                    message(WARNING "CUDA runtime lib ${_so} not found under ${_cuda_rt_dir}")
                endif ()
            endforeach ()
            message(STATUS "Copying CUDA 12 runtime libs from: ${_cuda_rt_dir}")
        else ()
            message(WARNING "CUDA 12 runtime libs (libcublasLt.so.12 etc.) not found under /usr/local/cuda*;"
                            " CUDA EP 依赖系统路径提供 cuBLAS/cuDNN 才能启用")
        endif ()
        message(STATUS "Using ONNX Runtime GPU (CUDA EP): ${ONNXRUNTIME_GPU_LIB_DIR}")
    elseif (EXISTS ${ONNXRUNTIME_STATIC_LIB})
        target_link_libraries(${PROJECT_NAME} PRIVATE
                ${ONNXRUNTIME_STATIC_LIB}
                pthread
                dl
        )
        message(STATUS "Using static ONNX Runtime: ${ONNXRUNTIME_STATIC_LIB}")
    elseif (DEFINED ONNXRUNTIME_SHARED_LIB AND EXISTS ${ONNXRUNTIME_SHARED_LIB})
        target_link_libraries(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_SHARED_LIB})
        # 复制运行时 .so 到输出目录（libFastYolo 经 $ORIGIN RPATH 加载）
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ONNXRUNTIME_SHARED_LIB}
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
        )
        message(STATUS "Using shared ONNX Runtime: ${ONNXRUNTIME_SHARED_LIB}")
    else ()
        message(FATAL_ERROR "ONNX Runtime library not found (static: ${ONNXRUNTIME_STATIC_LIB}, set -DONNXRUNTIME_SHARED_LIB=... for shared)")
    endif ()
endif ()

# 复制生成的库到 bin 目录
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:${PROJECT_NAME}>
        ${CMAKE_BINARY_DIR}/bin
)

# Linux RPATH 设置（虽然静态链接后不需要，但保留也没坏处）
if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(${PROJECT_NAME} PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_RPATH "$ORIGIN"
    )
endif ()
# 给 libonnxruntime_providers_cuda.so 打 $ORIGIN RUNPATH，
# 使同目录放置的 cuDNN .so 可被零环境变量解析（自包含部署）。
# 用法: cmake -DPATCHELF_EXE=<path> -DLIBFILE=<path> -P patch_providers_rpath.cmake
# 注：必须经此脚本传 "$ORIGIN"，直接写在 add_custom_command 里会被
# make/ninja 的 $ 转义规则破坏。
execute_process(
        COMMAND ${PATCHELF_EXE} --set-rpath "$ORIGIN" ${LIBFILE}
        RESULT_VARIABLE rc
        OUTPUT_QUIET
)
if (NOT rc EQUAL 0)
    message(FATAL_ERROR "patchelf --set-rpath $ORIGIN ${LIBFILE} 失败: ${rc}")
endif ()

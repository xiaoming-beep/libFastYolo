#!/usr/bin/env bash
# ================================================================
# 构建 Python SDK（Linux + ONNX Runtime 后端）
#
# 用法:
#   bash python/scripts/build_linux.sh          # CPU 版（默认，静态链接，兼容 slim wheel）
#   bash python/scripts/build_linux.sh --gpu    # CUDA 版（ONNX Runtime GPU 共享库）
#
# 前置条件:
#   1. GCC >= 11（C++20）、CMake >= 3.21
#   2. CPU 版：ONNX Runtime 静态库随仓库自带（source/backend/onnx/lib/linux/libonnxruntime.a）
#   3. GPU 版：source/backend/onnx/lib/linux-gpu 的 GPU 共享库随仓库自带；
#      目标机运行时需要 CUDA 12 运行时（libcublas/cublasLt/cudart，装 CUDA 12 即有）
#      + libcudnn.so.9（Pascal/Maxwell 老卡需 cuDNN 9.1，详见根目录《编译指南.md》）
#
# 产物: python/fastyolo/libs/ ← 由 collect_artifacts.py 收集
#   CPU 版: libFastYolo.so + libyolo_onnx.so（静态链接，无额外运行时 → 精简 wheel 可用）
#   GPU 版: 另含 libonnxruntime.so.1.24.4 / providers_cuda / providers_shared（约 340MB，
#           推荐目录式分发；slim wheel 不打包这些运行时，详见《编译指南.md》第 7 节）
# ================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPU=OFF
COLLECT_ARGS=("--backend" "onnx")
if [[ "${1:-}" == "--gpu" ]]; then
    GPU=ON
    COLLECT_ARGS+=("--runtime" "full")
    echo "[INFO] GPU 模式：链接 ONNX Runtime CUDA EP（lib/linux-gpu）"
fi

echo "[INFO] 配置 + 构建（ONNX 后端 ONNX_GPU=$GPU + Python C API）"
cmake -S . -B build-py -DONNX_BACKEND=ON -DONNX_GPU="$GPU" -DPYTHON_BACKEND=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-py -j"$(nproc)"

echo "[INFO] 收集产物"
python3 python/scripts/collect_artifacts.py build-py "${COLLECT_ARGS[@]}"

echo
echo "[OK] 构建完成。使用:"
echo "    cd python"
echo "    python3 -m pytest tests"
echo "    python3 examples/demo.py --model 模型.onnx --image 图片.jpg"
if [[ "$GPU" == "ON" ]]; then
    echo "    GPU 推理: python3 examples/demo.py --model 模型.onnx --image 图片.jpg --device CUDA"
fi

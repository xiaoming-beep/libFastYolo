# 常用构建命令速查

> 后端具体说明见 [backend/](../backend/README.md)。

## Linux

```bash
# ---- ONNX（默认）----
cmake -S . -B b -DONNX_GPU=OFF                      # CPU 静态
cmake -S . -B b                                     # CUDA（默认，有 lib/linux-gpu 即启用）
cmake --build b -j$(nproc)

# ---- 其他后端（六选一）----
cmake -S . -B b -DOPENVINO_BACKEND=ON -DOPENVINO_ROOT=$OV     # OpenVINO
cmake -S . -B b -DTENSORRT_BACKEND=ON                         # TensorRT（engine 模型）
cmake -S . -B b -DMNN_BACKEND=ON                              # MNN
cmake -S . -B b -DONNXCL_BACKEND=ON                           # onnxcl（OpenCL 直读 ONNX）
cmake -S . -B b -DRKNN_BACKEND=ON -DRKNN_SDK_DIR=<rknpu2> \
      -DRKNN_LIBRARY=<.../librknnrt.so>                       # RKNN（aarch64 板端）

# ---- 叠加绑定 ----
cmake -S . -B b -DJNI_BACKEND=ON -DJAVA_HOME=$JAVA_HOME
cmake -S . -B b -DPYTHON_BACKEND=ON
cmake -S . -B b -DJNI_BACKEND=ON -DPYTHON_BACKEND=ON          # 全部叠加
```

## Windows（cmd / VS 生成器）

```bat
cmake -B b -G "Visual Studio 17 2022" -DJNI_BACKEND=ON
cmake --build b --config Release -j
:: Python SDK 一把梭：
python\scripts\build_windows.bat

:: onnxcl（MinGW，无需 OpenCL SDK）
cmake -S . -B build-onnxcl -G "MinGW Makefiles" -DONNXCL_BACKEND=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-onnxcl -j4
```

## 后端自测工具

```bash
# onnxcl 集成验证（-DONNXCL_TEST=ON）
./build-onnxcl/bin/test_onnxcl <model.onnx> <bgr.raw> <w> <h> <输出前缀>
python source/backend/onnxcl/tools/verify_backend.py <model.onnx> <输出前缀>

# RKNN 板端自测（-DRKNN_TEST=ON）
LD_LIBRARY_PATH=lib ./bin/test_rknn <model.rknn> <img.jpg> 300 predict|track
```

## 产物收集 / wheel

```bash
python3 python/scripts/collect_artifacts.py b --backend onnx --runtime full [--cudnn-lib <dir>]
cd python && python3 scripts/build_wheel.py [--full]
```

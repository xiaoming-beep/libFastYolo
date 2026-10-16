# OpenVINO 后端

> 顶层开关：`-DOPENVINO_BACKEND=ON -DOPENVINO_ROOT=<openvino 安装目录>`
> 适配层：`source/backend/openvino/yolo_openvino.cpp`。

OpenVINO 适合 Intel 平台（CPU / NPU / 核显 GPU）。运行库由 POST_BUILD 拷到产物目录并
打 `RPATH=$ORIGIN`，运行无需 `LD_LIBRARY_PATH`。**不支持 IR `.xml + .bin` 双文件**，
只吃 ONNX 单文件字节流。

---

## 1. Linux

```bash
pip install openvino   # 或用独立安装的 OpenVINO
OV=$(python3 -c "import openvino,os;print(os.path.dirname(openvino.__file__))")
cmake -S . -B build -DOPENVINO_BACKEND=ON -DOPENVINO_ROOT="$OV" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

- 设备映射：`CPU→CPU`、`NPU→NPU`、`OPENCL→GPU`、`AUTO→AUTO`，其余（含 CUDA）回退 CPU 并打日志。
- `INFERENCE_NUM_THREADS` 仅 CPU 插件支持（后端已按设备条件设置）。

## 2. Windows（Python SDK 默认路线）

直接跑脚本（自动探测 pip openvino、构建、收集产物）：

```bat
cd C:\libFastYolo
python\scripts\build_windows.bat
```

等价手动命令：

```bat
cmake -S . -B build-py -G "Visual Studio 17 2022" -DOPENVINO_BACKEND=ON -DPYTHON_BACKEND=ON -DOPENVINO_ROOT="<pip site-packages>\openvino"
cmake --build build-py --config Release
python python\scripts\collect_artifacts.py build-py --backend openvino
```

## 3. 与 JNI / Python 组合

```bash
# Linux，OpenVINO + JNI
cmake -S . -B build -DOPENVINO_BACKEND=ON -DOPENVINO_ROOT="$OV" \
      -DJNI_BACKEND=ON -DJAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
cmake --build build -j$(nproc)
```

Windows + Python 见 [Python SDK](../build/python-sdk.md)。运行时策略：OpenVINO 运行时 DLL
不随 wheel 复制，加载时自动从已安装的 `openvino` pip 包获取（定位优先级：
`FASTYOLO_OPENVINO_ROOT` 环境变量 → 自动探测已安装的 openvino 包）。

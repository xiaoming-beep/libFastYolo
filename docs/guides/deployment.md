# 产物清单与部署目录布局

> 核心原则：`libFastYolo`（主库）与 `libyolo_<后端>`（后端库）**放同一目录**，
> 主库经 `RPATH=$ORIGIN` 从自身所在目录加载后端库，整个 lib 目录可整体复制/移动而不失效。

---

## 1. Linux — ONNX-CUDA 完整部署包（示例）

```
lib/
├── libFastYolo.so                        # 主库（核心 + ByteTrack + 可选 JNI/C-API）
├── libyolo_onnx.so                       # ONNX 后端适配层
├── libonnxruntime.so.1 → .so.1.x.y       # ORT GPU 主库（版本与 providers 同批）
├── libonnxruntime_providers_cuda.so      # CUDA EP（构建时已打 RUNPATH=$ORIGIN）
├── libonnxruntime_providers_shared.so    # EP 注册桥
├── libcublasLt.so.12 / libcublas.so.12 / libcudart.so.12 / libnvJitLink.so.12 ...
└── libcudnn*.so.9                        # 可选：自包含时放入
```

系统级依赖（不在包内）：NVIDIA 驱动 + CUDA 12 运行库（装 CUDA 12 即有）。
cuDNN 可随包分发，也可由系统 apt / `LD_LIBRARY_PATH` 提供，见
[Python SDK](../build/python-sdk.md#4-onnx-cuda-的部署形态推荐目录式而非-slim-wheel)。

## 2. Linux — ONNX-CPU 静态版

```
lib/
├── libFastYolo.so
└── libyolo_onnx.so          # onnxruntime 已静态链入，无其他依赖
```

## 3. Linux — onnxcl（OpenCL）

```
lib/
├── libFastYolo.so
└── libyolo_onnxcl.so        # 引擎已内嵌 OpenCL 内核，运行时只需系统 libOpenCL.so.1
```

## 4. Linux — RKNN（板端）

```
lib/
├── libFastYolo.so
├── libyolo_rknn.so
└── librknnrt.so             # POST_BUILD 自动拷入，RPATH=$ORIGIN
bin/
└── test_rknn                # 可选自测工具（-DRKNN_TEST=ON）
```

## 5. Windows — ONNX CPU

```
bin\Release\
├── FastYolo.dll          # 或 FastYolod.dll（Debug）
├── yolo_onnx.dll
└── onnxruntime.dll
```

## 6. Windows — onnxcl

```
bin\Release\
├── FastYolo.dll
└── yolo_onnxcl.dll       # 运行时只需系统 OpenCL.dll
```

MinGW 产物还需 `libgcc_s_seh-1.dll` / `libstdc++-6.dll` 在 PATH 或同目录（MSVC 无此问题）。

## 7. 各平台产物输出目录

| 生成器 | 产物位置 |
|---|---|
| Makefiles / Ninja（Linux） | `<build>/lib/`（全部 `.so`）+ POST_BUILD 复制一份 `<build>/bin/libyolo_*.so` |
| Visual Studio（多配置） | `<build>/bin/Release/`（`FastYolo.dll`、`yolo_*.dll`、运行时 dll）；Debug 主库带 `d` 后缀 |

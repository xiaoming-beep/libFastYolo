# Linux 编译

> 通用前提、CMake 配置、产物布局。**后端特有的编译参数见 [backend/](../backend/README.md) 下对应文档。**

---

## 1. 前置条件

| 项 | 要求 |
|---|---|
| CMake | ≥ 3.16（板端 Ubuntu 20.04 也能编）；日常用 ≥ 3.21 |
| 编译器 | 支持 C++20：**GCC ≥ 11** |
| ONNX Runtime 库 | 仓库按平台放置：CPU 静态 `.a`（`lib/linux/`）、GPU 共享 `.so`（`lib/linux-gpu/`）。工作副本里若目录为空，从官方 tgz 解包放入，版本配套见 [版本矩阵](../appendix/version-matrix.md) |
| OpenCV（仅示例程序） | 编译 `examples/vehicle_collector` 才需要 OpenCV 4 |

---

## 2. 默认配置（ONNX 后端）

```bash
cd /path/to/libFastYolo
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

- 默认走 ONNX 后端；检测到 `source/backend/onnx/lib/linux-gpu/` 时自动链接 CUDA EP，
  同一二进制同时具备 CPU/CUDA 能力（`-DONNX_GPU=OFF` 强制 CPU 静态库）。
- 详细开关与库布局见 [backend/onnx.md](../backend/onnx.md)。

## 3. 叠加绑定层

```bash
# JNI + Python C API 可叠加到任意后端
cmake -S . -B build -DJNI_BACKEND=ON -DPYTHON_BACKEND=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

- JNI 需要 `JAVA_HOME`（env 或 `-DJAVA_HOME=`），否则 CMake 直接 FATAL_ERROR。
- Python C API（`source/python/yolo_c_api.cpp`）不依赖任何 Python 头文件/库，产物与
  Python 版本无关，供 `python/fastyolo` 经 ctypes 调用。

## 4. 产物布局

| 生成器 | 产物位置 |
|---|---|
| Makefiles / Ninja | `<build>/lib/`（全部 `.so`），并 POST_BUILD 复制一份到 `<build>/bin/libyolo_*.so` |

`libFastYolo.so` 与 `libyolo_<后端>.so` 放同一目录即可部署：主库经 `RPATH=$ORIGIN`
从自身所在目录加载后端库，**整个 lib 目录可整体复制/移动而不失效**。

完整部署目录见 [部署布局](../guides/deployment.md)。

## 5. 示例程序（可选）

`examples/vehicle_collector` 演示 RKNN 板端 RTSP 车辆按跟踪摘拍（GStreamer 硬解 + OpenCV）：

```bash
cd examples/vehicle_collector
cmake -S . -B build -DFY_ROOT=/path/to/libFastYolo -DFY_LIB_DIR=/path/to/libFastYolo/build/lib
cmake --build build -j
```

## 6. CMake option 速查

| 选项 | 默认 | 作用 |
|---|---|---|
| `ONNX_GPU` | **ON** | Linux：`lib/linux-gpu` 存在时链 CUDA EP；`OFF` 强制 CPU 静态库 |
| `ONNX_CUDNN_LIB_DIR` | 空 | 手动指定含 `libcudnn.so.9` 的目录（CUDA 12 变体） |
| `MNN_BACKEND` | OFF | MNN 后端 |
| `TENSORRT_BACKEND` | OFF | TensorRT 后端 |
| `OPENVINO_BACKEND` | OFF | OpenVINO 后端（配 `-DOPENVINO_ROOT=`） |
| `RKNN_BACKEND` | OFF | RKNN 后端（配 `-DRKNN_SDK_DIR=` / `-DRKNN_LIBRARY=`） |
| `ONNXCL_BACKEND` | OFF | onnxcl（OpenCL 直读 ONNX）后端 |
| `JNI_BACKEND` | OFF | 编入 Java JNI 实现（需 `JAVA_HOME`） |
| `PYTHON_BACKEND` | OFF | 编入纯 C API（供 ctypes 调用） |

> 后端互斥，按 `MNN > TENSORRT > OPENVINO > RKNN > ONNXCL > ONNX` 取第一个开启的；
> 详见 [后端总览](../backend/README.md#3-互斥规则重要)。

# libFastYolo 文档中心

本目录集中维护 libFastYolo 的详细文档。根目录 [`../README.md`](../README.md) 只保留
项目定位、能力矩阵与快速上手入口，编译/后端细节都在这里，避免 README 过长。

---

## 文档地图

### 后端（推理引擎）—— 每个后端一份编译与使用说明

| 后端 | 文档 | 平台 | 模型 | 典型设备 |
|---|---|---|---|---|
| ONNX Runtime（CPU 静态 / CUDA EP） | [backend/onnx.md](backend/onnx.md) | Linux / Windows | `.onnx` | CPU / CUDA GPU |
| OpenVINO | [backend/openvino.md](backend/openvino.md) | Linux / Windows | `.onnx` | CPU / NPU / GPU |
| TensorRT | [backend/tensorrt.md](backend/tensorrt.md) | Linux | `.engine/.plan` | CUDA GPU |
| MNN | [backend/mnn.md](backend/mnn.md) | Linux / Windows | `.onnx` | CPU / OpenCL / Vulkan |
| onnxcl（OpenCL 直读 ONNX，自包含） | [backend/onnxcl.md](backend/onnxcl.md) | Linux / Windows | `.onnx` | 任意 OpenCL 设备（含核显） |
| RKNN（瑞芯微 NPU） | [backend/rknn.md](backend/rknn.md) | Linux aarch64 板端 | `.rknn` | RK3576 / RK3588 NPU |

- 后端总览、选型与互斥规则：[backend/README.md](backend/README.md)
- onnxcl 引擎内部实现与维护者交接：[backend/onnxcl-engine.md](backend/onnxcl-engine.md)

### 编译与绑定

| 文档 | 内容 |
|---|---|
| [build/linux.md](build/linux.md) | Linux 通用前提、CMake 配置、产物布局、RPATH 说明 |
| [build/windows.md](build/windows.md) | Windows（VS2022 / MinGW）、DLL 依赖与 VC++ 运行时 |
| [build/jni.md](build/jni.md) | JNI / Java 绑定：头文件生成、加载顺序、签名变更须知 |
| [build/python-sdk.md](build/python-sdk.md) | Python SDK 与 wheel 打包、离线部署 |

### 使用指南

| 文档 | 内容 |
|---|---|
| [guides/api.md](guides/api.md) | C++ 公共 API 速览（任务、结果对象、ByteTrack 参数） |
| [guides/segment-lazy-migration.md](guides/segment-lazy-migration.md) | SEGMENT 掩码懒加载 + ROI 迁移指南（含三语言示例） |
| [guides/deployment.md](guides/deployment.md) | 产物清单与部署目录布局 |
| [guides/commands.md](guides/commands.md) | 常用构建命令速查 |

### 运维与参考

| 文档 | 内容 |
|---|---|
| [troubleshooting.md](troubleshooting.md) | 故障排查汇总（GPU / JNI / Python / 构建） |
| [appendix/version-matrix.md](appendix/version-matrix.md) | ORT × CUDA × cuDNN 版本矩阵 |
| [appendix/measured-results.md](appendix/measured-results.md) | 本机实测记录（延迟、FPS、正确性） |
| [history/README.md](history/README.md) | 历史编译笔记归档说明 |

---

## 快速跳转

- 想最快跑起来：`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`（默认 ONNX 后端，见 [backend/onnx.md](backend/onnx.md)）
- 只有核显、不想装 CUDA/ORT：用 [backend/onnxcl.md](backend/onnxcl.md)
- 瑞芯微板端：用 [backend/rknn.md](backend/rknn.md)
- 给 Java 用：[build/jni.md](build/jni.md)
- 给 Python 用：[build/python-sdk.md](build/python-sdk.md)
- 出问题了：[troubleshooting.md](troubleshooting.md)

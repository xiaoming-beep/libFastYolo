# libFastYolo

> 目标检测 / 实例分割 / 姿态 / 旋转框 / 分类 / 语义分割 / 深度估计 + ByteTrack 多目标跟踪的
> **C++20 动态库**，屏蔽**推理后端、硬件设备、模型输出格式**三类差异，一套接口可跑在
> ONNX Runtime（CPU / CUDA）、OpenVINO、TensorRT、MNN、**onnxcl（OpenCL 直读 ONNX）**、
> **RKNN（瑞芯微 NPU）** 之上，并提供 **C++ / JNI(Java) / Python** 三种用法。
>
> 更新日期：2026-10-15。本版新增 **onnxcl**（自包含 OpenCL 推理，无需 CUDA/ORT）与
> **RKNN**（RK3576/RK3588 板端 NPU）两个后端。
>
> 📚 **完整文档见 [`docs/`](docs/README.md)**：每个后端的编译方式、JNI/Python 打包、
> 版本矩阵、故障排查都在那边，本 README 只做总览与入口。

---

## 目录

- [1. 能力总览](#1-能力总览)
- [2. 三层屏蔽设计](#2-三层屏蔽设计)
- [3. 后端矩阵](#3-后端矩阵)
- [4. 快速开始](#4-快速开始)
- [5. 三套 SDK](#5-三套-sdk)
- [6. 目录结构](#6-目录结构)
- [7. 文档导航](#7-文档导航)

---

## 1. 能力总览

- **任务**：`DETECT`、`SEGMENT`、`POSE`、`OBB`、`CLASSIFY`、`SEMSEG`、`DEPTH`；
  `TaskType::AUTO` 按模型输出形状自动推断。
- **跟踪**：内置 ByteTrack 两级关联（高/低置信度检测互补），`track()` 返回稳定 `track_id`，
  按需惰性创建；SEGMENT/POSE/OBB 的载荷（掩码/关键点/角度）随框保留。
- **后端**：ONNX Runtime（CPU 静态 / CUDA EP）、OpenVINO、TensorRT、MNN、onnxcl、RKNN ——
  CMake 选项六选一，核心层不感知后端。
- **绑定**：C++ 公共头（`fast_yolo/yolo.h`）、纯 C API（ctypes / Python SDK）、JNI（Java）。

## 2. 三层屏蔽设计

| 层 | 机制 | 说明 |
|---|---|---|
| 推理后端屏蔽 | C ABI 边界（`yolo_infer_create / init / predict / destroy` 四个 `extern "C"` 函数） | 核心层持 `void* context_`；后端编译为独立 `libyolo_*.so`，运行时经 `RPATH=$ORIGIN` 同目录加载；换后端只改 CMake option |
| 硬件设备屏蔽 | `InferDevice`（`AUTO/CPU/NPU/VULKAN/OPENCL/CUDA/ALL`）+ `YOLOConfig.device/gpu_id/num_threads/precision` | ONNX 后端按 device 注册 CUDA EP（失败自动回退 CPU 并打日志）；MNN/OpenVINO 各自映射设备 |
| 模型差异屏蔽 | 读模型元数据回填 input/output shape；解码器按任务/形状自适应 | v8 头 `[1,C,N]`、端到端 `[1,N,6]`（免 NMS）、SEG 双输出、POSE `4+nc+3k`、OBB、SEMSEG/DEPTH 稠密输出等 |

## 3. 后端矩阵

| 后端 | CMake 开关 | Linux | Windows | 设备 | 模型格式 |
|---|---|---|---|---|---|
| **ONNX Runtime**（CPU / CUDA） | 默认 / `ONNX_GPU` | ✅ | ✅（仅 CPU） | CPU / CUDA | `.onnx` |
| **OpenVINO** | `-DOPENVINO_BACKEND=ON` | ✅ | ✅ | CPU / NPU / GPU / AUTO | `.onnx` |
| **TensorRT** | `-DTENSORRT_BACKEND=ON` | ✅ | 理论 | CUDA | `.engine/.plan` |
| **MNN** | `-DMNN_BACKEND=ON` | ✅ | ✅（需自编） | CPU / OPENCL / VULKAN | `.onnx`（需转换） |
| **onnxcl**（新） | `-DONNXCL_BACKEND=ON` | ✅ | ✅ | 任意 OpenCL 设备（含核显） | `.onnx`（**直读，零转换**） |
| **RKNN**（新） | `-DRKNN_BACKEND=ON` | ✅ aarch64 板端 | ❌ | NPU | `.rknn` |

后端**六选一**，优先级 `MNN > TENSORRT > OPENVINO > RKNN > ONNXCL > 兜底 ONNX`；
`JNI_BACKEND` / `PYTHON_BACKEND` 可与任意后端叠加。

> 选型建议与各后端编译方式：[`docs/backend/README.md`](docs/backend/README.md)

## 4. 快速开始

```bash
# Linux 默认（ONNX 后端；检测到 lib/linux-gpu 时自动含 CUDA EP）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
# 产物：build/lib/libFastYolo.so + build/lib/libyolo_onnx.so（同目录即可部署）
```

```bat
:: Windows（VS2022）
cmake -B build-msvc -G "Visual Studio 17 2022"
cmake --build build-msvc --config Release -j
:: 产物：build-msvc\bin\Release\{FastYolo.dll, yolo_onnx.dll, onnxruntime.dll}
```

按后端查阅详细步骤：

| 想用什么 | 看这里 |
|---|---|
| ONNX Runtime（CPU / CUDA） | [docs/backend/onnx.md](docs/backend/onnx.md) |
| OpenVINO | [docs/backend/openvino.md](docs/backend/openvino.md) |
| TensorRT | [docs/backend/tensorrt.md](docs/backend/tensorrt.md) |
| MNN | [docs/backend/mnn.md](docs/backend/mnn.md) |
| onnxcl（只有核显 / 不想装 CUDA） | [docs/backend/onnxcl.md](docs/backend/onnxcl.md) |
| RKNN（瑞芯微板端） | [docs/backend/rknn.md](docs/backend/rknn.md) |
| 通用前提 / CMake 选项 | [docs/build/linux.md](docs/build/linux.md) · [docs/build/windows.md](docs/build/windows.md) |

## 5. 三套 SDK

| SDK | 入口 | 说明 |
|---|---|---|
| C++ | `source/include/fast_yolo/yolo.h` | `YOLOInference / Letterbox / YOLOConfig / YOLOResult`；见 [API 速览](docs/guides/api.md) |
| Java | `java/fast-yolo-jni/`（Maven） | `-DJNI_BACKEND=ON` 编入主库；见 [JNI 文档](docs/build/jni.md) |
| Python | `python/fastyolo/`（ctypes + wheel） | `-DPYTHON_BACKEND=ON`；见 [Python SDK](docs/build/python-sdk.md) · [`python/README.md`](python/README.md) |

## 6. 目录结构

```
CMakeLists.txt                      # 顶层构建；后端六选一 + JNI/PYTHON 绑定开关
source/
├── include/fast_yolo/
│   ├── yolo.h                      # 公共 API + C ABI 契约
│   ├── nms.h / img_util.h
│   ├── fastyolo_c.h                # Python SDK 的纯 C API
│   └── fast_yolo_jni.h / com_fastyolo_*.h   # JNI 头（javac -h 生成）
├── core/                           # Letterbox、predict/track、后处理解码、NMS
├── python/yolo_c_api.cpp           # C API 实现（-DPYTHON_BACKEND=ON）
├── jni/yolo_jni.cpp                # JNI 实现（-DJNI_BACKEND=ON）
├── backend/{onnx,openvino,TensorRT,MNN,onnxcl,RKNN}/   # 各后端适配层
├── track/ByteTrack/                # ByteTrack（含 Eigen、LAPJV）
└── utils/img_util.cpp
java/fast-yolo-jni/                 # Java 工程
python/fastyolo/                    # Python SDK 包
examples/vehicle_collector/         # RKNN 板端 RTSP 车辆按跟踪摘拍示例
docs/                               # 📚 详细文档（见下）
```

## 7. 文档导航

完整索引：[`docs/README.md`](docs/README.md)

- **后端编译与使用**：[docs/backend/](docs/backend/README.md)（onnx / openvino / tensorrt / mnn / onnxcl / rknn）
- **编译与绑定**：[Linux](docs/build/linux.md) · [Windows](docs/build/windows.md) · [JNI](docs/build/jni.md) · [Python SDK](docs/build/python-sdk.md)
- **使用指南**：[API 速览](docs/guides/api.md) · [SEGMENT 掩码懒加载迁移](docs/guides/segment-lazy-migration.md) · [部署布局](docs/guides/deployment.md) · [命令速查](docs/guides/commands.md)
- **参考**：[故障排查](docs/troubleshooting.md) · [版本矩阵](docs/appendix/version-matrix.md) · [实测记录](docs/appendix/measured-results.md) · [历史归档](docs/history/README.md)

---

## 许可说明

项目本身暂无 LICENSE 声明。捆绑/依赖的第三方组件遵循各自许可：ONNX Runtime（MIT）、
OpenVINO（Apache-2.0）、MNN（Apache-2.0）、TensorRT（NVIDIA EULA）、ByteTrack（Apache-2.0）、
Eigen（MPL-2.0）。发布前请补齐本项目许可证并核对第三方许可义务。

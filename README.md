# libFastYolo

> 目标检测 / 实例分割 / 姿态 / 旋转框 / 分类 / 语义分割 / 深度估计 + ByteTrack 多目标跟踪的
> **C++20 动态库**，屏蔽**推理后端、硬件设备、模型输出格式**三类差异，一套接口可跑在
> ONNX Runtime（CPU / CUDA）、OpenVINO、TensorRT、MNN、**onnxcl（OpenCL 直读 ONNX）**、
> **RKNN（瑞芯微 NPU）** 之上，并提供 **C++ / JNI(Java) / Python** 三种用法。
>
> 更新日期：2026-10-16。本版新增 **onnxcl**（自包含 OpenCL 推理，无需 CUDA/ORT）与
> **RKNN**（RK3576/RK3588 板端 NPU）两个后端。
>
> 📚 **完整文档见 [`docs/`](docs/README.md)**：每个后端的编译方式、JNI/Python 打包、
> 版本矩阵、故障排查都在那边，本 README 只做总览与入口。

---

## 定位说明

**libFastYolo 不是推理引擎，也没有自己实现任何算子、图优化器或硬件 kernel。** 所有前向计算都交给
下面的第三方引擎完成。它本身是一个 C++20 适配层，价值全部落在**抹平调用侧的差异**上。

**它要解决的具体麻烦**：同一个 YOLO 模型，在不同语言、不同平台、不同后端上的加载方式、输入约定、
输出张量布局、后处理 API 都不一样。

- **语言不一致**：C++ 用 `Ort::Session`、Java 要 JNI 再包一层、Python 用 `onnxruntime` 或 `cv2.dnn`
  ——同一套推理逻辑得写三遍，还得各自维护。
- **平台不一致**：x86 服务器用 CUDA，ARM 板端用 NPU，只有核显的机器只能吃 OpenCL——三套 SDK、三套编译链。
- **后端不一致**：TensorRT 要先把 `.onnx` 转成 `.engine`，MNN 要转 `.mnn`，RKNN 要转 `.rknn`；
  解码器还得分别适配 v8 头 `[1,C,N]`、端到端 `[1,N,6]`、SEG 双输出等不同输出布局。

**结果就是：换一次引擎，业务代码里散落的加载、预处理、解码、设备选择全都要跟着改。**

本项目把这件事压缩成**只改一个 CMake 开关**。业务侧永远只调用 `YOLOInference::predict()` 与
`track()` 这一套接口，后端差异被 C ABI 边界挡在库内部（见 [三层屏蔽设计](#2-三层屏蔽设计)）。

一句话：**它省的是"换引擎时改代码"的成本，不是"换引擎"本身的成本。**

### 关于第三方依赖

除 `onnxcl` 外，所有后端都是对第三方引擎的封装。ONNX Runtime、OpenVINO、TensorRT、MNN、RKNN 的
运行时与 SDK **均不随本项目分发**，需自行获取并遵守各自的许可（见文末[许可说明](#许可说明)）。

**唯一不依赖任何第三方推理引擎的是 `onnxcl` 后端**——它自包含 OpenCL 实现，直接读 `.onnx`，
不需要 ONNX Runtime、不需要 CUDA、也不需要单独安装 OpenCL SDK。

> ⚠️ **但请注意：onnxcl 的成熟度和其他后端不在一个量级。**
> 它由作者本人维护，目前主要用于开发期自测——覆盖面有限、真实场景验证不足，
> **作者也不确定还有哪些坑。**
>
> 生产环境建议优先选择有成熟社区支撑的 ONNX Runtime / OpenVINO / TensorRT / MNN / RKNN；
> onnxcl 更适合开发调试，或"只有核显、不想装 CUDA 与 ORT"的轻量场景。

---

## 目录

- [定位说明](#定位说明)
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

> 除 `onnxcl` 外的后端都是对第三方引擎的封装；`onnxcl` 是自包含 OpenCL 实现、不依赖任何第三方运行时，
> 但由作者自行维护、成熟度低于其他后端，详见 [定位说明](#定位说明)。

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

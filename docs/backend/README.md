# 推理后端总览

libFastYolo 的核心层不感知后端：核心与后端之间只通过 4 个 `extern "C"` 函数解耦
（`yolo_infer_create / init / predict / destroy`），后端编译成独立共享库
（`libyolo_onnx.so` / `yolo_onnxcl.dll` / …），运行时由主库经 **RPATH=`$ORIGIN`** 从同目录加载。
换后端只改一个 CMake option，核心、ByteTrack、三套 SDK 全部复用。

---

## 1. 后端矩阵

| 后端 | CMake 开关 | Linux | Windows | 支持设备（`YOLOConfig::device`） | 模型格式 | 第三方依赖 |
|---|---|---|---|---|---|---|
| **ONNX Runtime** | 默认（无需开关） | ✅ | ✅ | CPU；Linux GPU 版另含 CUDA（`CUDA`/`AUTO`） | `.onnx` | 仓库自带 ORT |
| **ONNX-CUDA** | `ONNX_GPU`（默认 ON） | ✅ | ❌ | CUDA / AUTO，失败自动回退 CPU | `.onnx` | ORT GPU + cuDNN 9 |
| **OpenVINO** | `-DOPENVINO_BACKEND=ON` | ✅ | ✅ | CPU / NPU / OPENCL(→GPU) / AUTO | `.onnx` | OpenVINO 运行时 |
| **TensorRT** | `-DTENSORRT_BACKEND=ON` | ✅ | 理论（未验证） | 仅 CUDA | 序列化 `.engine/.plan` | TensorRT SDK |
| **MNN** | `-DMNN_BACKEND=ON` | ✅ | ✅（需自编 MNN） | CPU / OPENCL / VULKAN / AUTO | `.onnx`（需 mnnconvert 转换） | MNN |
| **onnxcl** | `-DONNXCL_BACKEND=ON` | ✅ | ✅（MinGW/MSVC） | OpenCL（AUTO/CPU/ALL 均走 OpenCL） | `.onnx`（**直读，零转换**） | 仅系统 OpenCL 运行时 |
| **RKNN** | `-DRKNN_BACKEND=ON` | ✅（aarch64 板端） | ❌ | NPU | `.rknn` | librknnrt.so |

> `InferDevice` 枚举：`AUTO=0, CPU=1, NPU=2, VULKAN=3, OPENCL=4, CUDA=6, ALL=99`（5 空缺）。

---

## 2. 选型建议

| 场景 | 推荐后端 |
|---|---|
| 通用 Linux / Windows，CPU 或 N 卡，最省事 | **ONNX Runtime**（默认；Linux 有 `lib/linux-gpu` 时自动含 CUDA EP） |
| 只有核显、不想装 CUDA / 不想引第三方推理库 | **onnxcl**（模型丢进来就能在 OpenCL 上跑） |
| Intel 平台，想用 NPU / 核显 | **OpenVINO** |
| 极致 N 卡吞吐、能接受 engine 转换与版本绑定 | **TensorRT** |
| 已有 MNN 生态 / 需要 Vulkan | **MNN** |
| 瑞芯微 RK3576 / RK3588 板端 | **RKNN** |

---

## 3. 互斥规则（重要）

顶层 CMake 用 `if / elseif / else` 串联，**后端六选一，按下面优先级取第一个开启的**：

```
MNN > TENSORRT > OPENVINO > RKNN > ONNXCL > 兜底 ONNX
```

- `ONNX_BACKEND` 只是名义开关；三个别的后端全关时**自动**编译 ONNX，传不传它都可以。
- `JNI_BACKEND` / `PYTHON_BACKEND` 可与任意后端叠加（它们是绑定层，不是后端）。
- 同时开多个后端只会生效优先级最高的那个，不会报错——配置时留意 CMake 日志。

---

## 4. 后端目录一览

```
source/backend/
├── onnx/       # ONNX Runtime 适配（lib/linux、lib/linux-gpu、lib/dll 三套预编译库）
├── openvino/
├── TensorRT/
├── MNN/
├── onnxcl/     # 自包含 OpenCL 引擎（engine/ 约 5000 行，反哺可读性）
└── RKNN/
```

各后端的编译方式见本目录下对应文档：

- [onnx.md](onnx.md)
- [openvino.md](openvino.md)
- [tensorrt.md](tensorrt.md)
- [mnn.md](mnn.md)
- [onnxcl.md](onnxcl.md)
- [rknn.md](rknn.md)

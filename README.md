# libFastYolo

> 目标检测 / 实例分割 / 姿态 / 旋转框 / 分类 / 语义分割 / 深度 + ByteTrack 多目标跟踪的
> C++20 动态库，**屏蔽推理后端、硬件设备、模型输出格式三类差异**，一套接口跑在
> ONNX Runtime（CPU / CUDA）、OpenVINO、TensorRT、MNN 之上，并提供 C++ / JNI(Java) / Python SDK 三种用法。
>
> 更新日期：2026-09-10。本版完成 **SEGMENT 掩码懒加载 + ROI** 升级（详见第 2 章与
> [`docs/SEGMENT-lazy-migration.md`](docs/SEGMENT-lazy-migration.md)）；
> 其余构建说明整合自仓库历史编译笔记（原文件已归档至 `docs/history/`，见附录 B）。

---

## 目录

1. [项目介绍](#1-项目介绍)
2. [本次升级：SEGMENT 掩码懒加载 + ROI（2026-09）](#2-本次升级segment-掩码懒加载--roi2026-09)
3. [构建总览：体系与后端矩阵](#3-构建总览体系与后端矩阵)
4. [前置条件](#4-前置条件)
5. [Linux 编译](#5-linux-编译)
6. [Windows 编译](#6-windows-编译)
7. [JNI（Java 绑定）](#7-jnijava-绑定)
8. [Python SDK 与 wheel 分发](#8-python-sdk-与-wheel-分发)
9. [产物清单与部署目录布局](#9-产物清单与部署目录布局)
10. [版本依赖矩阵（ORT × CUDA × cuDNN，务必先看）](#10-版本依赖矩阵)
11. [常用构建命令速查](#11-常用构建命令速查)
12. [故障排查汇总](#12-故障排查汇总)
- [附录 A：本机实测记录](#附录-a本机实测记录)
- [附录 B：文档归档说明](#附录-b文档归档说明)

---

## 1. 项目介绍

### 1.1 定位与能力

- **任务**：`DETECT`（目标检测）、`SEGMENT`（实例分割）、`POSE`（姿态关键点）、
  `OBB`（旋转框）、`CLASSIFY`（分类）、`SEMSEG`（语义分割）、`DEPTH`（深度估计）；
  `TaskType::AUTO` 可按模型输出形状自动推断。
- **跟踪**：内置 ByteTrack 两级关联（高/低置信度检测互补），`track()` 返回稳定 `track_id`，按需惰性创建。
- **后端**：ONNX Runtime（CPU 静态 / CUDA EP）、OpenVINO、TensorRT、MNN —— CMake 选项四选一，
  核心层不感知后端。
- **绑定**：C++ 公共头（`fast_yolo/yolo.h`）、纯 C API（ctypes/Python SDK）、JNI（Java）。

### 1.2 三层屏蔽设计

| 层 | 机制 | 说明 |
|---|---|---|
| 推理后端屏蔽 | C ABI 边界（`yolo_infer_create / init / predict / destroy` 四个 `extern "C"` 函数） | 核心层持 `void* context_`；后端编译为独立 `libyolo_*.so`，运行时经 RPATH=`$ORIGIN` 同目录加载；换后端只改 CMake option |
| 硬件设备屏蔽 | `InferDevice`（`AUTO/CPU/NPU/VULKAN/OPENCL/CUDA/ALL`）+ `YOLOConfig.device/gpu_id/num_threads/precision` | ONNX 后端按 device 注册 CUDA EP（失败自动回退 CPU 并打日志）；MNN 经 `MapInferDevice()` 映射；OpenVINO 映射 CPU/NPU/GPU |
| 模型差异屏蔽 | 读模型元数据回填 input/output shape；解码器按任务/形状自适应 | v8 头 `[1,C,N]`、端到端 `[1,N,6]`（免 NMS）、SEG 双输出、POSE `4+nc+3k`、OBB、SEMSEG/DEPTH 稠密输出等 |

### 1.3 目录结构（核心部分）

```
CMakeLists.txt                      # 顶层构建；后端 option 四选一 + JNI/PYTHON 绑定开关
source/
├── include/fast_yolo/
│   ├── yolo.h                      # 公共 API：YOLOInference / Letterbox / YOLOConfig / YOLOResult / MaskBudget / C ABI
│   ├── nms.h                       # NMS 数据结构（Detection/DetectionBox）
│   ├── img_util.h                  # 图像工具（BGR→float planar、letterbox 填充）
│   ├── fastyolo_c.h                # Python SDK 的纯 C API
│   ├── fast_yolo_jni.h             # JNI 头文件聚合（由 javac -h 生成，见 §7.2）
│   └── com_fastyolo_*.h            # javac -h 生成的 per-class 头（签名变更必须重新生成）
├── core/
│   ├── yolo.cpp                    # Letterbox、predict/track（含 SEGMENT 两阶段懒加载）
│   ├── postprocess.{h,cpp}         # 各任务解码器；SEGMENT = decode_segment_boxes + build_segment_masks
│   └── nms.cpp                     # NMS（v8 头 + 旋转框 + 分段流式实现）
├── python/yolo_c_api.cpp           # C API 实现（-DPYTHON_BACKEND=ON）
├── jni/yolo_jni.cpp, jni_utils.h   # JNI 实现（-DJNI_BACKEND=ON）
├── backend/{onnx,openvino,TensorRT,MNN}/   # 各后端适配层（lib + 预编译库）
├── track/ByteTrack/                # ByteTrack 实现（含 Eigen、LAPJV）
└── utils/img_util.cpp              # 图像预处理实现

java/fast-yolo-jni/                 # Java 工程（Maven；com.fastyolo 包）
python/fastyolo/                    # Python SDK 包（ctypes 绑定 + examples/tests/scripts）
docs/SEGMENT-lazy-migration.md      # 本次升级迁移指南
docs/history/                       # 历史编译笔记归档（见附录 B）
```

### 1.4 端到端数据流（含跟踪）

```
BGR uint8
  → Letterbox.update(w,h)                   # 计算 scale/padding（分类任务走中心裁剪）
  → Letterbox.fill_bgr(data)                # letterbox + BGR→float planar (RRR…GGG…BBB…)
  → yolo_infer_predict(context_, …)         # C ABI → 后端推理
  → postprocess 解码（按任务；SEGMENT 默认只出框，掩码按需）
  → Letterbox.complete_result(det)          # 坐标反算回原图
  → vector<YOLOResult>（track_id = -1）
  → [track() 时] BYTETracker.update()       # 两级关联 → 回填 track_id（SEGMENT/POSE/OBB 保留框与载荷对应）
```

### 1.5 ByteTrack 关键参数（`YOLOConfig`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `track_frame_rate` | 30 | 视频帧率 |
| `track_buffer` | 30 | 轨迹保持帧数（丢失后 survive 时长） |
| `track_thresh` | 0.5 | 高/低分检测分界 |
| `high_thresh` | 0.6 | 新轨迹初始化阈值 |
| `match_thresh` | 0.8 | IoU 匹配阈值 |

### 1.6 结果对象

`YOLOResult`：`x/y/width/height`（原图坐标，已截断）、`clsId`、`score`、`track_id`，
以及任务载荷：`keypoints`（POSE）、`angle`（OBB）、`mask`（SEGMENT，**ROI**：
长度 `width*height`、0/1 二值、行跨距 `width`、原点 = 框左上角；默认按需生成）。

---

## 2. 本次升级：SEGMENT 掩码懒加载 + ROI（2026-09）

### 2.1 背景

4K 输入是 OCR 下游的硬需求。旧实现每帧为每个保留框生成**整幅原图分辨率掩码**
（4K 单框 8.8MB 清零+写回），叠加下游"整幅掩码找轮廓/连通域"，实测 4K 4 目标
后处理 ≈30ms/帧（50fps 下 ≈1.5 个核心）——而掩码的真实用途只是为每条轨迹挑一张
"面积最大"的抠图，平台期/远离期帧的掩码根本不影响结果。

### 2.2 新设计（要点）

1. **两阶段**：每帧只跑 `decode_segment_boxes`（解码+NMS，≈0.5ms"懒帧"）；
   `build_segment_masks` 只对"确实要消费"的框生成 ROI 掩码（"掩码帧"）。
2. **MaskBudget 预算表**：调用方每帧传 `{trackId → 已保留 best 抠图面积}`，库内 ByteTrack
   之后精确判定：**无记录（新船）或 `框面积 > bestArea`** 才生成；框面积是抠图面积的严格上界，
   跳过无损、无帧滞后。被触发框的重叠跟踪框一并生成（保住重叠仲裁语义）。
3. **ROI 掩码**：长度 = 框 `width*height`，行跨距 = 框宽，原点 = 框左上角；旧整幅掩码框外恒 0，
   故无信息损失（已逐位验证）。
4. **API 不兼容变更**：`predict(…, need_masks)` 默认只出框；`track(…, budgets, n)` 预算驱动；
   `mask_w/mask_h` 等整幅假设全部移除。

### 2.3 实测（纯后处理、与推理设备无关；本机 Hygon 8C16T / ORT 1.23.2）

| 场景 | 懒帧 | 掩码帧（ROI 全量） | +findContours(ROI) | 模拟轨迹（每 40 帧评估 3 帧） |
|---|---|---|---|---|
| face 1202×823，1 目标 | 0.54ms | 2.46ms | +0.20ms | 0.68ms（0.034 核@50fps） |
| 4K，1 大目标 | 0.52ms | 4.37ms | +0.35ms | 0.81ms（0.041 核@50fps） |
| 4K，4 目标 | 0.52ms | 4.43ms | +0.38ms | 0.81ms（0.041 核@50fps） |

对照旧实现（4K 4 目标）：掩码 15.0ms + 整幅轮廓 14.9ms ≈ **30ms/帧 → 1.5 核@50fps**。
ROI 无损校验：face 场景非零像素 183379 与旧整幅完全一致；Python/Java/JNI 三层冒烟全过
（详见迁移指南 §8）。

### 2.4 迁移与下游集成

- **完整变更清单、三语言示例、注意事项** → [`docs/SEGMENT-lazy-migration.md`](docs/SEGMENT-lazy-migration.md)；
- **ctfo-ship-pixel / ctfo-ship-common 合入补丁**（SegCutout/TypeCropCollector/common Java 类
  + 调用序说明）→ `libFastYolo/.working/pixel-lazy-patch/`；
- 升级后必须重生成 JNI 头文件（§7.2）与带 `-DPYTHON_BACKEND=ON` 构建 Python SDK。

---

## 3. 构建总览：体系与后端矩阵

### 3.1 架构

`libFastYolo`（`.so/.dll`）= 核心层（`source/core`：YOLOInference、Letterbox、NMS）+ ByteTrack 跟踪 + **一个推理后端**。
核心与后端之间通过 4 个 `extern "C"` 函数解耦（`yolo_infer_create / init / predict / destroy`），
后端编译为独立共享库 `libyolo_onnx.so / yolo_openvino.dll / ...`，运行时由 `libFastYolo` 经 **RPATH=$ORIGIN** 从同目录加载。
模型输入统一为 ONNX 字节流（TensorRT 后端例外，需序列化 engine，见 5.4）。

### 3.2 CMake 选项一览

| 选项 | 默认 | 作用 |
|---|---|---|
| `MNN_BACKEND` | OFF | 使用 MNN 后端 |
| `TENSORRT_BACKEND` | OFF | 使用 TensorRT 后端 |
| `OPENVINO_BACKEND` | OFF | 使用 OpenVINO 后端（需 `-DOPENVINO_ROOT=...`） |
| `ONNX_BACKEND` | OFF | 名义开关。**实际逻辑是四选一 if-elseif-else：`MNN > TENSORRT > OPENVINO > 兜底 ONNX`**，三个其他后端全关时自动编译 ONNX，因此 ONNX 构建传不传它都可以 |
| `ONNX_GPU` | **ON** | （2026-09 新增，仅 Linux）`lib/linux-gpu` 存在 ONNX Runtime GPU 共享库时自动链接 CUDA EP；`-DONNX_GPU=OFF` 强制回退 CPU 静态库 |
| `JNI_BACKEND` | OFF | 编入 Java JNI 实现（需 `JAVA_HOME`） |
| `PYTHON_BACKEND` | OFF | 编入纯 C API（`source/python/yolo_c_api.cpp`，供 ctypes 调用，不依赖 Python 头文件） |

组合规则：后端选项互斥（选一），`JNI_BACKEND` / `PYTHON_BACKEND` 可与任意后端叠加。

### 3.3 后端 × 平台 × 设备支持矩阵

| 后端 | Linux | Windows | 实际支持的设备（`YOLOConfig::device`） | 模型格式 |
|---|---|---|---|---|
| **ONNX（CPU 静态）** | ✅ | ✅ | CPU（`num_threads`） | `.onnx` |
| **ONNX-CUDA**（2026-09 新增） | ✅ | ❌（后续可加 DirectML） | `CUDA`/`AUTO` 走 GPU（`gpu_id` 选卡），注册失败自动回退 CPU | `.onnx` |
| OpenVINO | ✅ | ✅ | CPU / NPU / GPU（`OPENCL`→GPU）/ AUTO | `.onnx` |
| TensorRT | ✅ | 理论（未在仓库验证） | 仅 CUDA | **序列化 engine**（`.engine/.plan`，用 trtexec 从 onnx 转换） |
| MNN | ✅（自带 lib 为 CPU 构建） | ✅（需自编 MNN） | CPU / OPENCL / VULKAN / AUTO（经 `MapInferDevice`） | `.onnx`（MNN 自带转换器） |

> `InferDevice` 枚举：`AUTO=0, CPU=1, NPU=2, VULKAN=3, OPENCL=4, CUDA=6, ALL=99`（5 空缺）。

### 3.4 产物输出目录

| 生成器 | 产物位置 |
|---|---|
| Makefiles/Ninja（Linux） | `<build>/lib/`（全部 `.so`）+ POST_BUILD 复制一份 `<build>/bin/libyolo_*.so` |
| Visual Studio（Windows，多配置） | `<build>/bin/Release/`（`FastYolo.dll`、`yolo_*.dll`、运行时 dll）；Debug 版主库带后缀 `d`（`FastYolod.dll`） |

---

## 4. 前置条件

### 通用

| 项 | 要求 |
|---|---|
| CMake | ≥ 3.21 |
| C++ 编译器 | 支持 C++20：Linux **GCC ≥ 11**；Windows **VS2022（MSVC 14.3x+）**（含"使用 C++ 的桌面开发"工作负载，BuildTools 亦可） |
| ONNX Runtime 库 | 仓库按平台放置：Linux CPU 静态 `.a`（`lib/linux/`）、Linux GPU 共享 `.so`（`lib/linux-gpu/`）、Windows `.dll/.lib`（`lib/dll/`）。**工作副本里若目录为空**，从官方 tgz 解包放入即可（CPU：`onnxruntime-linux-x64-*.tgz`；GPU：`onnxruntime-linux-x64-gpu-*.tgz`），版本配套见 §10.1 |
| SVN 工作副本完整性 | `source/core/postprocess.cpp` 等若显示缺失，多为 "obstructed" 文件挡住了 `svn update`，见[第 12 节](#12-故障排查汇总) |

### 按功能附加

| 功能 | 附加要求 |
|---|---|
| JNI | JDK ≥ 17（`jni.h` + `libjvm/jvm.lib`）；环境变量 `JAVA_HOME` 或 `-DJAVA_HOME=`。重新生成头文件需与 Java 工程同 JDK + Lombok（见 7.2） |
| OpenVINO 后端 | OpenVINO 安装（推荐 `pip install openvino`，CMake 用 `-DOPENVINO_ROOT=<site-packages/openvino 路径>` 指过去） |
| TensorRT 后端 | CUDA toolkit + TensorRT SDK（头文件与 `libnvinfer/libnvinfer_plugin`，CMake 自动搜 `/usr/{src/tensorrt,include}/...`） |
| ONNX-CUDA（构建机） | 无额外要求（链接的是仓库自带共享库；构建机可以没有 GPU） |
| ONNX-CUDA（运行机） | NVIDIA 驱动（支持 CUDA 12）+ CUDA 12 运行库（libcublas/cublasLt/cudart，装 CUDA 12 即有）+ **libcudnn.so.9**（见第 10 节版本矩阵；可随包分发） |
| Python SDK | Python 3.8+；打包机装 `pip install cmake`（Windows 还需 `pip install openvino`）；使用机仅需 Python + 网络 |

---

## 5. Linux 编译

### 5.1 ONNX 后端 — CPU（静态链接，最简）

```bash
cd /path/to/libFastYolo
cmake -S . -B build -DONNX_GPU=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

- 链接 `source/backend/onnx/lib/linux/libonnxruntime.a`（仓库 tgz 版 **1.24.4**；本机实测 1.23.2 / 1.29.0 亦可），产物运行时**不依赖任何 onnxruntime 文件**。
- 产物：`build/lib/libFastYolo.so` + `build/lib/libyolo_onnx.so`。两个 `.so` 放同一目录即可部署（`libFastYolo.so` 经 `$ORIGIN` 找 `libyolo_onnx.so`）。
- CMake 配置日志确认：`Using static ONNX Runtime: .../libonnxruntime.a`。

### 5.2 ONNX 后端 — CUDA GPU（2026-09 新增，默认开启）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # 检测到 lib/linux-gpu 即自动走 GPU
cmake --build build -j$(nproc)
```

- CMake 日志应出现：`Using ONNX Runtime GPU (CUDA EP): .../source/backend/onnx/lib/linux-gpu`。
- 构建的库同时具备 CPU 和 CUDA 能力（同一二进制）：运行时按 `YOLOConfig::device` 决定——
  `CUDA`/`AUTO` 先注册 CUDA EP（`gpu_id` 选卡），**任何失败（无驱动/无卡/cuDNN 缺失或架构不支持）打印日志后自动回退 CPU**；`CPU` 直接走 CPU。
- 构建机不要求有 GPU（链接的是 `lib/linux-gpu/` 中的 GPU 共享库；本机当前 1.23.2，1.24.4 / 1.29.0 同样验证通过）；没有 patchelf 时给个 WARNING 继续。
- 产物：`build/lib/` 下除主库外还有 `libonnxruntime.so.1(.24.4)`、`libonnxruntime_providers_cuda.so`（约 316MB，含 sm_50–sm_90 全架构 fatbin）、`libonnxruntime_providers_shared.so`。
- **自包含部署关键步骤**（已在构建里固化）：POST_BUILD 用 patchelf 给产物目录中的
  `libonnxruntime_providers_cuda.so` 打 `RUNPATH=$ORIGIN`，使同目录放置的 cuDNN `.so` 可被解析——
  即整包拷贝 + cuDNN 放同一目录，即可**零环境变量**运行（实测见附录）。
  仓库源文件 `lib/linux-gpu/` 保持未打补丁的原版。

运行与验证：

```bash
# 方式 A：目标机系统里有 libcudnn.so.9（apt 安装或 LD_LIBRARY_PATH 提供）
./demo --device cuda

# 方式 B：完全自包含 —— cuDNN 的 .so 与 providers_cuda.so 放同一目录，不设任何环境变量
ldd build/lib/libonnxruntime_providers_cuda.so | grep cudnn   # 应指向同目录
```

日志出现 `[yolo_onnx] CUDA EP enabled, gpu_id=0` 即成功；回退时打印
`[yolo_onnx] CUDA EP unavailable, fallback to CPU: ...`。

### 5.3 OpenVINO 后端

```bash
pip install openvino   # 或指向独立安装的 OpenVINO
OV=$(python3 -c "import openvino,os;print(os.path.dirname(openvino.__file__))")
cmake -S . -B build -DOPENVINO_BACKEND=ON -DOPENVINO_ROOT="$OV" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

- OpenVINO 运行库由 POST_BUILD 拷到 `build/lib/`（RPATH=$ORIGIN，运行无需 LD_LIBRARY_PATH）。
- 设备映射：`CPU→CPU`、`NPU→NPU`、`OPENCL→GPU`、`AUTO→AUTO`，其余（含 CUDA）回退 CPU 并打日志。
- 注意 `INFERENCE_NUM_THREADS` 仅 CPU 插件支持（后端已按设备条件设置）。

### 5.4 TensorRT 后端

```bash
cmake -S . -B build -DTENSORRT_BACKEND=ON -DJNI_BACKEND=ON \
      -DJAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

- 依赖：本机 CUDA toolkit（`find` 自动搜 `/usr/local/cuda*`）+ TensorRT SDK（`/usr/src/tensorrt` 或系统路径的 `nvinfer/nvinfer_plugin`）。
- **模型必须是序列化的 engine**（后端直接 `deserializeCudaEngine`）：先用 trtexec 把 onnx 转出 `.engine`：
  `trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n.engine`。engine 绑定 TRT 版本与具体显卡，换卡/升级 TRT 需重新生成。
- 若嫌 TensorRT 安装重：Linux GPU 场景优先考虑 3.2 的 ONNX-CUDA（同为 N 卡加速，免装 TRT，模型免转换）。

### 5.5 MNN 后端

- 仓库自带 `source/backend/MNN/lib/libMNN.so` 为 **CPU 构建**（无 OpenCL/CUDA 符号）；`device=OPENCL/CUDA` 需按第 6.4 节的思路在 Linux 上自行编译 MNN 并替换该 `.so`（如 `-DMNN_OPENCL=ON`）。
- 构建：`cmake -S . -B build -DMNN_BACKEND=ON && cmake --build build`。
- 非 CPU 设备会自动启用磁盘 cache（`./cachefile`），多实例并发注意工作目录。

---

## 6. Windows 编译

### 6.1 环境要求

- VS2022（或 BuildTools）+ "使用 C++ 的桌面开发"工作负载。
- **VC++ 2022 Redistributable** 必须安装——缺失时 `onnxruntime.dll` 拒绝加载。
- JDK（JNI 时）：`set JAVA_HOME=D:\...\jdk21`。⚠️ JDK 自带 VC++ 运行时可能过旧导致 onnxruntime 初始化失败，处理见[第 12 节](#12-故障排查汇总)。

```bat
:: 多配置 VS 生成器：CMAKE_BUILD_TYPE 无效，靠 --config
cmake -B build-msvc -G "Visual Studio 17 2022" -DJNI_BACKEND=ON
cmake --build build-msvc --config Release -j
```

产物在 `build-msvc\bin\Release\`（Debug 为 `bin\Debug\`，主库名带 `d` 后缀）。
CLion 配置：Settings → CMake 添加配置，Generator 选 `Visual Studio 17 2022`，
Options 填 `-DJNI_BACKEND=ON` 等即可。

### 6.2 ONNX 后端（CPU）

Windows 分支固定链接仓库自带 `lib/dll/onnxruntime.lib`，POST_BUILD 拷 `onnxruntime.dll` 到产物目录。
产物三件套必须同目录：`FastYolo.dll` + `yolo_onnx.dll` + `onnxruntime.dll`。
（Windows GPU 路线暂未实现；同一后端将来可加 DirectML EP，覆盖 N 卡/A/I 卡且不需 CUDA。）

### 6.3 OpenVINO 后端（Python SDK 默认路线）

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

### 6.4 MNN 后端（完整自编译流程，含 JNI）

来源：Windows MNN-JNI 编译记录（MNN 3.6.1 + VS2022 BuildTools）。

1. **编译 MNN（启用 OpenCL）**：

   ```bash
   cmake -G "Visual Studio 17 2022" -A x64 -B build \
     -DMNN_BUILD_DEMO=OFF -DMNN_BUILD_BENCHMARK=OFF -DMNN_BUILD_CONVERTER=OFF \
     -DMNN_BUILD_TRAIN=OFF -DMNN_SEP_BUILD=OFF -DMNN_OPENCL=ON -DMNN_USE_SYSTEM_LIB=OFF
   cmake --build build --config Release -j
   ```

   产物 `build/Release/MNN.dll + MNN.lib`（`MNN_SEP_BUILD=OFF` 所以是单库）。
   **不需要 `OpenCL.lib`**：`MNN_USE_SYSTEM_LIB=OFF` 使 MNN 运行时 `LoadLibrary("OpenCL.dll")`，编译期零依赖。

2. **拷贝进仓库**：`cp MNN.lib MNN.dll → source/backend/MNN/lib/`（Linux 用的 `libMNN.so` 保留无妨）。

3. **生成 JNI 头文件**（见 7.2，⚠️ 生成头文件必须用 **JDK ≥ 17**，Java 工程是 17+Lombok）。

4. **构建 libFastYolo**（此处 C++ 库用 JDK8 的 `jvm.lib` 链接——历史环境偏好，与生成头的 JDK 版本无关）：

   ```bash
   export JAVA_HOME=/d/works/software/jdk/jdk8
   cmake -G "Visual Studio 17 2022" -A x64 -B build-mnn-jni -DMNN_BACKEND=ON -DJNI_BACKEND=ON
   cmake --build build-mnn-jni --config Release -j
   ```

5. **收尾**：`MNN.dll` 被 POST_BUILD 拷到 `build-mnn-jni/bin/`（注意**不在** `bin/Release/`），
   手动 `cp bin/MNN.dll bin/Release/`，保证三 DLL 同目录：`FastYolo.dll → yolo_mnn.dll → MNN.dll`。
   验证导出：`dumpbin /exports FastYolo.dll | find /c "Java_com_fastyolo"` 应 ≥ 15；
   Git Bash 下 dumpbin 的 `/exports` 会被 MSYS 当路径，加 `MSYS_NO_PATHCONV=1`。

---

## 7. JNI（Java 绑定）

### 7.1 结构

- Java 工程：`java/fast-yolo-jni/`（Maven，`mvn clean package` → `target/fast-yolo-jni-1.0.0.jar`）。
  类：`InferDevice / YoloException / YoloConfig / YoloResult / Letterbox / YoloInference`（后两者持 native handle，AutoCloseable）。
- JNI 实现编进主库：`-DJNI_BACKEND=ON` 时 `source/jni/yolo_jni.cpp` 直接打进 `libFastYolo.so / FastYolo.dll`。
  CMake 用 `JAVA_HOME`（env 或 `-DJAVA_HOME=` 传入）找 `include/` 与 `libjvm/jvm.lib`，找不到直接 FATAL_ERROR。
- 调用流程与线程模型：Letterbox 可跨帧复用；`predict/track` 结果坐标已映射回原图；实例非线程安全，多线程各自建实例。
- native 方法：`nativePredict / nativePredictMask / nativeTrack / nativeTrackMask`（2026-09 新增后两者，
  SEGMENT 掩码懒加载：`nativePredictMask(…, needMask)` 全量 ROI 掩码、`nativeTrackMask(…, budgetIds, budgetAreas)`
  预算表驱动）；参数签名字符串见 `source/jni/jni_utils.h`，**任何签名变更后必须按 §7.2 重新生成头文件**，
  否则新符号会被 C++ 改名（`_Z…`）导致 Java 端 `UnsatisfiedLinkError`。

### 7.2 重新生成 JNI 头文件（Java native 方法签名变更时必须执行）

```bash
# 生成头文件必须用 JDK ≥ 17（Java 工程 target=17 + Lombok，需 processorpath 挂 lombok jar）
JDK21_BIN=<jdk17+/bin>
LOMBOK=lombok-<与pom一致>.jar   # 正式做法：mvn dependency:copy -Dartifact=org.projectlombok:lombok:<ver>
SRC=java/fast-yolo-jni/src/main/java
"$JDK21_BIN/javac" -h source/include/fast_yolo -cp "$LOMBOK" -processorpath "$LOMBOK" \
  -d /tmp/jclasses "$SRC"/com/fastyolo/*.java
# 生成 com_fastyolo_Letterbox.h / com_fastyolo_YoloInference.h，与 fast_yolo_jni.h 一同入库
```

⚠️ 头文件必须与**提交版 Java 源码**一致；签名变了而没重新生成，运行时会 `GetMethodID`/`UnsatisfiedLinkError`。
另外 5 个数据类（TaskType/Keypoint/ClassifyResult/SemSegResult/DepthResult）的字段构造器签名必须匹配 `source/jni/jni_utils.h` 中的 JNI 签名字符串。

### 7.3 Java 侧加载顺序

DLL/so 依赖链：`FastYolo → yolo_* → 后端运行时`。

```java
// Windows / ONNX CPU：按序显式加载
System.load(dir + "/onnxruntime.dll");
System.load(dir + "/yolo_onnx.dll");
System.load(dir + "/FastYolo.dll");

// Linux ONNX CPU 静态版：只需加载主库（onnxruntime 已静态链入，libyolo_onnx.so 经 $ORIGIN 自动解析）
System.load("/path/libFastYolo.so");

// Linux ONNX-CUDA 自包含版：同样只需加载主库，但整个目录必须含
// libonnxruntime*.so、providers_cuda/shared、（可选）libcudnn*.so.9 或系统提供 cuDNN；
// jvm.dll 在 Java 进程内已存在，无需加载。
```

---

## 8. Python SDK 与 wheel 分发

### 8.1 流程总览

```
编译机（cmake+编译器，一次性）                     使用机（只要 Python，无需编译器）
  build_windows.bat / build_linux.sh                  pip install fastyolo-0.1.0-*.whl
  → collect_artifacts.py（产物进 python/fastyolo/libs/）      │
  → build_wheel.py → dist/*.whl  ──── 微信/网盘/内部 PyPI ────┘
```

### 8.2 各平台构建

**Windows（OpenVINO 后端）**：

```bat
python -m pip install cmake openvino
python\scripts\build_windows.bat
cd python && python scripts\build_wheel.py        :: 精简 wheel ~0.2MB
```

**Linux（ONNX 后端）**：

```bash
bash python/scripts/build_linux.sh          # CPU 版（-DONNX_GPU=OFF，静态，适配 slim wheel）
bash python/scripts/build_linux.sh --gpu    # CUDA 版（-DONNX_GPU=ON + 收集 GPU 运行时 ~340MB）
cd python && python3 scripts/build_wheel.py # 仅对 CPU 版有意义（见 8.4）
```

wheel 说明：`py3-none` 标签**不绑定 Python 版本**（3.8+ 通用，换版本重装一次即可）；
`--full` 为全捆绑模式（OpenVINO 运行时打进包，目标机可完全离线；实现上依赖本机装有 openvino pip 包）。
注意 `pyproject.toml` 无条件声明了 `openvino>=2024.0` 依赖——Linux 精简 wheel 安装时也会拉 openvino，属已知待优化点。

### 8.3 使用机安装与验证

```bash
pip install fastyolo-0.1.0-py3-none-win_amd64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import fastyolo; print(fastyolo.__version__)"   # 0.1.0
python examples/demo.py --model yolo11n.onnx --image photo.jpg --device CUDA   # Linux GPU 版
```

`FastYolo(model, device=InferDevice.X, num_threads=, min_score=, iou_threshold=, gpu_id=, class_names=[...])`；
`predict(img)` / `track(img)`（逐帧调用保持 track_id）。实例非线程安全。

> 2026-09 起：`SegResult.mask` 为 **ROI**（每框 `[h, w]`）且默认不生成——离线取掩码用
> `predict(img, need_masks=True)`，在线逐帧场景走 JNI 预算表路径；`track()` 只出框。
> 详见 [`docs/SEGMENT-lazy-migration.md`](docs/SEGMENT-lazy-migration.md)。
> 另：ctypes 用的纯 C API 需以 `-DPYTHON_BACKEND=ON` 构建，否则报 `undefined symbol: fastyolo_create`。

### 8.4 ONNX-CUDA 的部署形态（推荐目录式而非 slim wheel）

GPU 版运行时需要 340MB 的 ORT GPU 运行时 + cuDNN（约 1GB 的 9.1 全模块，精简后只留 conv 相关也可），
**slim wheel 打包不下这些**。推荐用 collect_artifacts 收集完整目录后打包 zip 分发：

```bash
bash python/scripts/build_linux.sh --gpu
# 自包含（含 cuDNN，客户机零配置）：
python3 python/scripts/collect_artifacts.py build-py --backend onnx --runtime full \
        --cudnn-lib <cudnn 9.x 的 lib 目录>
zip -r fastyolo-linux-gpu.zip python/fastyolo
```

目标机三选一即可（从省事到省事）：
1. **自包含**：包内 `fastyolo/libs/` 同时有 ORT 运行时 + `libcudnn*.so.9`（构建已把 providers 打成 `$ORIGIN`）→ 什么都不用装，只要驱动；
2. 系统装 `libcudnn9-cuda-12`（apt）；
3. `LD_LIBRARY_PATH` 指向 cuDNN 目录。

---

## 9. 产物清单与部署目录布局

### Linux（ONNX-CUDA 完整部署包示例，全部同目录）

```
lib/
├── libFastYolo.so                        # 主库（核心 + ByteTrack + 可选 JNI/C-API）
├── libyolo_onnx.so                       # ONNX 后端适配层
├── libonnxruntime.so.1 → .so.1.x.y       # ORT GPU 主库（版本与 providers 同批）
├── libonnxruntime_providers_cuda.so      # CUDA EP（构建时已打 RUNPATH=$ORIGIN）
├── libonnxruntime_providers_shared.so    # EP 注册桥
└── libcudnn*.so.9                        # 可选：自包含时放入（与上表 cuDNN 版本要求一致）
```

系统级依赖（不在包内）：NVIDIA 驱动 + `libcublas/libcublasLt/libcudart.so.12`（装 CUDA 12 即有）。

### Windows（ONNX CPU 示例）

```
bin\Release\
├── FastYolo.dll          # 或 FastYolod.dll（Debug）
├── yolo_onnx.dll
└── onnxruntime.dll
```

---

## 10. 版本依赖矩阵

### 10.1 仓库当前锁定版本（三者必须配套，勿单独升级）

| 组件 | 版本 | 位置 |
|---|---|---|
| ONNX Runtime（CPU 静态） | **1.24.4**（需自备 tgz；本工作副本 `lib/linux/` 为空） | `source/backend/onnx/lib/linux/libonnxruntime.a` |
| ONNX Runtime（GPU 共享） | **1.23.2**（当前工作副本；1.24.4 / 1.29.0 实测同样可用，CUDA 12 构建） | `source/backend/onnx/lib/linux-gpu/` |
| ONNX Runtime（Windows） | 仓库自带 dll 套件 | `source/backend/onnx/lib/dll/` |
| 头文件 | ORT `ORT_API_VERSION 23`（v23 代码兼容 v23/v24 库） | `source/backend/onnx/include/` |

升级方法：去 [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases) 下载
`onnxruntime-linux-x64-<ver>.tgz`（CPU）与 `onnxruntime-linux-x64-gpu-<ver>.tgz`（GPU），
**同时替换** `include/` 与两侧 lib；网络不通可用 `https://gh-proxy.com/<原URL>` 前缀加速。

### 10.2 ORT × CUDA × cuDNN 官方矩阵（Linux）

| ORT 版本 | 需要的 CUDA | 需要的 cuDNN |
|---|---|---|
| 1.21 – 1.26 | **12.x**（官方以 12.8 构建） | **9.x** |
| 1.27+ | **13.x** ⚠️ | 9.x |
| 1.19 – 1.20 | 12.x | 9.x |
| 1.18.0 / 1.17 | 12.x* | 8.x |
| ≤ 1.20（CUDA11 线） | 11.8 | 8.x |

实测校正（官方表格与预编译包有出入，以实测为准）：

- **1.18.0 的 `linux-x64-gpu` C++ 包实际链接的是 CUDA 11 库**（`libcublasLt.so.11`）——
  在 CUDA 12 环境不可用，别按表格选它；
- pip 的 `onnxruntime-gpu ≥ 1.27` 默认 CUDA 13 构建，本机（CUDA 12.8）直接加载失败；
- CUDA EP 运行时精确依赖：`libcudart.so.12`、`libcublas.so.12`、`libcublasLt.so.12`（CUDA 12 自带）+ `libcudnn.so.9`。

### 10.3 cuDNN 9 的显卡架构门槛（最容易踩的坑）

| GPU | 可用 cuDNN 9 版本 |
|---|---|
| Turing 及更新（RTX 20系 / 30 / 40 等，sm ≥ 75） | 任意 9.x（官方最新亦可） |
| **Pascal / Maxwell（GTX 10 系等，sm 5.x–6.x）** | 需较旧的 9.x，**实测 9.1.0.70 可用**；最新 9.25 已放弃（Conv 报 `CUDNN_BACKEND_API_FAILED`）。中间分界版本未逐一实测，客户机有老卡就**锁 9.1** 最稳 |

- cuDNN 9 获取途径：pip wheel `nvidia-cudnn-cu12==9.1.0.70`（解包即用，注意全模块约 976MB，
  目标机已有则不必带）；或 NVIDIA 官网 tar/deb。
- **没有"完全不需要 cuDNN 的 CUDA 后端"**（官方 CUDA EP 依赖它）；部署时按第 8.4 节三选一。

---

## 11. 常用构建命令速查

```bash
# ---- Linux ----
cmake -S . -B b -DONNX_GPU=OFF                      # ONNX CPU 静态
cmake -S . -B b                                     # ONNX CUDA（默认，有 lib/linux-gpu 即启用）
cmake -S . -B b -DOPENVINO_BACKEND=ON -DOPENVINO_ROOT=$OV     # OpenVINO
cmake -S . -B b -DTENSORRT_BACKEND=ON               # TensorRT（engine 模型）
cmake -S . -B b -DMNN_BACKEND=ON                    # MNN
cmake -S . -B b -DJNI_BACKEND=ON -DPYTHON_BACKEND=ON          # 叠加绑定
cmake --build b -j$(nproc)

# ---- Windows（cmd / VS 生成器）----
cmake -B b -G "Visual Studio 17 2022" -DJNI_BACKEND=ON
cmake --build b --config Release -j
:: Python SDK 一把梭：python\scripts\build_windows.bat

# ---- 产物收集 / wheel ----
python3 python/scripts/collect_artifacts.py b --backend onnx --runtime full [--cudnn-lib <dir>]
cd python && python3 scripts/build_wheel.py [--full]
```

---

---

## 12. 故障排查汇总

| 症状 | 原因 | 处理 |
|---|---|---|
| `[yolo_onnx] CUDA EP unavailable, fallback to CPU: ... Failed to load shared library` | 系统找不到 `libcudnn.so.9`（或驱动缺失） | 按 §8.4 三选一补 cuDNN；`ldd providers_cuda.so` 确认还缺什么 |
| Conv 节点报 `CUDNN_FE failure 11: CUDNN_BACKEND_API_FAILED`（GTX 10/16 系老卡） | cuDNN 9.x 新版本放弃 Pascal/Maxwell | 换 cuDNN 9.1（见 §10.3） |
| `Failed to load ... error: libcublasLt.so.11: cannot open shared object file` | 用了 CUDA 11 构建的 ORT 包（如官方 1.18 C++ 包） | 换 CUDA 12 构建（1.21–1.26） |
| pip 版 CUDA EP 警告 `Require cuDNN 9.* and CUDA 13.*` | onnxruntime-gpu ≥1.27 是 CUDA 13 构建 | 锁 `onnxruntime-gpu==1.26.*` 或升级 CUDA 13 |
| CUDA EP enabled 后报 `Invalid handle. Cannot load symbol cublasLtGetVersion` | ORT 进程内加载 cuBLAS/cuDNN 组合与老卡（Pascal）环境不匹配；GTX 1060 实测 1.23.2 / 1.24.4 / 1.29.0 均复现 | 先按 §10.3 锁 cuDNN 9.1；仍失败时用 CPU EP 验证逻辑（**后处理与 EP 无关**），GPU 场景换 RTX 20 系以上验证（其驱动/cuBLAS 组合正常） |
| `onnxruntime.dll: 动态链接库(DLL)初始化例程失败`（Windows+Java） | JDK/bin 自带旧版 VC++ 运行时抢先加载 | 复制 `C:\Windows\System32\` 的 `vcruntime140[_1].dll`、`msvcp140[_1].dll` 到 `%JAVA_HOME%\bin\`（JVM 未运行时操作） |
| `FastYolo.dll: 找不到指定的模块/程序` | 三 DLL 不同目录，或缺 VC++ Redistributable | 全部放同一目录；装 VC++ 2022 Redist |
| `is not a valid Win32 application` | DLL 位数与 JVM 不匹配 | 统一 x64 |
| CMake 报 `JAVA_HOME not set` / `libjvm not found` | 未设 JAVA_HOME 或 JDK 不含 dev 组件 | `-DJAVA_HOME=...`，常见路径见第 4 节 |
| MSVC 编译 `yolo::STATUS_SUCCESS` 报 C2589 | `STATUS_SUCCESS` 宏与 winnt.h 冲突 | JNI 代码用字面量 `0`，勿写 `yolo::STATUS_XXX` |
| Java 调 `track(budgetIds, budgetAreas)` / `predict(…, needMask)` 报 `UnsatisfiedLinkError` | 新增 native 方法未写入 `javac -h` 生成头 → 符号被 C++ 改名（`_Z…`） | 按 §7.2 重新生成 JNI 头（`com_fastyolo_YoloInference.h`）并重编；本次升级已踩坑 |
| Python 报 `AttributeError: ... undefined symbol: fastyolo_create` | 构建未加 `-DPYTHON_BACKEND=ON`（或重新 configure 时被缓存覆盖为 OFF） | `cmake -S . -B <build> -DPYTHON_BACKEND=ON …` 后重编 |
| 掩码长度不再是 `imgW*imgH` / 下游数组越界 | 2026-09 起掩码为 ROI（长度 = 框 `width*height`），旧整幅假设代码未迁移 | 见 `docs/SEGMENT-lazy-migration.md` §4/§7；索引公式 `(px-x)+(py-y)*width` |
| `python scripts\build_wheel.py --full` 报错找不到 openvino | `--full` 实现依赖本机 openvino 包 | 先 `pip install openvino`；GPU 部署走 §8.4 目录式 |
| 构建后 libs/ 里 CUDA 运行时消失 | `collect_artifacts.py` 清理与当前模式不匹配的文件 | 用 `--runtime full`（必要时加 `--cudnn-lib`）重新收集 |
| `postprocess.cpp` 等文件"缺失/异常"（SVN） | 本地未版本化同名文件 obstruct 了 `svn update` | 删本地挡路文件再 update；或从 `.svn/pristine/<sha1前两位>/<sha1>.svn-base` 还原 |
| Git Bash 下 `dumpbin /exports` 无输出/报错 | MSYS 把 `/exports` 转成 Windows 路径 | 命令前加 `MSYS_NO_PATHCONV=1` |
| Windows 构建/加载时"另一个程序正在使用此文件" | 杀软扫描大 DLL（WinError 32） | SDK loader 已内置指数退避重试，稍候重跑脚本 |
| GitHub 下载超时 | 网络不通 | 用 `https://gh-proxy.com/` 前缀；pip 走清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |

---

## 附录 A：本机实测记录

**环境（2026-09-01/02，构建与 GPU 验证）**

- Ubuntu，NVIDIA driver 580.173.02，CUDA 12.8（`/usr/local/cuda-12.8`），系统另有 cuDNN 8.9.7（本方案用不到，属 TensorRT/OpenVINO 旧依赖）。
- 构建：`cmake -S . -B build-onnx-gpu -DPYTHON_BACKEND=ON` → `Using ONNX Runtime GPU (CUDA EP)`。
- 验证（`python/examples/demo.py --device CUDA`，真实摄像头照片，yolo11n）：
  - `CUDA` 与 `CPU` 两种 device 的检测结果逐位一致（boat 0.654 / cake 0.546，坐标相同）；
  - 系统缺 cuDNN9 时自动回退 CPU（日志含 fallback 原因），推理不中断；
  - `providers_cuda.so` 打 `$ORIGIN` + cuDNN 9.1 同目录后，**零环境变量自包含运行成功**。
- Python 侧冒烟（ORT 1.21 wheel 交叉验证）：yolo11n 10.3ms/帧、yolo11n-seg 12.4ms/帧、yolo26n 9.9ms/帧（GTX 1060 5GB，640×640），与 CPU 输出 maxdiff ≤2e-3（浮点级），yolo26 输出仅 TopK 排序差异。
- `collect_artifacts.py --backend onnx --runtime full --cudnn-lib <dir>` 一条命令收出完整 GPU 部署包。

**环境（2026-09-10，掩码懒加载升级验证）**

- 机器：Hygon C86-3G（8C16T，AVX2）、GTX 1060 5GB、ORT 1.23.2（CUDA 12 构建，GPU EP 因
  `cublasLtGetVersion` 兼容问题未能在该老卡启用，CPU EP 下完成全部后处理测量——后处理为纯 CPU，
  结论与 EP 无关）。
- 纯后处理基准（`MyFastYolo/bench_postonly`，推理只跑一次后循环解码/NMS/掩码/轮廓）：
  - 懒帧（decode+NMS）0.52–0.54ms/帧（≈1900fps 上限，0.027 核@50fps）；
  - ROI 掩码帧（全量）：face 1 目标 2.46ms、4K 1 大目标 4.37ms、4K 4 目标 4.43ms；
  - `findContours` 净耗时（ROI）：face +0.20ms、4K 4 目标 +0.38ms（旧整幅 4 目标为 14.9ms）；
  - 模拟轨迹（每 40 帧评估 3 帧）：4K 4 目标平均 0.81ms/帧（含轮廓 0.84ms）≈0.041 核@50fps，
    对照旧实现 ~30ms/帧 ≈1.5 核@50fps。
- 正确性：ROI 掩码非零像素与旧整幅掩码逐位一致（face 183379）；`MyFastYolo/test_lazy_track`
  预算语义 4 场景通过；Python SDK 实跑（懒帧/`need_masks=True`/track 连续性）通过；
  Java→JNI 联调（懒帧、预算{0}/预算{best}/无预算、`predict(needMask)`）5 场景通过。

---

## 附录 B：文档归档说明

本 README 整合了仓库原有的分散编译笔记，原文件已移入 `docs/history/` 仅作历史参考：

| 归档文件 | 原内容去向 |
|---|---|
| `编译指南.md` | 本文第 3–12 章与附录 A（同源升级） |
| `jni编译.txt` | §7（JNI：头文件生成、加载顺序、Windows VC++ 运行时坑） |
| `mnn-jni-windows编译上下文.md` | §6.4（Windows MNN+OpenCL 完整流程与踩坑） |
| `python安装.md` | §8（wheel 制作 / 离线安装 / 常见问题） |
| `nms测试环境上下文.md` | §1.2/§3（架构速览）与附录 A（基线数据） |
| `ai-session.md` | §1（项目介绍：三层抽象、ByteTrack、关键类） |
| `日志.txt` | 纯构建日志，已删除 |

> 注：历史文档中的 `jni方案.md` 在当前工作副本中已不存在，本文不再引用；
> JNI 接口与调用流程以 §7 与 `docs/SEGMENT-lazy-migration.md` 为准。

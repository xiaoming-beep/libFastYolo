# ONNX Runtime 后端（默认）

> 顶层开关：**默认后端**，其他后端全关时自动编译；`-DONNX_BACKEND=ON` 只是名义开关。
> 适配层：`source/backend/onnx/yolo_onnx.cpp`。

ONNX Runtime 是仓库的默认后端，Linux 上分 **CPU 静态**与 **CUDA GPU** 两种链接形态，
同一份代码、同一个二进制同时具备 CPU 与 CUDA 能力，运行时按 `YOLOConfig::device` 决定。

---

## 1. 平台与库布局

| 平台 | 链接的库 | 位置 |
|---|---|---|
| Linux CPU | `libonnxruntime.a`（静态） | `source/backend/onnx/lib/linux/` |
| Linux GPU | `libonnxruntime.so`（共享，CUDA EP） | `source/backend/onnx/lib/linux-gpu/` |
| Windows | `onnxruntime.lib` + `onnxruntime.dll` | `source/backend/onnx/lib/dll/` |

> **工作副本里这些目录可能是空的**（例如 `lib/linux/` 被 `.gitignore` 忽略）。
> 从官方 tgz 解包放入即可：CPU 用 `onnxruntime-linux-x64-*.tgz`，GPU 用
> `onnxruntime-linux-x64-gpu-*.tgz`，版本配套见 [版本矩阵](../appendix/version-matrix.md)。
> 头文件在 `source/backend/onnx/include/`。

---

## 2. Linux — CPU（静态链接，最简）

```bash
cd /path/to/libFastYolo
cmake -S . -B build -DONNX_GPU=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

- CMake 配置日志出现 `Using static ONNX Runtime: .../libonnxruntime.a` 即成功。
- 产物运行时**不依赖任何 onnxruntime 文件**：`build/lib/libFastYolo.so` + `build/lib/libyolo_onnx.so`
  两个 `.so` 放同一目录即可部署（主库经 `$ORIGIN` 找后端库）。

## 3. Linux — CUDA GPU（默认开启）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # 检测到 lib/linux-gpu 即自动走 GPU
cmake --build build -j$(nproc)
```

- CMake 日志应出现：`Using ONNX Runtime GPU (CUDA EP): .../source/backend/onnx/lib/linux-gpu`。
- 构建的库**同时具备 CPU 和 CUDA 能力**：运行时按 `YOLOConfig::device` 决定——
  `CUDA`/`AUTO` 先注册 CUDA EP（`gpu_id` 选卡），**任何失败（无驱动/无卡/cuDNN 缺失或架构不支持）
  打印日志后自动回退 CPU**；`CPU` 直接走 CPU。
- 构建机不要求有 GPU（链接的是 GPU 共享库；没有 `patchelf` 时给 WARNING 继续）。
- 产物：除主库外还有 `libonnxruntime.so.1(.x.y)`、`libonnxruntime_providers_cuda.so`
  （约 316MB，含 sm_50–sm_90 全架构 fatbin）、`libonnxruntime_providers_shared.so`，
  以及构建时自动收集的 CUDA 12 运行时（cuBLAS/cuBLASLt/cudart/nvJitLink/curand/cufft）与
  （探测到时的）cuDNN 9。
- **自包含部署关键步骤（构建里已固化）**：POST_BUILD 用 `patchelf` 给产物目录中的
  `libonnxruntime_providers_cuda.so` 打 `RUNPATH=$ORIGIN`，使同目录放置的 cuDNN `.so`
  可被解析——整包拷贝 + cuDNN 放同一目录即可**零环境变量**运行。仓库源文件
  `lib/linux-gpu/` 保持未打补丁的原版。

运行与验证：

```bash
# 方式 A：目标机系统里有 libcudnn.so.9（apt 安装或 LD_LIBRARY_PATH 提供）
./demo --device cuda

# 方式 B：完全自包含 —— cuDNN 的 .so 与 providers_cuda.so 放同一目录，不设任何环境变量
ldd build/lib/libonnxruntime_providers_cuda.so | grep cudnn   # 应指向同目录
```

日志出现 `[yolo_onnx] CUDA EP enabled, gpu_id=0` 即成功；回退时打印
`[yolo_onnx] CUDA EP unavailable, fallback to CPU: ...`。

### 3.1 cuDNN 选择（最容易踩的坑）

- `-DONNX_CUDNN_LIB_DIR=/path/to/dir` 可显式指定 cuDNN 9 目录；不指定时按
  `lib/linux-gpu` → `lib/linux-cudnn-cu12` → `python/fastyolo/libs` → 系统常见路径自动探测。
- 必须选 **CUDA 12 变体**的 cuDNN 9（`libcudnn_engines_precompiled.so.9` 内含
  `libcublasLt.so.12`），否则 CMake 直接 FATAL_ERROR。
- Pascal / Maxwell（GTX 10 系）需较旧的 9.x，实测 **9.1.0.70** 可用；cuDNN ≥ 9.11 已剔除
  sm_61 内核。详见 [版本矩阵](../appendix/version-matrix.md#3-cudnn-9-的显卡架构门槛)。

---

## 4. Windows — CPU

```bat
cmake -B build-msvc -G "Visual Studio 17 2022"
cmake --build build-msvc --config Release -j
```

- Windows 分支固定链接 `lib/dll/onnxruntime.lib`，POST_BUILD 拷 `onnxruntime.dll` 到产物目录。
- 产物三件套必须同目录：`FastYolo.dll` + `yolo_onnx.dll` + `onnxruntime.dll`。
- 见 [Windows 编译](../build/windows.md)。
- Windows GPU 路线暂未实现；同一后端将来可加 DirectML EP，覆盖 N / A / I 卡且不需 CUDA。

---

## 5. 与其他绑定组合

```bash
# Linux：ONNX + JNI + Python C API
cmake -S . -B build -DJNI_BACKEND=ON -DPYTHON_BACKEND=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

GPU 版做 Python 部署时不要把 340MB 运行时塞进 slim wheel，改用目录式分发，
见 [Python SDK](../build/python-sdk.md#4-onnx-cuda-的部署形态推荐目录式而非-slim-wheel) 与
[部署布局](../guides/deployment.md)。

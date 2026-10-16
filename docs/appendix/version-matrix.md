# 版本依赖矩阵

## 1. 仓库当前锁定版本（三者必须配套，勿单独升级）

| 组件 | 版本 | 位置 |
|---|---|---|
| ONNX Runtime（CPU 静态） | **1.24.4**（需自备 tgz；本工作副本 `lib/linux/` 为空） | `source/backend/onnx/lib/linux/libonnxruntime.a` |
| ONNX Runtime（GPU 共享） | **1.23.2**（当前工作副本；1.24.4 / 1.29.0 实测同样可用，CUDA 12 构建） | `source/backend/onnx/lib/linux-gpu/` |
| ONNX Runtime（Windows） | 仓库自带 dll 套件 | `source/backend/onnx/lib/dll/` |
| 头文件 | ORT `ORT_API_VERSION 23`（v23 代码兼容 v23/v24 库） | `source/backend/onnx/include/` |

**升级方法**：去 [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases) 下载
`onnxruntime-linux-x64-<ver>.tgz`（CPU）与 `onnxruntime-linux-x64-gpu-<ver>.tgz`（GPU），
**同时替换** `include/` 与两侧 lib；网络不通可用 `https://gh-proxy.com/<原URL>` 前缀加速。

## 2. ORT × CUDA × cuDNN 官方矩阵（Linux）

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
- CUDA EP 运行时精确依赖：`libcudart.so.12`、`libcublas.so.12`、`libcublasLt.so.12`
  （CUDA 12 自带）+ `libcudnn.so.9`。

## 3. cuDNN 9 的显卡架构门槛（最容易踩的坑）

| GPU | 可用 cuDNN 9 版本 |
|---|---|
| Turing 及更新（RTX 20 系 / 30 / 40 等，sm ≥ 75） | 任意 9.x（官方最新亦可） |
| **Pascal / Maxwell（GTX 10 系等，sm 5.x–6.x）** | 需较旧的 9.x，**实测 9.1.0.70 可用**；最新 9.25 已放弃（Conv 报 `CUDNN_BACKEND_API_FAILED`）。中间分界版本未逐一实测，客户机有老卡就**锁 9.1** 最稳 |

- cuDNN 9 获取途径：pip wheel `nvidia-cudnn-cu12==9.1.0.70`（解包即用，注意全模块约 976MB，
  目标机已有则不必带）；或 NVIDIA 官网 tar/deb。
- **必须用 CUDA 12 变体**：cuDNN 子库内部按硬编码 soname `dlopen` cuBLAS，cu13 变体要
  `libcublasLt.so.13`、cuda11 变体要 `.so.11`，与本 CUDA 12 构建不匹配。CMake 会校验
  `libcudnn_engines_precompiled.so.9` 是否含 `libcublasLt.so.12`。
- **没有"完全不需要 cuDNN 的 CUDA 后端"**（官方 CUDA EP 依赖它）；部署时按
  [Python SDK](../build/python-sdk.md#4-onnx-cuda-的部署形态推荐目录式而非-slim-wheel) 三选一。

## 4. 其他后端版本

| 后端 | 版本 | 备注 |
|---|---|---|
| OpenVINO | `pip install openvino` 最新即可 | 构建时 `-DOPENVINO_ROOT` 指过去 |
| MNN | 3.6.1（Windows 验证） | Linux 自带 `libMNN.so` 为 CPU 构建 |
| TensorRT | 与本机 CUDA 配套 | engine 绑定 TRT 版本与具体显卡 |
| RKNN | rknn-toolkit2 2.3.2 / librknnrt.so 2.3.2 | RK3576 / RK3588，aarch64 |
| onnxcl | 无第三方版本 | C++17+ 编译器 + Python3（构建期） |

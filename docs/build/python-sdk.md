# Python SDK 与 wheel 分发

> 源码包：`python/fastyolo/`；构建脚本：`python/scripts/`。
> 底层是 libFastYolo 的**纯 C API**（`source/include/fast_yolo/fastyolo_c.h`，ctypes 调用），
> 需要以 `-DPYTHON_BACKEND=ON` 构建，否则报 `undefined symbol: fastyolo_create`。

安装即拷贝：包是纯 Python + 预编译动态库，**使用机不需要 cmake / 编译器 / VS**；
`py3-none` 标签不绑定 Python 版本（3.8+ 通用）。

---

## 1. 流程总览

```
编译机（cmake + 编译器，一次性）                   使用机（只要 Python）
  build_windows.bat / build_linux.sh                  pip install fastyolo-0.1.0-*.whl
  → collect_artifacts.py（产物进 python/fastyolo/libs/）      │
  → build_wheel.py → dist/*.whl  ──── 微信/网盘/内部 PyPI ────┘
```

---

## 2. 各平台构建

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
cd python && python3 scripts/build_wheel.py # 仅对 CPU 版有意义（见第 4 节）
```

构建脚本做的事：CMake 配置（后端 + `-DPYTHON_BACKEND=ON`）→ 编译 →
`collect_artifacts.py` 把核心库 + 后端库收集到 `fastyolo/libs/`。

---

## 3. 安装与验证

```bash
pip install fastyolo-0.1.0-py3-none-win_amd64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import fastyolo; print(fastyolo.__version__)"   # 0.1.0
python examples/demo.py --model yolo11n.onnx --image photo.jpg --device CUDA
```

wheel 说明：

- 精简版（默认，~2MB）：OpenVINO 运行时从已安装的 `openvino` pip 包加载；
- `--full` 全捆绑版（~170MB）：运行时打进包，目标机可完全离线（实现上依赖本机装有
  openvino pip 包）。
- 注意 `pyproject.toml` 无条件声明了 `openvino>=2024.0` 依赖——Linux 精简 wheel 安装时
  也会拉 openvino，属已知待优化点。

---

## 4. ONNX-CUDA 的部署形态（推荐目录式而非 slim wheel）

GPU 版运行时需要 340MB 的 ORT GPU 运行时 + cuDNN，**slim wheel 打包不下这些**。
推荐用 `collect_artifacts` 收集完整目录后打包 zip 分发：

```bash
bash python/scripts/build_linux.sh --gpu
# 自包含（含 cuDNN，客户机零配置）：
python3 python/scripts/collect_artifacts.py build-py --backend onnx --runtime full \
        --cudnn-lib <cudnn 9.x 的 lib 目录>
zip -r fastyolo-linux-gpu.zip python/fastyolo
```

目标机三选一即可：

1. **自包含**：包内 `fastyolo/libs/` 同时有 ORT 运行时 + `libcudnn*.so.9`
   （构建已把 providers 打成 `$ORIGIN`）→ 什么都不用装，只要驱动；
2. 系统装 `libcudnn9-cuda-12`（apt）；
3. `LD_LIBRARY_PATH` 指向 cuDNN 目录。

---

## 5. API 与目录

完整 API 参考、任务推断表、跟踪语义、常见问题见包内文档
[`python/README.md`](../../python/README.md)。

```
fastyolo/
├── __init__.py      # FastYolo 高层 API（任务分发）
├── _loader.py       # DLL 路径处理 + 加载（Windows os.add_dll_directory）
├── _binding.py      # ctypes 函数签名 + 结果转换
├── config.py        # YoloConfig / InferDevice / TaskType
├── letterbox.py     # numpy 双线性缩放 + C++ fill_bgr 零拷贝
├── result.py        # Detection / SegResult / PoseResult / ... / DepthResult
└── libs/            # 构建产物（核心库 + 后端库）
```

> 2026-09 起 `SegResult.mask` 为 **ROI** 且默认不生成——离线取掩码用
> `predict(img, need_masks=True)`，在线逐帧走 JNI 预算表路径；`track()` 只出框。
> 详见 [SEGMENT 迁移指南](../guides/segment-lazy-migration.md)。

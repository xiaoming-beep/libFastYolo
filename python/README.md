# fastyolo —— libFastYolo Python SDK

YOLO 多任务推理（检测 / 实例分割 / 姿态 / 旋转框 / 分类 / 语义分割 / 深度估计）
+ ByteTrack 多目标跟踪的 Python 封装，基于 libFastYolo 的纯 C 绑定层
（`source/include/fast_yolo/fastyolo_c.h`，ctypes 调用），**装完即用、跨 Python 版本**：

- 安装 = 拷贝：包是纯 Python + 预编译动态库，安装时不编译
- import 即自动就绪：包内 `_loader.py` 自动处理 DLL 搜索路径（Windows `os.add_dll_directory`），
  无需设置 PATH
- 同一份动态库兼容任意 Python 3.8+（绑定不依赖任何 Python 头文件）

## 快速开始

```python
import numpy as np
from fastyolo import FastYolo, InferDevice

yolo = FastYolo("yolo11n.onnx", device=InferDevice.CPU, num_threads=4)
dets = yolo.predict(bgr_img)   # bgr_img: np.ndarray HxWx3 uint8 BGR
for d in dets:
    print(d.x, d.y, d.width, d.height, d.cls_id, d.score)   # 原图坐标，整数像素

dets = yolo.track(bgr_img)     # ByteTrack 跟踪，d.track_id >= 0（需连续帧调用）

# 跟踪支持所有带框任务：SEGMENT/POSE/OBB 同样可用，载荷（掩码/关键点/角度）保留
segs = seg.track(bgr_img)      # list[SegResult]，s.box.track_id >= 0
```

多任务示例（`predict()` 按任务自动返回类型化结果）：

```python
from fastyolo import FastYolo, TaskType

seg = FastYolo("yolo11n-seg.onnx")            # 任务自动推断
for s in seg.predict(bgr_img):                # list[SegResult]
    print(s.box, s.mask.shape)                # mask: 原图分辨率二值 uint8

cls = FastYolo("yolo11n-cls.onnx")            # 分类：自动中心裁剪预处理（Resize+CenterCrop）
r = cls.predict(bgr_img)                      # ClassifyResult
print(r.top_class, r.top_score, r.top_ids)

obb = FastYolo("yolo11n-obb.onnx", task=TaskType.OBB)   # v8 OBB 与 DETECT 形状歧义，需显式指定
for o in obb.predict(bgr_img):                # list[ObbResult]
    print(o.box, o.angle, o.corners())        # AABB + 弧度角 + 4 角点

dep = FastYolo("yolo26n-depth.onnx")
r = dep.predict(bgr_img)                      # DepthResult（原图分辨率 float32 + 值域）
sem = FastYolo("yolo26n-sem.onnx")
r = sem.predict(bgr_img)                      # SemSegResult（原图分辨率 int32 类别图）
```

完整示例见 `examples/demo.py`：

```bash
python examples/demo.py --model yolo11n.onnx --image test.jpg --track --output out.jpg
python examples/demo.py --model yolo26n-sem.onnx --image test.jpg   # 语义分割
```

## 安装

**用户机器上不需要 cmake / 编译器 / VS** —— 安装就是纯拷贝（纯 Python + 预编译 DLL）。

### 方式 A：预编译 wheel（推荐，给使用者）

开发机/CI 上构建一次 wheel（见下），然后任意一种方式分发：

```bash
# 1. 直接安装 wheel 文件
pip install fastyolo-0.1.0-py3-none-win_amd64.whl

# 2. 上传到内部 PyPI / Nexus 后
pip install fastyolo -i http://<内部源>

# 3. 挂到 Web / 网盘
pip install https://<URL>/fastyolo-0.1.0-py3-none-win_amd64.whl
```

- wheel 自动携带 `numpy` + `openvino` 依赖，无需手动装
- `py3-none` 标签：**同一个 wheel 覆盖所有 Python 3.x**，不按 Python 版本区分
- 精简版（默认，~2MB）：OpenVINO 运行时从 `openvino` pip 包加载；
  `--full` 全捆绑版（~170MB）自包含、可离线安装

### 方式 B：源码构建（开发 / 定制后端）

```bash
# 1. 构建动态库（一次性，见下节）—— 产物收集到 fastyolo/libs/
# 2. 安装（两种方式等价）
pip install ./python            # 打包安装
# 或直接把 python/fastyolo/ 目录拷进 site-packages
```

## 构建

| 平台 | 后端 | 命令 | 前置条件 |
|---|---|---|---|
| Windows | OpenVINO（默认选择） | `python/scripts/build_windows.bat` | VS2022 BuildTools、CMake ≥ 3.21、`pip install openvino` |
| Linux | ONNX Runtime（默认选择） | `python/scripts/build_linux.sh` | GCC ≥ 11、CMake ≥ 3.21（libonnxruntime.a 随仓库自带） |
| Linux | OpenVINO（可选） | cmake 加 `-DOPENVINO_BACKEND=ON -DOPENVINO_ROOT=<site-packages/openvino>` | pip openvino |

构建脚本做的事：CMake 配置（后端 + `-DPYTHON_BACKEND=ON`）→ 编译 →
`collect_artifacts.py` 把核心库 + 后端库收集到 `fastyolo/libs/`（约 1.5MB）。

**运行时策略（与 C++ 同思路，不重复携带依赖）**：OpenVINO 运行时 DLL 不随项目/
wheel 复制，加载时自动从已安装的 `openvino` pip 包（site-packages/openvino/libs）
获取——这正是 C++ 构建时 `-DOPENVINO_ROOT` 指向的同一批文件。定位优先级：
1. 环境变量 `FASTYOLO_OPENVINO_ROOT`（手动指定，可指向 site-packages/openvino）
2. 自动探测已安装的 openvino 包

需要完全离线部署时：`python scripts/collect_artifacts.py build-py --backend openvino --runtime full`
把运行时收进 libs/，再 `python scripts/build_wheel.py --full` 出全捆绑 wheel（~63MB，
自动从 openvino 包补齐运行时并去掉 openvino 依赖）。

**产出可分发的 wheel**（在已构建的基础上）：

```bash
python scripts/build_wheel.py         # 精简版（默认，~2MB，依赖 openvino pip 包）
python scripts/build_wheel.py --full  # 全捆绑（~170MB，离线可装）
```

> 后端绑定于构建产物：Windows 构建的包 = OpenVINO 后端，Linux 构建的包 = ONNX 后端，
> 换后端需重新构建。Linux/OpenVINO 组合**未实测**（仓库文档已验证 Linux GCC + ONNX 可行）。

## 后端对照

| 后端 | 支持平台 | 模型格式 | 设备 | 说明 |
|---|---|---|---|---|
| OpenVINO | Windows / Linux | ONNX 单文件（**不支持 IR .xml+.bin 双文件**） | CPU / NPU / OPENCL(→GPU) / AUTO；VULKAN/CUDA 回退 CPU | 运行时来自 pip `openvino` 包，构建时自动定位 |
| ONNX Runtime | Windows / Linux | .onnx | CPU（`device` 其他值由核心层透传，行为以后端为准） | Linux 静态链接，产物自包含 |

macOS 不在支持范围（C++ 工程本身无 macOS 分支）。

## API 参考

- `FastYolo(model, *, config=None, box_size=None, letterbox_color=114.0, class_names=None, **overrides)`
  — 高层入口。`model` 为路径或字节流；`box_size=None` 时按模型输入尺寸自动取
  （如 640/224/1024/768）；`overrides` 覆盖 `YoloConfig` 字段；
  构造时自动 init 并暴露 `input_shape` / `output_shape` / `output_shapes` / `task`（解析后任务）
- `FastYolo.predict(image, score_threshold=-1.0, iou_threshold=-1.0)` — 按任务返回：
  - DETECT → `list[Detection]`
  - SEGMENT → `list[SegResult]`（box + 原图分辨率二值掩码）
  - POSE → `list[PoseResult]`（box + 关键点）
  - OBB → `list[ObbResult]`（box + 弧度角 + `corners()`）
  - CLASSIFY → `ClassifyResult`（top1 + top_k + 全量分数）
  - SEMSEG → `SemSegResult`（原图分辨率 int32 类别图）
  - DEPTH → `DepthResult`（原图分辨率 float32 + 值域）
- `FastYolo.classify()/semseg()/depth()` — 稠密任务的类型化便捷入口（类型不符抛 `YoloError(-5)`）
- `FastYolo.track(image, ...)` — 检测 + ByteTrack（支持 DETECT/SEGMENT/POSE/OBB，
  带框任务返回与 predict 相同的类型化结果并回填 `box.track_id`；稠密任务不支持，抛 -5）
- `Detection` / `Keypoint` / `SegResult` / `PoseResult` / `ObbResult` / `ClassifyResult` /
  `SemSegResult` / `DepthResult` — 结果类型（frozen dataclass）
- `YoloConfig` — 配置 dataclass，字段/默认值与 C++ `yolo::YOLOConfig` 一致
  （`min_score=0.25, iou_threshold=0.45, device=CPU, num_threads=1, task=AUTO,
  num_classes=0, num_keypoints=17, top_k=5, ...` 及 ByteTrack 参数）
- `InferDevice` / `TaskType` — 枚举（值与 C++ 一致）
- `Letterbox` — 底层预处理器（`process(image)` 一条龙；`mode="crop"` 中心裁剪模式供分类；
  `pixels` 属性只读缓冲）
- `YoloError` — 错误异常，`code` 为 `STATUS_*` 错误码（0 / -1 / -2 / -3 / -4 / -5）

## 任务与 AUTO 推断

`TaskType.AUTO`（默认）按模型输出形状推断任务，覆盖常见模型：

| 输出形状 | 推断任务 |
|---|---|
| `[1, N]` | CLASSIFY |
| `[1, H, W]`（uint8/int64） | SEMSEG |
| `[1, 1, H, W]`（float） | DEPTH |
| 检测头 + `[1, 32, H, W]` 原型掩码 | SEGMENT |
| `[1, N, 6]` / `[1, N, 7]` / `[1, N, 38]`（N≤300，端到端） | DETECT / OBB / SEGMENT |
| `[1, C, N]`（C = 4+nc+3k，k≥10） | POSE |
| 其余 `[1, C, N]` | DETECT |

**已知歧义**：v8 风格 OBB（`[1, 4+1+nc, N]`，如 yolo11n-obb）与 DETECT 形状相近，
AUTO 会解析为 DETECT —— 请显式指定 `task=TaskType.OBB`。

## 跟踪说明（ByteTrack）

- `track()` 支持 DETECT/SEGMENT/POSE/OBB：框类任务均可用，掩码/关键点/角度载荷
  与检测框一致保留，仅回填 `track_id`
- **阈值语义**：检测分数 ≥ `high_thresh`（默认 0.6）才建立新轨迹；
  `track_thresh`（默认 0.5）只用于丢失轨迹的找回关联。低置信度目标想跟踪时
  请调低 `high_thresh`（如 `YoloConfig(high_thresh=0.4)`）
- 稠密任务（CLASSIFY/SEMSEG/DEPTH）无检测框，不支持跟踪

## 关键契约（与 C++ 库一致）

1. **输入图像为 HxWx3 uint8 BGR**（OpenCV 读图即 BGR，不做色彩转换）；缩放插值在
   Python 侧完成（双线性，OpenCV `INTER_LINEAR` 同公式），C++ 侧只做 letterbox 填充
2. **letterbox 数学严格对齐 C++**：`scale=min(box/w, box/h)`、`dst=round(w*scale)`、
   灰边 114、BGR→NCHW float32 planar、`/255` 归一化（见 `tests/test_letterbox.py`）
3. **分类任务预处理**：自动使用中心裁剪模式（Resize 最短边 + CenterCrop，与
   ultralytics `classify_transforms` 一致），缩放用 PIL BILINEAR（依赖 Pillow），
   保证与 ultralytics 分类结果对齐
4. **模型为 ONNX 单文件字节流**（OpenVINO 后端从内存 `read_model`，不支持 IR 双文件）
5. **实例非线程安全**（同 Java SDK）：多线程请每线程一个实例或外部加锁
6. 结果坐标/掩码/类别图/深度图已反算回原图，NMS 与后处理在 C++ 核心层完成；
   语义分割类别值限 < 256（uint8 存储）

## 常见问题

- **import 报"未找到 libFastYolo 动态库"** — 未构建或 libs/ 被清空，运行构建脚本
- **用户机器需要 cmake / VS 吗？** — 不需要。安装 = 拷贝；cmake 只在开发机构建 wheel 时用一次
- **`YoloError(-2)`（模型初始化失败）** — 模型格式不支持（如 IR .xml+.bin）、
  模型损坏、或所选 device 后端不支持（OpenVINO 对 VULKAN/CUDA 会打印回退 CPU 警告）
- **`YoloError(-3)`** — 句柄已被 close/destroy 后继续调用
- **想换后端** — 重跑对应平台构建脚本，再 `python scripts/build_wheel.py` 重新出包
- **Python 升级后 DLL 加载失败** — 本包与 Python 版本无关，重装包即可，无需重新构建

## 目录结构

```
fastyolo/
├── __init__.py      # FastYolo 高层 API（任务分发，公开入口）
├── _loader.py       # DLL 路径处理 + 加载
├── _binding.py      # ctypes 函数签名 + 统一 FastyoloResult 转换
├── config.py        # YoloConfig / InferDevice / TaskType（与 C++ 枚举值一致）
├── letterbox.py     # numpy 双线性缩放 + C++ fill_bgr 零拷贝（pad / crop 模式）
├── result.py        # Detection / SegResult / PoseResult / ObbResult / ClassifyResult / SemSegResult / DepthResult
└── libs/            # 构建产物（核心库 + 后端库；OpenVINO 运行时复用 openvino pip 包）
```

## 许可说明

项目本身无 LICENSE 声明。捆绑的第三方组件需遵守各自许可：
OpenVINO（Apache-2.0）、ONNX Runtime（MIT）、ByteTrack（Apache-2.0）、
Eigen（MPL-2.0）。发布前请补齐本项目许可证并核对第三方许可义务。

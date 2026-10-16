# RKNN 后端方案（Rockchip NPU）

> @author 方明 · 2026-09-15 与葛洲坝项目联调产出
> 目标硬件：RK3576 工控机（100.15/115/116，NPU 0.9.8 驱动）； equally 适用于 RK3588

## 1. 背景

葛洲坝高速事件识别项目采用 RK3576 做端侧推理，模型为 RKNN 格式
（rknn-toolkit2 2.3.2 转换，i8 量化）。本后端让 libFastYolo 的
C++/JNI/Python 三套 SDK 与 ByteTrack 直接跑在 NPU 上。

## 2. 架构落点

- 新增 `source/backend/RKNN/`：`yolo_rknn.cpp` 实现 `yolo_infer_*` 四函数契约，
  CMake `option(RKNN_BACKEND)` 接入顶层 elseif 链。
- 链接：`librknnrt.so`（2.3.2，aarch64），路径由 `RKNN_SDK_DIR`/`RKNN_LIBRARY` 传入。
- 输入：核心层 Letterbox 产出 float RGB CHW(0-1)，后端内转 uint8 NHWC(0-255)
  （模型转换时 mean=0/std=255）。约 1-2ms/帧。
- 输出：`rknn_outputs_get(want_float=1)` 自动反量化，核心解码器零改动。

## 3. 模型转换约定（重要）

标准图模型转换前**必须** `strip_sigmoid.py` 去掉输出路径上的 Sigmoid：

```
int8 量化下 Sigmoid 输出分支会整体饱和为 0（框分支 DFL 正常、类别分数全零）。
kl_divergence / mmse 量化算法均不能规避；fp16 不受影响但速度减半。
Rockchip 官方"优化图"（9/13 输出）替换 Sigmoid 为 ReduceSum 也是同款动机。
```

- strip 后类别分数为 logits，后端在反量化后**首帧自动判定**（分数区最大值 >1
  → logits 补 sigmoid；≤1 → 模型自带 sigmoid，跳过）。`FASTYOLO_RKNN_NO_SIG=0` 可强制关闭。
- 转换链路（200 机器 docker 镜像 rknn-convert:2.3.2）：
  1. `strip_sigmoid.py xxx.onnx xxx_nosig.onnx`
  2. `onnx2rknn.py --onnx xxx_nosig.onnx --platform rk3576 --dtype i8 --dataset calib.txt`

## 4. 板端编译

```bash
cd libFastYolo && mkdir build && cd build
cmake -DRKNN_BACKEND=ON -DRKNN_TEST=ON \
      -DRKNN_SDK_DIR=/home/hzhy/rknn_yolo/3rdparty/rknpu2 \
      -DRKNN_LIBRARY=/home/hzhy/rknn_yolo/3rdparty/rknpu2/Linux/aarch64/librknnrt.so \
      -DCMAKE_BUILD_TYPE=Release ..
make -j8
# 产物: lib/libFastYolo.so lib/libyolo_rknn.so lib/librknnrt.so bin/test_rknn
LD_LIBRARY_PATH=lib ./bin/test_rknn <model.rknn> <img.jpg> 300 predict|track
```

## 5. 实测（RK3576, NPU 850MHz, i8, 640×640, bus.jpg, 2026-09-15）

| 模型 | 图 | 全流水(含解码+NMS) | 备注 |
|---|---|---|---|
| **yolov8n** | **优化图 9 输出 + int8 直读** | **48.1 FPS** | 5 目标，分数与官方 demo process_i8 完全一致 |
| yolo11n | 优化图 9 输出 + int8 直读 | 43.9 FPS | |
| yolov8n | 标准图(strip sigmoid) | 38.6 FPS | 兜底路线 |
| yolo26n-seg | 标准图(strip sigmoid) | 32.2 FPS | SEGMENT 自动识别 |
| yolo26n | 标准图(原生) | 42.8 FPS | 框为归一化坐标，解码适配待做 |

优化图加速三板斧（缺一不可）：
1. **图改造**：解码卷积/Sigmoid/Concat 移出 NPU（NPU 17-21ms → 12.7ms），
   Sigmoid 换 ReduceSum 顺带绕开 int8 饱和问题；
2. **int8 直读**：want_float=0 传原始量化数据 + zp/scale（ABI 增设 elem_type=3
   与 zp/scale 字段），避免 runtime 全量转 float 的 11MB 拷贝（省 5-8ms）；
3. **量化域解码**：sum/score 在 int8 域直接比较阈值，仅对过筛候选反量化
   4*dfl_len 个框值（解码+NMU 仅 0.9ms）。注意 int8 输出内存布局为通道优先
   [C][h][w]（与 want_float 一致，与 rk3588 官方 process_i8 的逐位置假设不同）。

标准图模型仍被支持（strip_sigmoid + 后端首帧自动补 sigmoid），作为兜底路线。

## 6. 已知限制 / 待办

- yolo26 检测标准图输出为归一化坐标（实测 max≈1.0），核心层 decode 按像素解释
  会得到 1×1 框；需在解码层按模型适配（或 yolo26 训练导出时关掉归一化）。
- SEGMENT 优化图（13 输出）路由已识别但解码未实现（会干净报错）；当前分割走
  标准图路线。
- SEGMENT 掩码懒加载/ROI 已由核心层天然支持（need_masks / MaskBudget）。
- NPU 多核（rknn_set_core_mask）与零拷贝输入（rknn_create_mem）未启用，二期优化。
- 转换侧工具：`gezhouba/scripts/strip_sigmoid.py`、`onnx2rknn.py`（镜像内 /opt/）。

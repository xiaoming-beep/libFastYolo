# C++ 公共 API 速览

公共头：[`source/include/fast_yolo/yolo.h`](../../source/include/fast_yolo/yolo.h)。

---

## 1. 任务类型

`TaskType`：`AUTO / DETECT / SEGMENT / POSE / OBB / CLASSIFY / SEMSEG / DEPTH`。
`AUTO`（默认）按模型输出形状自动推断，覆盖常见 YOLO 家族。

## 2. 三层屏蔽设计

| 层 | 机制 | 说明 |
|---|---|---|
| 推理后端屏蔽 | C ABI 边界（`yolo_infer_create / init / predict / destroy` 四个 `extern "C"` 函数） | 核心层持 `void* context_`；后端编译为独立 `libyolo_*.so`，运行时经 `RPATH=$ORIGIN` 同目录加载；换后端只改 CMake option |
| 硬件设备屏蔽 | `InferDevice`（`AUTO/CPU/NPU/VULKAN/OPENCL/CUDA/ALL`）+ `YOLOConfig.device/gpu_id/num_threads/precision` | ONNX 后端按 device 注册 CUDA EP（失败自动回退 CPU 并打日志）；MNN 经 `MapInferDevice()` 映射；OpenVINO 映射 CPU/NPU/GPU |
| 模型差异屏蔽 | 读模型元数据回填 input/output shape；解码器按任务/形状自适应 | v8 头 `[1,C,N]`、端到端 `[1,N,6]`（免 NMS）、SEG 双输出、POSE `4+nc+3k`、OBB、SEMSEG/DEPTH 稠密输出等 |

## 3. 端到端数据流（含跟踪）

```
BGR uint8
  → Letterbox.update(w,h)                   # 计算 scale/padding（分类任务走中心裁剪）
  → Letterbox.fill_bgr(data)                # letterbox + BGR→float planar (RRR…GGG…BBB…)
  → yolo_infer_predict(context_, …)         # C ABI → 后端推理
  → postprocess 解码（按任务；SEGMENT 默认只出框，掩码按需）
  → Letterbox.complete_result(det)          # 坐标反算回原图
  → vector<YOLOResult>（track_id = -1）
  → [track() 时] BYTETracker.update()       # 两级关联 → 回填 track_id
```

## 4. ByteTrack 关键参数（`YOLOConfig`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `track_frame_rate` | 30 | 视频帧率 |
| `track_buffer` | 30 | 轨迹保持帧数（丢失后 survive 时长） |
| `track_thresh` | 0.5 | 高/低分检测分界 |
| `high_thresh` | 0.6 | 新轨迹初始化阈值 |
| `match_thresh` | 0.8 | IoU 匹配阈值 |

> 阈值语义：检测分数 ≥ `high_thresh` 才建立新轨迹；`track_thresh` 只用于丢失轨迹的找回关联。

## 5. 结果对象

`YOLOResult`：`x/y/width/height`（原图坐标，已截断）、`clsId`、`score`、`track_id`，
以及任务载荷：

- `keypoints`（POSE）
- `angle`（OBB）
- `mask`（SEGMENT，**ROI**：长度 `width*height`、0/1 二值、行跨距 `width`、
  原点 = 框左上角；默认按需生成）

## 6. 错误码

| 宏 | 值 | 含义 |
|---|---|---|
| `STATUS_SUCCESS` | 0 | 成功 |
| `STATUS_MODEL_NOT_FOUND` | -1 | 模型未找到 |
| `STATUS_MODEL_INIT_FAILED` | -2 | 模型初始化失败 |
| `STATUS_NOT_INITIALIZED` | -3 | 尚未初始化 |
| `STATUS_INFERENCE_FAILED` | -4 | 推理失败 |
| `STATUS_TASK_UNSUPPORTED` | -5 | 当前任务不支持该操作（如对非带框任务调用 `track()`） |

## 7. 各语言 SDK

| SDK | 入口 | 文档 |
|---|---|---|
| C++ | `fast_yolo/yolo.h` | 本文 |
| Java | `java/fast-yolo-jni/` | [JNI](../build/jni.md) |
| Python | `python/fastyolo/` | [python/README.md](../../python/README.md) |

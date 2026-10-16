"""配置类型：与 C++ yolo::YOLOConfig / yolo::InferDevice / yolo::TaskType 一一对应。"""
from __future__ import annotations

import enum
from dataclasses import dataclass


class InferDevice(enum.IntEnum):
    """推理设备（值必须与 C++ yolo::InferDevice 一致）。

    注意：具体后端对设备支持不同（如 OpenVINO 后端 VULKAN/CUDA 会回退 CPU），
    见 README 后端对照。
    """
    AUTO = 0
    CPU = 1
    NPU = 2
    VULKAN = 3
    OPENCL = 4
    CUDA = 6
    ALL = 99


class TaskType(enum.IntEnum):
    """任务类型（值必须与 C++ yolo::TaskType 一致）。

    AUTO 时按模型输出形状自动推断，大多数情况无需显式指定；
    歧义场景（如 v8 风格 OBB 与 DETECT）建议显式指定。
    """
    AUTO = 0
    DETECT = 1
    SEGMENT = 2
    POSE = 3
    OBB = 4
    CLASSIFY = 5
    SEMSEG = 6
    DEPTH = 7


@dataclass
class YoloConfig:
    """推理配置（默认值与 C++ yolo::YOLOConfig 一致）。"""
    min_score: float = 0.25          # 最低置信度
    iou_threshold: float = 0.45      # NMS IoU 阈值
    device: InferDevice = InferDevice.CPU
    num_threads: int = 1             # CPU 线程数（GPU/NPU 可忽略）
    gpu_id: int = 0                  # 多 GPU 时指定 ID
    precision: int = 32
    task: TaskType = TaskType.AUTO   # 任务类型（AUTO 自动推断）
    num_classes: int = 0             # POSE 等需要类别数（0 = 自动推断）
    num_keypoints: int = 17          # POSE 关键点数（默认 COCO 17 点）
    top_k: int = 5                   # CLASSIFY 返回前 K 个类别
    input_name: str = "images"       # ONNX 输入节点名
    output_name: str = ""            # 输出节点名（空 = 使用全部输出）
    # ---- ByteTrack 跟踪器参数（仅 track() 使用）----
    track_frame_rate: int = 30       # 视频帧率，与 track_buffer 一起决定目标丢失存活秒数
    track_buffer: int = 30           # 丢失缓冲帧数（30fps 下 = 1s）
    track_thresh: float = 0.5        # 低阈值，ByteTrack 第二轮 IoU 关联
    high_thresh: float = 0.6         # 高阈值，第一轮关联
    match_thresh: float = 0.8        # IoU 匹配阈值，越高越不易串 ID

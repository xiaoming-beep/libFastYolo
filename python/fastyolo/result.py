"""推理结果类型：detect / seg / pose / obb / cls / semseg / depth 各任务对应。

坐标/掩码/类别图/深度图均已反算回**原图**空间（与 C++ 库约定一致）：
  - 框：原图整数像素（xywh）
  - 掩码：原图分辨率二值 uint8 数组
  - 类别图：原图分辨率 int32 数组
  - 深度图：原图分辨率 float32 数组（模型原始深度值）
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Detection:
    """单个检测结果。

    坐标为原图空间整数像素；track_id 为 ByteTrack 跟踪 id，-1 表示未跟踪
    （predict() 结果恒为 -1，track() 结果 >= 0）。
    """
    x: int
    y: int
    width: int
    height: int
    cls_id: int
    score: float
    track_id: int = -1

    def to_xyxy(self) -> tuple[int, int, int, int]:
        """(x1, y1, x2, y2) 格式（x2/y2 为开区间右边界，同 OpenCV rectangle 约定）。"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def to_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
            "cls_id": self.cls_id, "score": self.score, "track_id": self.track_id,
        }


@dataclass(frozen=True)
class Keypoint:
    """关键点（原图坐标）。"""
    x: float
    y: float
    score: float


@dataclass(frozen=True)
class SegResult:
    """实例分割结果：检测框 + ROI 掩码。

    mask：框内二值掩码 uint8 [框高, 框宽]（0/1），局部坐标原点 = 框左上角；
    仅在 predict(need_masks=True) 时生成；track/默认 predict 为 None。
    """
    box: Detection
    mask: np.ndarray | None = field(repr=False)  # uint8 [h, w] 0/1 | None

    @property
    def cls_id(self) -> int:
        return self.box.cls_id

    @property
    def score(self) -> float:
        return self.box.score


@dataclass(frozen=True)
class PoseResult:
    """姿态估计结果：检测框 + 关键点列表（原图坐标）。"""
    box: Detection
    keypoints: tuple[Keypoint, ...]

    @property
    def cls_id(self) -> int:
        return self.box.cls_id

    @property
    def score(self) -> float:
        return self.box.score


@dataclass(frozen=True)
class ObbResult:
    """旋转目标检测结果：AABB 框 + 旋转角（弧度）。"""
    box: Detection
    angle: float  # 弧度

    @property
    def cls_id(self) -> int:
        return self.box.cls_id

    @property
    def score(self) -> float:
        return self.box.score

    def corners(self) -> np.ndarray:
        """旋转框 4 角点（原图坐标，float32 [4, 2]），便于绘制/计算 rIoU。"""
        cx = self.box.x + self.box.width / 2.0
        cy = self.box.y + self.box.height / 2.0
        c, s = np.cos(self.angle), np.sin(self.angle)
        hw, hh = self.box.width / 2.0, self.box.height / 2.0
        local = np.array([[hw, hh], [-hw, hh], [-hw, -hh], [hw, -hh]], dtype=np.float32)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        return (local @ rot.T) + np.array([cx, cy], dtype=np.float32)


@dataclass(frozen=True)
class ClassifyResult:
    """图像分类结果。"""
    top_class: int          # 置信度最高类别
    top_score: float
    top_k: int
    top_ids: tuple[int, ...]      # 前 top_k 类别 id（分数降序）
    top_scores: tuple[float, ...]
    scores: np.ndarray = field(repr=False)  # 全部分类分数（softmax 后，float32）


@dataclass(frozen=True)
class SemSegResult:
    """语义分割结果：原图分辨率逐像素类别图。"""
    class_map: np.ndarray  # int32 [imgH, imgW]
    width: int
    height: int


@dataclass(frozen=True)
class DepthResult:
    """深度估计结果：原图分辨率 float32 深度图（模型原始值）+ 值域。"""
    depth_map: np.ndarray  # float32 [imgH, imgW]
    depth_min: float
    depth_max: float
    width: int
    height: int

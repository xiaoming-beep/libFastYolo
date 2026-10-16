"""fastyolo —— libFastYolo Python SDK。

高层入口 FastYolo 提供多任务推理（detect / seg / pose / obb / cls / semseg / depth）
+ ByteTrack 跟踪；底层还暴露 Letterbox / YoloConfig / 各任务结果类型。

用法::

    from fastyolo import FastYolo, InferDevice, TaskType, Detection

    yolo = FastYolo("yolo11n.onnx", device=InferDevice.CPU, num_threads=4)
    dets: list[Detection] = yolo.predict(bgr_img)   # np.ndarray HxWx3 uint8 BGR
    dets = yolo.track(bgr_img)                      # 带 track_id（ByteTrack，仅 DETECT）

    seg = FastYolo("yolo11n-seg.onnx", task=TaskType.SEGMENT)   # 返回 list[SegResult]
    cls = FastYolo("yolo11n-cls.onnx")                          # 返回 ClassifyResult
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np

from . import _binding
from ._binding import YoloError
from .config import InferDevice, TaskType, YoloConfig
from .letterbox import Letterbox
from .result import (
    ClassifyResult,
    DepthResult,
    Detection,
    Keypoint,
    ObbResult,
    PoseResult,
    SegResult,
    SemSegResult,
)

__version__ = "0.1.0"

__all__ = [
    "FastYolo", "Detection", "SegResult", "PoseResult", "ObbResult",
    "ClassifyResult", "SemSegResult", "DepthResult", "Keypoint",
    "InferDevice", "TaskType", "YoloConfig", "YoloError", "Letterbox",
    "__version__",
]


class FastYolo:
    """libFastYolo 高层入口：多任务推理 + 跟踪。

    约定（与 C++ 库一致）：
      - 输入图像：np.ndarray HxWx3 uint8 **BGR**（OpenCV 读图即为 BGR，不做色彩转换）
      - 模型：ONNX 单文件字节流；OpenVINO 后端不支持 IR .xml+.bin 双文件
      - 实例非线程安全：多线程请每线程一个实例或外部加锁（同 Java SDK）
      - predict() 按任务返回类型化结果：
          DETECT → list[Detection]；SEGMENT → list[SegResult]；
          POSE → list[PoseResult]；OBB → list[ObbResult]；
          CLASSIFY → ClassifyResult；SEMSEG → SemSegResult；DEPTH → DepthResult
      - 所有坐标/掩码/类别图/深度图均反算回原图分辨率

    参数:
        model: 模型路径或模型字节流
        config: YoloConfig；传入后可用同名关键字参数覆盖个别字段
        box_size: letterbox 目标边长（None 时按模型输入尺寸自动取，如 640/224/1024/768）
        letterbox_color: 填充灰值 0~255（默认 114）
        class_names: 可选类别名列表（cls_id -> 名字），用于 label 显示
    """

    def __init__(self, model, *, config: YoloConfig | None = None,
                 box_size: int | None = None, letterbox_color: float = 114.0,
                 class_names: list[str] | None = None, **overrides):
        cfg = YoloConfig(**overrides) if config is None else replace(config, **overrides)
        self._cfg = cfg
        self.class_names = class_names
        self.input_shape: list[int] = []
        self.output_shape: list[int] = []
        self.output_shapes: list[tuple[list[int], int]] = []
        self.task: TaskType | None = None   # init 后为解析出的任务类型

        model_data = self._read_model(model)
        self._handle = _binding.create(model_data, cfg)
        self.input_shape, self.output_shape = _binding.init(self._handle)
        self.output_shapes = _binding.output_info(self._handle)
        self.task = TaskType(_binding.task(self._handle))
        # box_size 未指定时按模型输入尺寸自动取（NCHW 的 H/W，模型输入为正方形）
        if box_size is None:
            box_size = int(self.input_shape[3]) if len(self.input_shape) >= 4 else 640
        # 分类任务用中心裁剪预处理（与 ultralytics Resize+CenterCrop 一致），其余用填充
        mode = "crop" if self.task == TaskType.CLASSIFY else "pad"
        self._letterbox = Letterbox(box_size, letterbox_color, mode=mode)

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #

    def predict(self, image: np.ndarray, score_threshold: float = -1.0,
                iou_threshold: float = -1.0, need_masks: bool = False):
        """统一推理入口：按任务类型返回对应结果（见类文档）。

        need_masks=True 时 SEGMENT 对全部保留框生成 ROI 掩码（每框
        [框高, 框宽] uint8 0/1，局部坐标原点 = 框左上角）；默认只出框。
        """
        self._check_image(image)
        self._letterbox.process(image)
        raw = _binding.predict(self._handle, self._letterbox._handle,
                               score_threshold, iou_threshold, need_masks)
        return self._to_result(raw)

    def classify(self, image: np.ndarray) -> ClassifyResult:
        """图像分类（仅 CLASSIFY 任务）。"""
        result = self.predict(image)
        if not isinstance(result, ClassifyResult):
            raise YoloError(-5, "model task is not CLASSIFY")
        return result

    def semseg(self, image: np.ndarray) -> SemSegResult:
        """语义分割（仅 SEMSEG 任务）。"""
        result = self.predict(image)
        if not isinstance(result, SemSegResult):
            raise YoloError(-5, "model task is not SEMSEG")
        return result

    def depth(self, image: np.ndarray) -> DepthResult:
        """深度估计（仅 DEPTH 任务）。"""
        result = self.predict(image)
        if not isinstance(result, DepthResult):
            raise YoloError(-5, "model task is not DEPTH")
        return result

    def track(self, image: np.ndarray, score_threshold: float = -1.0,
              iou_threshold: float = -1.0):
        """检测 + ByteTrack 多目标跟踪（结果带 track_id，需连续帧调用）。

        支持 DETECT（返回 list[Detection]，行为同 predict）以及 SEGMENT/POSE/OBB
        （返回带 track_id 的 SegResult/PoseResult/ObbResult，关键点/角度与框一致；
        注意 SEGMENT 跟踪只出框，mask 为 None——在线按需掩码请走 JNI track(budgets)）；
        稠密任务（CLASSIFY/SEMSEG/DEPTH）不支持跟踪，抛 YoloError(-5)。
        """
        self._check_image(image)
        self._letterbox.process(image)
        raw = _binding.track(self._handle, self._letterbox._handle,
                             score_threshold, iou_threshold)
        return self._to_result(raw)

    def label_of(self, det: Detection) -> str:
        """类别名（class_names 未提供时返回 "class<N>"）。"""
        if self.class_names and 0 <= det.cls_id < len(self.class_names):
            return self.class_names[det.cls_id]
        return "class%d" % det.cls_id

    # ------------------------------------------------------------------ #
    # 结果转换
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_result(raw: _binding.RawResult):
        task = raw.task
        if task == int(TaskType.DETECT):
            return [Detection(*b) for b in raw.boxes]
        if task == int(TaskType.SEGMENT):
            masks = raw.masks if raw.masks is not None else [None] * len(raw.boxes)
            return [SegResult(Detection(*raw.boxes[i]), masks[i])
                    for i in range(len(raw.boxes))]
        if task == int(TaskType.POSE):
            return [
                PoseResult(
                    Detection(*raw.boxes[i]),
                    tuple(Keypoint(*raw.keypoints[i, j]) for j in range(raw.keypoint_count)),
                )
                for i in range(len(raw.boxes))
            ]
        if task == int(TaskType.OBB):
            return [ObbResult(Detection(*raw.boxes[i]), float(raw.angles[i]))
                    for i in range(len(raw.boxes))]
        if task == int(TaskType.CLASSIFY):
            return ClassifyResult(
                top_class=int(raw.class_ids[0]),
                top_score=float(raw.class_scores[0]),
                top_k=raw.top_k,
                top_ids=tuple(int(x) for x in raw.class_ids),
                top_scores=tuple(float(x) for x in raw.class_scores),
                scores=raw.all_class_scores,
            )
        if task == int(TaskType.SEMSEG):
            return SemSegResult(class_map=raw.class_map, width=raw.map_w, height=raw.map_h)
        if task == int(TaskType.DEPTH):
            return DepthResult(depth_map=raw.depth_map, depth_min=raw.depth_min,
                               depth_max=raw.depth_max, width=raw.map_w, height=raw.map_h)
        raise YoloError(-5, "unknown task %d" % task)

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        if getattr(self, "_letterbox", None):
            self._letterbox.close()
            self._letterbox = None
        handle = getattr(self, "_handle", 0)
        if handle:
            _binding.destroy(handle)
            self._handle = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    @staticmethod
    def _read_model(model) -> bytes:
        if isinstance(model, (bytes, bytearray, memoryview)):
            return bytes(model)
        if isinstance(model, (str, os.PathLike)):
            with open(model, "rb") as f:
                return f.read()
        raise TypeError("model 必须是模型路径或 bytes/bytearray")

    @staticmethod
    def _check_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image 必须是 np.ndarray（HxWx3 uint8 BGR）")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image 必须是 HxWx3 uint8 BGR 图像")

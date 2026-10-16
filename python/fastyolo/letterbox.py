"""Letterbox 预处理（镜像 C++ yolo::Letterbox，数学严格一致）。

契约（与 C++ 侧一致）：
  scale = min(box_size / w, box_size / h)
  dst   = round(w * scale)，四舍五入
  灰度填充 color（默认 114），BGR(HWC) uint8 → NCHW float32 planar，归一化 /255

缩放/插值由本模块负责（C++ 库不做缩放，调用方自理），实现与
OpenCV INTER_LINEAR 同公式：src = (dst + 0.5) * scale - 0.5。
"""
from __future__ import annotations

import ctypes

import numpy as np

from . import _binding


def _resize_bilinear(src: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    """双线性缩放 HWC uint8（OpenCV INTER_LINEAR 同公式，结果 rint 后 clamp）。"""
    src_h, src_w = src.shape[:2]
    if src_w == dst_w and src_h == dst_h:
        return src
    scale_x = src_w / dst_w
    scale_y = src_h / dst_h
    x = (np.arange(dst_w, dtype=np.float64) + 0.5) * scale_x - 0.5
    y = (np.arange(dst_h, dtype=np.float64) + 0.5) * scale_y - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    x0 = np.clip(x0, 0, src_w - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    y0 = np.clip(y0, 0, src_h - 1)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    tl = src[y0][:, x0]
    tr = src[y0][:, x1]
    bl = src[y1][:, x0]
    br = src[y1][:, x1]
    fxw = fx[np.newaxis, :, np.newaxis]
    fyw = fy[:, np.newaxis, np.newaxis]
    top = tl * (1.0 - fxw) + tr * fxw
    bottom = bl * (1.0 - fxw) + br * fxw
    out = top * (1.0 - fyw) + bottom * fyw
    # 注意：广播运算产物可能非连续（strided），必须转连续并转 uint8，
    # 否则 fill_bgr 的 data 指针按连续内存传给 C++ 会读到错乱数据
    return np.ascontiguousarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


class Letterbox:
    """letterbox 预处理器：update（C++ 计算缩放）→ 缩放 → fill_bgr（C++ 零拷贝）。

    mode:
      - "pad"（默认）: 均匀缩放 + 灰边填充（检测/分割/深度等任务）
      - "crop": 按最短边缩放后中心裁剪 box_size×box_size（分类任务，无填充，
        与 ultralytics classify_transforms 的 Resize+CenterCrop 一致）
    """

    def __init__(self, box_size: int = 640, color: float = 114.0, mode: str = "pad"):
        if mode not in ("pad", "crop"):
            raise ValueError("mode 必须是 'pad' 或 'crop'")
        self.box_size = box_size
        self.color = color
        self.mode = mode
        self._handle = _binding.letterbox_create(box_size, color, center_crop=(mode == "crop"))
        self.img_width = 0
        self.img_height = 0
        self.dst_width = 0
        self.dst_height = 0

    def update(self, width: int, height: int) -> tuple[int, int]:
        """按原图尺寸计算缩放目标尺寸（与 C++ Letterbox::update 完全一致）。"""
        self.dst_width, self.dst_height = _binding.letterbox_update(
            self._handle, width, height)
        self.img_width, self.img_height = width, height
        return self.dst_width, self.dst_height

    def fill_bgr(self, image: np.ndarray) -> None:
        """填充像素。image 必须为 dst_width×dst_height×3 的 uint8 BGR（C++ 契约，不缩放）。"""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("fill_bgr 需要 HxWx3 uint8 BGR 图像")
        if image.shape[:2] != (self.dst_height, self.dst_width):
            raise ValueError(
                "图像尺寸 %dx%d 与 letterbox 目标 %dx%d 不一致，"
                "请先 update() 并缩放到目标尺寸" % (image.shape[1], image.shape[0],
                                               self.dst_width, self.dst_height))
        # C++ 契约要求连续内存（按行连续读取）；非连续数组先转连续
        if not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)
        ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        _binding.letterbox_fill_bgr(self._handle, ptr)

    def process(self, image: np.ndarray) -> None:
        """整条流水线：update(原图尺寸) → 缩放 → fill_bgr（零拷贝）。

        crop 模式：按最短边缩放到 dst_width×dst_height（一边 == box_size），
        由 C++ 侧中心裁剪后写入（等价 ultralytics 的 Resize + CenterCrop）。
        缩放用 PIL BILINEAR（与 ultralytics classify_transforms 一致，分类结果对齐）；
        无 PIL 时回退内置双线性。
        """
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("process 需要 HxWx3 uint8 BGR 图像")
        h, w = image.shape[:2]
        self.update(w, h)
        if self.mode == "crop":
            resized = self._resize_classify(image, self.dst_width, self.dst_height)
        else:
            resized = _resize_bilinear(image, self.dst_width, self.dst_height)
        self.fill_bgr(resized)

    @staticmethod
    def _resize_classify(image: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
        """分类任务缩放（PIL BILINEAR，与 ultralytics classify_transforms 对齐）。"""
        try:
            from PIL import Image as PILImage
            resized = np.array(
                PILImage.fromarray(image).resize((dst_w, dst_h), PILImage.BILINEAR),
                dtype=np.uint8,
            )
            return np.ascontiguousarray(resized)
        except ImportError:
            return _resize_bilinear(image, dst_w, dst_h)

    @property
    def pixels(self) -> np.ndarray:
        """像素缓冲视图（float32 planar RRR..GGG..BBB，零拷贝，仅测试/调试用）。"""
        arr = _binding.letterbox_pixels(self._handle)
        return np.frombuffer(arr, dtype=np.float32)

    def close(self) -> None:
        if self._handle:
            _binding.letterbox_destroy(self._handle)
            self._handle = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

"""ctypes 绑定层：封装 fastyolo_c.h 的纯 C 标量 API。

错误码（与 C++ yolo.h 一致）：0 成功；-1 模型未找到；-2 模型初始化失败；
-3 未初始化；-4 推理失败；-5 任务不支持。
"""
from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_char_p, c_float, c_int, c_int64, c_size_t, c_uint8, c_void_p

import numpy as np

from ._loader import load_library
from .config import YoloConfig

_STATUS_MESSAGES = {
    0: "success",
    -1: "model not found",
    -2: "model init failed",
    -3: "inference not initialized",
    -4: "inference failed",
    -5: "task not supported",
}


class YoloError(RuntimeError):
    """libFastYolo 错误；code 为 STATUS_* 错误码（0 以外的负整数）。"""

    def __init__(self, code: int, message: str | None = None):
        self.code = code
        super().__init__(message or _STATUS_MESSAGES.get(code, "unknown error code %d" % code))


def check(code: int) -> None:
    if code != 0:
        raise YoloError(code)


class FastyoloBox(ctypes.Structure):
    _fields_ = [
        ("x", c_int),
        ("y", c_int),
        ("width", c_int),
        ("height", c_int),
        ("cls_id", c_int),
        ("score", c_float),
        ("track_id", c_int),
    ]


class FastyoloKeypoint(ctypes.Structure):
    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("score", c_float),
    ]


class FastyoloResult(ctypes.Structure):
    _fields_ = [
        ("task", c_int),
        ("boxes", POINTER(FastyoloBox)),
        ("box_count", c_int),
        ("keypoints", POINTER(FastyoloKeypoint)),
        ("keypoint_count", c_int),
        ("masks", POINTER(c_uint8)),   # SEGMENT: ROI 掩码按框拼接（每框长度 = 框 w*h）
        ("angles", POINTER(c_float)),
        ("class_ids", POINTER(c_int)),
        ("class_scores", POINTER(c_float)),
        ("top_k", c_int),
        ("all_class_scores", POINTER(c_float)),
        ("class_count", c_int),
        ("class_map", POINTER(c_uint8)),
        ("depth_map", POINTER(c_float)),
        ("depth_min", c_float),
        ("depth_max", c_float),
        ("map_w", c_int),
        ("map_h", c_int),
    ]


_lib = load_library()

_lib.fastyolo_create.restype = c_void_p
_lib.fastyolo_create.argtypes = [
    POINTER(c_uint8), c_size_t,
    c_float, c_float,
    c_int, c_int, c_int, c_int,
    c_char_p, c_char_p,
    c_int, c_int,
    c_float, c_float, c_float,
    c_int, c_int, c_int, c_int,
    POINTER(c_int),
]

_lib.fastyolo_init.restype = c_int
_lib.fastyolo_init.argtypes = [
    c_void_p,
    POINTER(c_int64), c_int, POINTER(c_int),
    POINTER(c_int64), c_int, POINTER(c_int),
]

_lib.fastyolo_output_info.restype = c_int
_lib.fastyolo_output_info.argtypes = [
    c_void_p,
    POINTER(c_int64), c_int,
    POINTER(c_int), POINTER(c_int), c_int,
    POINTER(c_int),
]

_lib.fastyolo_task.restype = c_int
_lib.fastyolo_task.argtypes = [c_void_p]

_lib.fastyolo_letterbox_create.restype = c_void_p
_lib.fastyolo_letterbox_create.argtypes = [c_int, c_float, c_int]

_lib.fastyolo_letterbox_update.restype = None
_lib.fastyolo_letterbox_update.argtypes = [c_void_p, c_int, c_int, POINTER(c_int), POINTER(c_int)]

_lib.fastyolo_letterbox_fill_bgr.restype = None
_lib.fastyolo_letterbox_fill_bgr.argtypes = [c_void_p, POINTER(c_uint8)]

_lib.fastyolo_letterbox_pixels.restype = POINTER(c_float)
_lib.fastyolo_letterbox_pixels.argtypes = [c_void_p, POINTER(c_int)]

_lib.fastyolo_predict.restype = POINTER(FastyoloResult)
_lib.fastyolo_predict.argtypes = [c_void_p, c_void_p, c_float, c_float, c_int, POINTER(c_int)]

_lib.fastyolo_track.restype = POINTER(FastyoloResult)
_lib.fastyolo_track.argtypes = [c_void_p, c_void_p, c_float, c_float, POINTER(c_int)]

_lib.fastyolo_free_result.restype = None
_lib.fastyolo_free_result.argtypes = [POINTER(FastyoloResult)]

_lib.fastyolo_letterbox_destroy.restype = None
_lib.fastyolo_letterbox_destroy.argtypes = [c_void_p]

_lib.fastyolo_destroy.restype = None
_lib.fastyolo_destroy.argtypes = [c_void_p]

_SHAPE_CAP = 8  # 维度数上限（实际模型 ≤ 4）


def create(model_data: bytes, config: YoloConfig) -> int:
    """创建推理引擎，返回不透明句柄（int）。"""
    code = c_int()
    buf = (c_uint8 * len(model_data)).from_buffer_copy(model_data)
    handle = _lib.fastyolo_create(
        buf, len(model_data),
        config.min_score, config.iou_threshold,
        int(config.device), config.num_threads, config.gpu_id, config.precision,
        config.input_name.encode("utf-8"), config.output_name.encode("utf-8"),
        config.track_frame_rate, config.track_buffer,
        config.track_thresh, config.high_thresh, config.match_thresh,
        int(config.task), config.num_classes, config.num_keypoints, config.top_k,
        byref(code),
    )
    check(code.value)
    if not handle:
        raise YoloError(-2, "failed to create inference engine")
    return int(handle)


def init(handle: int) -> tuple[list[int], list[int]]:
    """初始化后端引擎，返回 (input_shape, 首个 output_shape)。"""
    in_shape = (c_int64 * _SHAPE_CAP)()
    out_shape = (c_int64 * _SHAPE_CAP)()
    in_rank = c_int()
    out_rank = c_int()
    code = _lib.fastyolo_init(c_void_p(handle), in_shape, _SHAPE_CAP, byref(in_rank),
                              out_shape, _SHAPE_CAP, byref(out_rank))
    check(code)
    return (list(in_shape[:in_rank.value]), list(out_shape[:out_rank.value]))


def output_info(handle: int) -> list[tuple[list[int], int]]:
    """查询全部输出，返回 [(shape, elem_type), ...]（elem_type: 1=float32, 7=int64）。"""
    ranks = (c_int * 4)()
    dtypes = (c_int * 4)()
    count = c_int()
    shapes = (c_int64 * 64)()
    code = _lib.fastyolo_output_info(c_void_p(handle), shapes, 64,
                                     ranks, dtypes, 4, byref(count))
    check(code)
    result = []
    offset = 0
    for i in range(count.value):
        rank = ranks[i]
        result.append((list(shapes[offset:offset + rank]), dtypes[i]))
        offset += rank
    return result


def task(handle: int) -> int:
    """查询解析后的任务类型（FastyoloTask 值，AUTO 时返回推断结果）。"""
    return _lib.fastyolo_task(c_void_p(handle))


def letterbox_create(box_size: int, color: float, center_crop: bool = False) -> int:
    handle = _lib.fastyolo_letterbox_create(box_size, color, 1 if center_crop else 0)
    if not handle:
        raise YoloError(-2, "failed to create letterbox")
    return int(handle)


def letterbox_update(handle: int, width: int, height: int) -> tuple[int, int]:
    dst_w = c_int()
    dst_h = c_int()
    _lib.fastyolo_letterbox_update(c_void_p(handle), width, height, byref(dst_w), byref(dst_h))
    return dst_w.value, dst_h.value


def letterbox_fill_bgr(handle: int, bgr_ptr) -> None:
    """bgr_ptr: POINTER(c_uint8)，指向 dst_width*dst_height*3 字节 BGR（零拷贝）。"""
    _lib.fastyolo_letterbox_fill_bgr(c_void_p(handle), bgr_ptr)


def letterbox_pixels(handle: int) -> ctypes.Array:
    """读取 letterbox 像素缓冲（float planar RRR..GGG..BBB），返回 ctypes 数组视图（不拷贝）。"""
    length = c_int()
    ptr = _lib.fastyolo_letterbox_pixels(c_void_p(handle), byref(length))
    if not ptr or length.value == 0:
        return (c_float * 0)()
    return (c_float * length.value).from_address(ctypes.addressof(ptr.contents))


class RawResult:
    """FastyoloResult 的 Python 视图：各字段已转为 numpy 数组（拷贝，可安全持有）。"""

    __slots__ = (
        "task", "boxes", "keypoints", "keypoint_count",
        "masks", "angles",
        "class_ids", "class_scores", "top_k", "all_class_scores", "class_count",
        "class_map", "depth_map", "depth_min", "depth_max", "map_w", "map_h",
    )

    def __init__(self, task: int, boxes, keypoints, keypoint_count: int,
                 masks, angles,
                 class_ids, class_scores, top_k: int, all_class_scores, class_count: int,
                 class_map, depth_map, depth_min: float, depth_max: float,
                 map_w: int, map_h: int):
        self.task = task
        self.boxes = boxes          # list[tuple] (x, y, w, h, cls_id, score, track_id)
        self.keypoints = keypoints  # np.ndarray [n, k, 3] | None
        self.keypoint_count = keypoint_count
        self.masks = masks          # list[np.ndarray(h_i, w_i) uint8 0/1] | None（每框 ROI）
        self.angles = angles        # np.ndarray [n] | None
        self.class_ids = class_ids
        self.class_scores = class_scores
        self.top_k = top_k
        self.all_class_scores = all_class_scores
        self.class_count = class_count
        self.class_map = class_map  # np.ndarray [H, W] int32 | None
        self.depth_map = depth_map  # np.ndarray [H, W] float32 | None
        self.depth_min, self.depth_max = depth_min, depth_max
        self.map_w, self.map_h = map_w, map_h


def _read_result(ptr) -> RawResult:
    """将 C 侧 FastyoloResult 拷贝为 RawResult 并释放 C 侧内存。"""
    r = ptr.contents
    try:
        n = r.box_count
        boxes = [
            (r.boxes[i].x, r.boxes[i].y, r.boxes[i].width, r.boxes[i].height,
             r.boxes[i].cls_id, r.boxes[i].score, r.boxes[i].track_id)
            for i in range(n)
        ] if n and r.boxes else []

        keypoints = None
        if r.task == 3 and n and r.keypoints and r.keypoint_count > 0:
            arr = np.ctypeslib.as_array(r.keypoints, shape=(n * r.keypoint_count,))
            keypoints = np.array(arr, dtype=np.float32, copy=True).reshape(n, r.keypoint_count, 3)

        # SEGMENT：ROI 掩码按框顺序拼接，每框长度 = boxes[i].w*h，按 (h_i, w_i) 还原
        masks = None
        if r.task == 2 and n and r.masks:
            sizes = [b[2] * b[3] for b in boxes]
            total = sum(sizes)
            arr = np.array(np.ctypeslib.as_array(r.masks, shape=(total,)),
                           dtype=np.uint8, copy=True)
            masks, off = [], 0
            for (_, _, w, h, *_), sz in zip(boxes, sizes):
                masks.append(arr[off:off + sz].reshape(h, w))
                off += sz

        angles = None
        if r.task == 4 and n and r.angles:
            angles = np.array(np.ctypeslib.as_array(r.angles, shape=(n,)),
                              dtype=np.float32, copy=True)

        class_ids = np.array(np.ctypeslib.as_array(r.class_ids, shape=(r.top_k,)),
                             dtype=np.int32, copy=True) if r.top_k and r.class_ids else None
        class_scores = np.array(np.ctypeslib.as_array(r.class_scores, shape=(r.top_k,)),
                                dtype=np.float32, copy=True) if r.top_k and r.class_scores else None
        all_class_scores = np.array(
            np.ctypeslib.as_array(r.all_class_scores, shape=(r.class_count,)),
            dtype=np.float32, copy=True) if r.class_count and r.all_class_scores else None

        class_map = None
        if r.task == 6 and r.class_map and r.map_w > 0 and r.map_h > 0:
            arr = np.ctypeslib.as_array(r.class_map, shape=(r.map_w * r.map_h,))
            class_map = np.array(arr, dtype=np.int32, copy=True).reshape(r.map_h, r.map_w)

        depth_map = None
        if r.task == 7 and r.depth_map and r.map_w > 0 and r.map_h > 0:
            arr = np.ctypeslib.as_array(r.depth_map, shape=(r.map_w * r.map_h,))
            depth_map = np.array(arr, dtype=np.float32, copy=True).reshape(r.map_h, r.map_w)

        return RawResult(
            task=r.task, boxes=boxes, keypoints=keypoints, keypoint_count=r.keypoint_count,
            masks=masks, angles=angles,
            class_ids=class_ids, class_scores=class_scores, top_k=r.top_k,
            all_class_scores=all_class_scores, class_count=r.class_count,
            class_map=class_map, depth_map=depth_map,
            depth_min=r.depth_min, depth_max=r.depth_max,
            map_w=r.map_w, map_h=r.map_h,
        )
    finally:
        _lib.fastyolo_free_result(ptr)


def _run(which, handle: int, lb_handle: int, score_threshold: float, iou_threshold: float,
         need_masks: bool = False) -> RawResult:
    code = c_int()
    if which == "predict":
        ptr = _lib.fastyolo_predict(c_void_p(handle), c_void_p(lb_handle),
                                    score_threshold, iou_threshold,
                                    c_int(1 if need_masks else 0), byref(code))
    else:
        ptr = _lib.fastyolo_track(c_void_p(handle), c_void_p(lb_handle),
                                  score_threshold, iou_threshold, byref(code))
    check(code.value)
    if not ptr:
        return RawResult(task=0, boxes=[], keypoints=None, keypoint_count=0,
                         masks=None, angles=None,
                         class_ids=None, class_scores=None, top_k=0,
                         all_class_scores=None, class_count=0,
                         class_map=None, depth_map=None, depth_min=0.0, depth_max=0.0,
                         map_w=0, map_h=0)
    return _read_result(ptr)


def predict(handle: int, lb_handle: int, score_threshold: float, iou_threshold: float,
            need_masks: bool = False) -> RawResult:
    return _run("predict", handle, lb_handle, score_threshold, iou_threshold, need_masks)


def track(handle: int, lb_handle: int, score_threshold: float, iou_threshold: float) -> RawResult:
    return _run("track", handle, lb_handle, score_threshold, iou_threshold)


def letterbox_destroy(handle: int) -> None:
    _lib.fastyolo_letterbox_destroy(c_void_p(handle))


def destroy(handle: int) -> None:
    _lib.fastyolo_destroy(c_void_p(handle))

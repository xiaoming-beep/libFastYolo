"""端到端冒烟测试（需要真实 ONNX 模型）。

通过 FASTYOLO_MODEL 环境变量指定模型路径；未提供时自动在常见位置查找，
找不到则跳过（skip）。
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from fastyolo import Detection, FastYolo, InferDevice, YoloError

_MODEL_CANDIDATES = [
    "yolo11n.onnx",
    "models/yolo11n.onnx",
    "model.onnx",
    "best.onnx",
]


def _find_model() -> str | None:
    env = os.environ.get("FASTYOLO_MODEL")
    if env and os.path.exists(env):
        return env
    for c in _MODEL_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def _model_or_skip() -> str:
    model = _find_model()
    if model is None:
        pytest.skip("未找到 ONNX 模型；设置 FASTYOLO_MODEL 环境变量后重跑")
    return model


def _gray_image(w=640, h=480) -> np.ndarray:
    return np.full((h, w, 3), 128, np.uint8)


def test_predict_smoke():
    model = _model_or_skip()
    yolo = FastYolo(model, device=InferDevice.CPU, num_threads=2)
    try:
        assert yolo.input_shape[0] == 1 and yolo.input_shape[1] == 3
        dets = yolo.predict(_gray_image())
        assert isinstance(dets, list)
        for d in dets:
            assert isinstance(d, Detection)
            assert d.width > 0 and d.height > 0
            assert d.x >= 0 and d.y >= 0
            assert 0.0 <= d.score <= 1.0
            assert d.track_id == -1
    finally:
        yolo.close()


def test_track_smoke():
    model = _model_or_skip()
    yolo = FastYolo(model, device=InferDevice.CPU, num_threads=2)
    try:
        # 连续两帧调用，跟踪器惰性创建，track_id 应 >= 0
        for _ in range(2):
            dets = yolo.track(_gray_image())
            for d in dets:
                assert d.track_id >= 0
    finally:
        yolo.close()


def test_threshold_override():
    model = _model_or_skip()
    yolo = FastYolo(model, device=InferDevice.CPU, min_score=0.99)
    try:
        low = yolo.predict(_gray_image(), score_threshold=0.01)
        high = yolo.predict(_gray_image(), score_threshold=0.99)
        assert len(high) <= len(low)
        assert all(d.score >= 0.99 for d in high)
    finally:
        yolo.close()


def test_invalid_model_raises():
    with pytest.raises(YoloError):
        FastYolo(b"this is not an onnx model")


def test_bad_image_raises():
    model = _model_or_skip()
    yolo = FastYolo(model)
    try:
        with pytest.raises(ValueError):
            yolo.predict(np.zeros((10, 10), np.uint8))  # 缺通道
        with pytest.raises(ValueError):
            yolo.predict(np.zeros((10, 10, 3), np.float32))  # dtype 错误
        with pytest.raises(TypeError):
            yolo.predict("not an image")
    finally:
        yolo.close()


def test_model_bytes_input():
    model = _model_or_skip()
    with open(model, "rb") as f:
        data = f.read()
    yolo = FastYolo(data)  # 字节流输入
    try:
        yolo.predict(_gray_image())
    finally:
        yolo.close()


def test_context_manager_and_close_idempotent():
    model = _model_or_skip()
    with FastYolo(model) as yolo:
        yolo.predict(_gray_image())
        yolo.close()  # 重复 close 不应报错

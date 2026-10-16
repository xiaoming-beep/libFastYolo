"""Letterbox 数学与填充契约测试（与 C++ 实现逐位对齐）。

不依赖模型，直接通过 C API 验证 C++ 侧 yolo::Letterbox 的行为，
并验证 Python 端封装（fastyolo.Letterbox）与之一致。
"""
from __future__ import annotations

import numpy as np
import pytest

from fastyolo import _binding
from fastyolo.letterbox import Letterbox, _resize_bilinear

GRAY = np.float32(114.0) / np.float32(255.0)


# --------------------------------------------------------------------------- #
# C++ Letterbox::update 数学
# --------------------------------------------------------------------------- #

def test_update_math_cases():
    lb = _binding.letterbox_create(640, 114.0)
    try:
        # 等比缩放：scale = min(box/w, box/h)，dst = round(原边 * scale)
        assert _binding.letterbox_update(lb, 1280, 720) == (640, 360)   # scale=0.5
        assert _binding.letterbox_update(lb, 720, 1280) == (360, 640)
        # 非整倍：scale=min(0.64, 0.8)=0.64 → round(800*0.64)=512
        assert _binding.letterbox_update(lb, 1000, 800) == (640, 512)
        # 小图放大
        assert _binding.letterbox_update(lb, 320, 240) == (640, 480)
        # 正方形
        assert _binding.letterbox_update(lb, 640, 640) == (640, 640)
    finally:
        _binding.letterbox_destroy(lb)


# --------------------------------------------------------------------------- #
# C++ fill_bgr：BGR(HWC) uint8 → planar float，/255 归一化，灰边 114
# --------------------------------------------------------------------------- #

def _pixels(lb) -> np.ndarray:
    arr = _binding.letterbox_pixels(lb)
    return np.frombuffer(arr, dtype=np.float32)


def test_fill_full_no_padding():
    """原图 2x2、box=4：scale=2 → dst=4x4 全填充，无灰边。"""
    lb = _binding.letterbox_create(4, 114.0)
    try:
        assert _binding.letterbox_update(lb, 2, 2) == (4, 4)
        bgr = np.zeros((4, 4, 3), np.uint8)
        bgr[..., 0] = 255  # B=255, G=0, R=0
        _binding.letterbox_fill_bgr(lb, bgr.ctypes.data_as(
            np.ctypeslib.ctypes.POINTER(np.ctypeslib.ctypes.c_uint8)))
        pix = _pixels(lb)
        assert pix.shape == (4 * 4 * 3,)
        assert np.allclose(pix[:16], 0.0)         # R 平面 = 0/255
        assert np.allclose(pix[16:32], 0.0)       # G 平面 = 0/255
        assert np.allclose(pix[32:48], 1.0)       # B 平面 = 255/255
    finally:
        _binding.letterbox_destroy(lb)


def test_fill_vertical_padding():
    """原图 4x2、box=8：scale=2 → dst=8x4，上下各 2 行灰边（vertical_fill_bgr）。"""
    lb = _binding.letterbox_create(8, 114.0)
    try:
        assert _binding.letterbox_update(lb, 4, 2) == (8, 4)
        # fill_bgr 需要 dst_width x dst_height 尺寸的缓冲（即缩放后的图像）
        bgr = np.zeros((4, 8, 3), np.uint8)  # H=dst_height, W=dst_width
        bgr[..., 0] = 255
        _binding.letterbox_fill_bgr(lb, bgr.ctypes.data_as(
            np.ctypeslib.ctypes.POINTER(np.ctypeslib.ctypes.c_uint8)))
        pix = _pixels(lb)
        assert pix.shape == (8 * 8 * 3,)
        b_plane = pix[2 * 64:3 * 64].reshape(8, 8)   # B 平面
        assert np.allclose(b_plane[:2], GRAY)        # 顶部灰边（2 行）
        assert np.allclose(b_plane[2:6], 1.0)        # 图像区
        assert np.allclose(b_plane[6:], GRAY)        # 底部灰边（2 行）
        r_plane = pix[:64].reshape(8, 8)             # R 平面
        assert np.allclose(r_plane[2:6], 0.0)
        assert np.allclose(r_plane[:2], GRAY)
    finally:
        _binding.letterbox_destroy(lb)


def test_fill_horizontal_padding():
    """原图 2x4、box=8：scale=2 → dst=4x8，左右各 2 列灰边（horizontal_fill_bgr）。"""
    lb = _binding.letterbox_create(8, 114.0)
    try:
        assert _binding.letterbox_update(lb, 2, 4) == (4, 8)
        bgr = np.zeros((8, 4, 3), np.uint8)  # H=dst_height, W=dst_width
        bgr[..., 0] = 255
        _binding.letterbox_fill_bgr(lb, bgr.ctypes.data_as(
            np.ctypeslib.ctypes.POINTER(np.ctypeslib.ctypes.c_uint8)))
        pix = _pixels(lb)
        b_plane = pix[2 * 64:3 * 64].reshape(8, 8)
        assert np.allclose(b_plane[:, :2], GRAY)
        assert np.allclose(b_plane[:, 2:6], 1.0)
        assert np.allclose(b_plane[:, 6:], GRAY)
    finally:
        _binding.letterbox_destroy(lb)


# --------------------------------------------------------------------------- #
# Python 端封装 fastyolo.Letterbox（update + 缩放 + fill 全流程）
# --------------------------------------------------------------------------- #

def test_letterbox_class_process_uniform():
    """均匀图：process 后所有像素 = 128/255（无灰边场景）。"""
    lb = Letterbox(box_size=4, color=114.0)
    try:
        img = np.full((2, 2, 3), 128, np.uint8)
        lb.process(img)
        assert (lb.img_width, lb.img_height) == (2, 2)
        assert (lb.dst_width, lb.dst_height) == (4, 4)
        assert np.allclose(lb.pixels, np.float32(128.0) / 255.0)
    finally:
        lb.close()


def test_letterbox_class_gray_pad():
    """非均匀图：灰边与图像区并存（box=8，原图 4x2）。"""
    lb = Letterbox(box_size=8, color=114.0)
    try:
        img = np.full((2, 4, 3), 128, np.uint8)
        lb.process(img)
        assert (lb.dst_width, lb.dst_height) == (8, 4)
        b_plane = lb.pixels[2 * 64:3 * 64].reshape(8, 8)
        assert np.allclose(b_plane[:2], GRAY)
        assert np.allclose(b_plane[2:6], np.float32(128.0) / 255.0)
        assert np.allclose(b_plane[6:], GRAY)
    finally:
        lb.close()


def test_resize_bilinear_identity_and_shape():
    src = np.random.randint(0, 256, (100, 200, 3), np.uint8)
    same = _resize_bilinear(src, 200, 100)
    assert same is src or np.array_equal(same, src)
    small = _resize_bilinear(src, 100, 50)
    assert small.shape == (50, 100, 3)
    assert small.dtype == np.uint8
    # 插值结果应在原像素值范围内
    assert small.min() >= src.min() and small.max() <= src.max()


def test_fill_bgr_size_mismatch_raises():
    lb = Letterbox(box_size=4, color=114.0)
    try:
        lb.update(2, 2)  # dst = 4x4
        with pytest.raises(ValueError):
            lb.fill_bgr(np.zeros((3, 3, 3), np.uint8))
    finally:
        lb.close()

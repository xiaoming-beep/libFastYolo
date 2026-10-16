"""fastyolo 示例：单图检测（可选跟踪 + 画框保存）。

用法（在 python/ 目录下）:
    python examples/demo.py --model 模型.onnx --image 图片.jpg
    python examples/demo.py --model 模型.onnx --image 图片.jpg --track --output out.jpg

读图优先用 OpenCV（需 pip install opencv-python），否则用 Pillow（pip install pillow）；
两者都没有时仅支持纯 numpy 生成图像。画框需要 OpenCV。
"""
from __future__ import annotations

import argparse
import os
import sys

# 允许从任意目录直接运行：python examples/demo.py ...
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fastyolo import FastYolo, InferDevice

# COCO 80 类（YOLO 系列预训练模型默认类别）
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

try:
    import cv2
except ImportError:
    cv2 = None


def load_image(path: str) -> np.ndarray:
    if cv2 is not None:
        # imdecode 支持中文/特殊字符路径（cv2.imread 在 Windows 上不行）
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError("无法读取图像: %s" % path)
        return img
    from PIL import Image  # 需要 pip install pillow
    img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1]  # RGB -> BGR
    return np.ascontiguousarray(img, dtype=np.uint8)


def draw_boxes(image: np.ndarray, dets, yolo: FastYolo) -> np.ndarray:
    if cv2 is None:
        return image
    out = image.copy()
    for d in dets:
        x1, y1, x2, y2 = d.to_xyxy()
        label = "%s %.2f" % (yolo.label_of(d), d.score)
        if d.track_id >= 0:
            label = "ID%d %s" % (d.track_id, label)
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def print_result(r, yolo: FastYolo) -> None:
    """按任务类型打印结果概要。"""
    from fastyolo import (
        ClassifyResult, DepthResult, ObbResult, PoseResult, SegResult, SemSegResult,
    )
    if isinstance(r, list) and r and isinstance(r[0], SegResult):
        print("实例分割 %d 个目标:" % len(r))
        for s in r:
            cov = float(s.mask.mean()) if s.mask is not None else -1.0
            print("  %s: score=%.3f box=(%d,%d,%d,%d) 掩码占比=%.3f"
                  % (yolo.label_of(s.box), s.score, s.box.x, s.box.y,
                     s.box.width, s.box.height, cov))
    elif isinstance(r, list) and r and isinstance(r[0], PoseResult):
        print("姿态估计 %d 个目标:" % len(r))
        for p in r:
            vis = sum(1 for k in p.keypoints if k.score > 0.5)
            print("  %s: score=%.3f box=(%d,%d,%d,%d) 可见关键点 %d/%d"
                  % (yolo.label_of(p.box), p.score, p.box.x, p.box.y,
                     p.box.width, p.box.height, vis, len(p.keypoints)))
    elif isinstance(r, list) and r and isinstance(r[0], ObbResult):
        print("旋转框 %d 个目标:" % len(r))
        for o in r:
            print("  %s: score=%.3f AABB=(%d,%d,%d,%d) 角度=%.3f rad"
                  % (yolo.label_of(o.box), o.score, o.box.x, o.box.y,
                     o.box.width, o.box.height, o.angle))
    elif isinstance(r, ClassifyResult):
        print("分类: top1=%d score=%.4f top5=%s" % (r.top_class, r.top_score, r.top_ids))
    elif isinstance(r, SemSegResult):
        classes, counts = np.unique(r.class_map, return_counts=True)
        top = sorted(zip(classes.tolist(), counts.tolist()), key=lambda p: -p[1])[:5]
        print("语义分割 %dx%d, 类别数=%d, 占比前5: %s"
              % (r.width, r.height, len(classes), top))
    elif isinstance(r, DepthResult):
        print("深度图 %dx%d, 值域 [%.3f, %.3f]" % (r.width, r.height, r.depth_min, r.depth_max))
    else:  # detect
        print("检测到 %d 个目标:" % len(r))
        for d in r:
            print("  %s: score=%.3f box=(%d,%d,%d,%d) track_id=%d"
                  % (yolo.label_of(d), d.score, d.x, d.y, d.width, d.height, d.track_id))


def main() -> int:
    ap = argparse.ArgumentParser(description="fastyolo 检测/跟踪示例")
    ap.add_argument("--model", required=True, help="ONNX 模型路径")
    ap.add_argument("--image", required=True, help="输入图像路径")
    ap.add_argument("--device", default="CPU", choices=[d.name for d in InferDevice],
                    help="推理设备（默认 CPU）")
    ap.add_argument("--threads", type=int, default=4, help="CPU 线程数")
    ap.add_argument("--min-score", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--track", action="store_true", help="启用 ByteTrack 跟踪（仅 DETECT）")
    ap.add_argument("--task", default=None, choices=["auto", "detect", "segment", "pose",
                                                     "obb", "classify", "semseg", "depth"],
                    help="任务类型（默认 auto 按模型输出自动推断；v8 OBB 需显式 obb）")
    ap.add_argument("--output", default=None, help="画框结果保存路径（仅框类任务）")
    args = ap.parse_args()

    from fastyolo import TaskType
    task = TaskType[args.task.upper()] if args.task and args.task != "auto" else TaskType.AUTO
    yolo = FastYolo(args.model, device=InferDevice[args.device],
                    num_threads=args.threads, min_score=args.min_score,
                    iou_threshold=args.iou, class_names=COCO_NAMES, task=task)
    print("任务:", yolo.task, "输入 shape:", yolo.input_shape, "输出 shape:", yolo.output_shape)

    image = load_image(args.image)
    result = yolo.track(image) if args.track else yolo.predict(image, need_masks=True)
    print_result(result, yolo)

    if args.output:
        from fastyolo import SegResult, ObbResult, PoseResult
        if isinstance(result, list) and result and isinstance(result[0], (SegResult, ObbResult, PoseResult)):
            dets = [r.box for r in result]
        elif isinstance(result, list):
            dets = result
        else:
            dets = []
        out = draw_boxes(image, dets, yolo)
        if cv2 is None:
            print("[WARN] 未安装 opencv-python，无法画框，直接保存原图")
            _save_with_pil(out, args.output)
        else:
            cv2.imencode(".jpg", out)[1].tofile(args.output)  # 支持中文路径
        print("结果已保存: %s" % args.output)
    return 0


def _save_with_pil(image: np.ndarray, path: str) -> None:
    from PIL import Image
    Image.fromarray(image[:, :, ::-1]).save(path)


if __name__ == "__main__":
    sys.exit(main())

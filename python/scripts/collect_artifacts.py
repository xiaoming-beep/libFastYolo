#!/usr/bin/env python3
"""构建后收集产物到 fastyolo/libs/（供 ctypes 加载）。

用法:
  python python/scripts/collect_artifacts.py <build_dir> [--backend onnx|openvino|mnn|tensorrt]
                                                   [--runtime slim|full]

产物位置:
  Windows: <build>/bin/FastYolo.dll + 后端 dll + OpenVINO 运行时 dll
  Linux:   <build>/lib/libFastYolo.so + 后端 so + OpenVINO 运行时 so

--runtime 控制是否收集后端运行时（默认 slim）：
  slim  只收核心库 + 后端库本身（约 0.2MB）。OpenVINO 运行时 DLL 不复制，
        运行时由 _loader 自动从已安装的 openvino pip 包加载（与 C++ 构建时
        用 OPENVINO_ROOT 指向 site-packages/openvino 是同一思路，避免在项目里
        保留 170MB 的 Windows/Linux 双份运行时副本）。
  full  连同后端运行时一起收集（离线/独立部署场景，wheel 打包用）。

--backend 不传时根据产物自动识别。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

_CORE = ("FastYolo.dll", "libFastYolo.so", "libFastYolo.dylib")

# 后端库本身（slim 模式也会收集）
_BACKEND_LIBS = {
    "onnx": ("yolo_onnx.", "libyolo_onnx."),
    "openvino": ("yolo_openvino.", "libyolo_openvino."),
    "mnn": ("yolo_mnn.", "libyolo_mnn."),
    "tensorrt": ("yolo_tensorrt.", "libyolo_tensorrt."),
}

# 后端运行时（仅 --runtime full 收集；openvino 的运行时优先从 pip openvino 包复用）
_RUNTIME_PREFIXES = {
    "onnx": ("onnxruntime", "libonnxruntime"),
    "openvino": ("openvino", "libopenvino", "tbb", "libtbb", "hwloc", "libhwloc"),
}


def _relevant(name: str, backend: str | None, runtime: str) -> bool:
    low = name.lower()
    # 只收集运行时动态库（跳过 .lib 导入库 / .exp / .pdb 等）
    if not (low.endswith(".dll") or low.endswith(".so") or ".so." in low):
        return False
    if name in _CORE:
        return True
    if backend is not None:
        if any(name.startswith(p) for p in _BACKEND_LIBS[backend]):
            return True
        if runtime == "full" and any(name.startswith(p) for p in _RUNTIME_PREFIXES[backend]):
            return True
        return False
    # 未指定 backend：收所有后端库与运行时
    for prefixes in _BACKEND_LIBS.values():
        if any(name.startswith(p) for p in prefixes):
            return True
    if runtime == "full":
        for prefixes in _RUNTIME_PREFIXES.values():
            if any(name.startswith(p) for p in prefixes):
                return True
    return False


def _detect_backend(build_dir: str) -> str | None:
    for d in ("bin", "lib"):
        for scan in (d, *(os.path.join(d, sub) for sub in ("Release", "Debug"))):
            dir_path = os.path.join(build_dir, scan)
            if not os.path.isdir(dir_path):
                continue
            for f in os.listdir(dir_path):
                for backend, prefixes in _BACKEND_LIBS.items():
                    if any(f.startswith(p) for p in prefixes):
                        return backend
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("build_dir", help="CMake 构建目录（如 build-py）")
    ap.add_argument("--backend", choices=sorted(_BACKEND_LIBS), default=None,
                    help="后端名；默认按产物自动识别")
    ap.add_argument("--runtime", choices=("slim", "full"), default="slim",
                    help="是否收集后端运行时（默认 slim：运行时复用 openvino pip 包）")
    ap.add_argument("--cudnn-lib", metavar="DIR", default=None,
                    help="额外从 DIR 拷贝 libcudnn*.so.9* 到 libs/（ONNX CUDA 后端自包含部署用；"
                         "构建时 patchelf 已给 providers_cuda 打 $ORIGIN，cuDNN 与之同目录即可零环境变量运行）")
    args = ap.parse_args()

    build = os.path.abspath(args.build_dir)
    if not os.path.isdir(build):
        print("[ERROR] 构建目录不存在: %s" % build, file=sys.stderr)
        return 1

    backend = args.backend or _detect_backend(build)
    if backend is None:
        print("[ERROR] 无法识别后端（未找到后端库产物）", file=sys.stderr)
        return 1

    libs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "fastyolo", "libs")
    os.makedirs(libs_dir, exist_ok=True)

    # 清理旧文件：删除当前模式下不再需要的（如 slim 模式下的运行时 DLL 残留）
    for f in os.listdir(libs_dir):
        path = os.path.join(libs_dir, f)
        if os.path.isfile(path) and not _relevant(f, backend, args.runtime):
            os.remove(path)

    copied = []
    for d in ("bin", "lib"):
        # 多配置生成器（VS）产物在 bin/Release、lib/Debug 等配置子目录
        for scan in (d, *(os.path.join(d, sub) for sub in ("Release", "Debug", "RelWithDebInfo", "MinSizeRel"))):
            full = os.path.join(build, scan)
            if not os.path.isdir(full):
                continue
            for f in sorted(os.listdir(full)):
                src = os.path.join(full, f)
                if not os.path.isfile(src) or not _relevant(f, backend, args.runtime):
                    continue
                shutil.copy2(src, os.path.join(libs_dir, f))
                copied.append(f)

    if not copied:
        print("[ERROR] 未在 %s 的 bin/ 或 lib/ 下找到任何产物" % build, file=sys.stderr)
        return 1

    # --cudnn-lib：拷贝 cuDNN 运行时（清理阶段刚执行过，这里新增不会被删）
    if args.cudnn_lib:
        if not os.path.isdir(args.cudnn_lib):
            print("[ERROR] --cudnn-lib 目录不存在: %s" % args.cudnn_lib, file=sys.stderr)
            return 1
        n = 0
        for f in sorted(os.listdir(args.cudnn_lib)):
            if f.startswith("libcudnn") and (".so" in f):
                shutil.copy2(os.path.join(args.cudnn_lib, f), os.path.join(libs_dir, f))
                copied.append(f)
                n += 1
        if n == 0:
            print("[ERROR] %s 下未找到 libcudnn*.so*（Pascal/Maxwell 老卡需 cuDNN 9.1，见《编译指南.md》）"
                  % args.cudnn_lib, file=sys.stderr)
            return 1
        print("[OK] 已拷贝 %d 个 cuDNN 运行时库" % n)

    print("[OK] backend=%s，产物已收集到 %s" % (backend, libs_dir))
    for f in copied:
        print("  - %s" % f)
    return 0


if __name__ == "__main__":
    sys.exit(main())

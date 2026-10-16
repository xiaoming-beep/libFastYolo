"""定位并加载 libFastYolo 动态库。

运行时文件两种布局，加载器自动兼容：
  1. 捆绑布局：包内 libs/ 自带 OpenVINO 运行时（build_wheel.py --full）
  2. 精简布局：libs/ 只有 FastYolo.dll + 后端库（默认 wheel），
     OpenVINO 运行时从已安装的 openvino pip 包（site-packages/openvino/libs）加载

Windows 下通过 os.add_dll_directory() 把相关目录加入 DLL 搜索路径，
ctypes 加载 FastYolo.dll 时其依赖（后端库 + OpenVINO 运行时）才能被找到，
用户无需手动设置 PATH。
"""
from __future__ import annotations

import ctypes
import glob
import importlib.util
import os
import sys
import time

_LIBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")

# 各平台的 OpenVINO 运行时核心库（按依赖顺序，先 _c 后主库）
_RUNTIME_CORES = {
    "win32": ("openvino_c.dll", "openvino.dll"),
    "linux": ("libopenvino_c.so", "libopenvino.so"),
}


def _candidate_names() -> list[str]:
    if sys.platform == "win32":
        return ["FastYolo.dll", "libFastYolo.dll"]
    return ["libFastYolo.so"]


def _has_bundled_runtime() -> bool:
    """libs/ 是否自带 OpenVINO 运行时（捆绑布局）。"""
    for name in _RUNTIME_CORES.get(sys.platform, ()):
        if os.path.exists(os.path.join(_LIBS_DIR, name)):
            return True
    return False


def _openvino_package_libs() -> str | None:
    """定位 OpenVINO 运行时目录（与 C++ 构建时 -DOPENVINO_ROOT 同一来源）。

    优先级：
      1. 环境变量 FASTYOLO_OPENVINO_ROOT（手动指定，例如指向 pip 的
         site-packages/openvino 目录）
      2. 自动探测已安装的 openvino pip 包（不真正 import，避免副作用/开销）
    """
    env = os.environ.get("FASTYOLO_OPENVINO_ROOT")
    if env:
        candidate = env if env.endswith(("libs", "libs/", "libs\\")) else os.path.join(env, "libs")
        if os.path.isdir(candidate):
            return candidate
    try:
        spec = importlib.util.find_spec("openvino")
        if spec and spec.submodule_search_locations:
            root = next(iter(spec.submodule_search_locations))
            candidate = os.path.join(root, "libs")
            if os.path.isdir(candidate):
                return candidate
    except Exception:
        pass
    return None


def _preload_runtime(dirs: list[str]) -> None:
    """按绝对路径预加载 OpenVINO 运行时核心库（Linux 需要 GLOBAL 符号可见）。"""
    mode = getattr(os, "RTLD_GLOBAL", 0)
    for d in dirs:
        for name in _RUNTIME_CORES.get(sys.platform, ()):
            path = os.path.join(d, name)
            if not os.path.exists(path):
                # pip 包内只有带版本号的文件（libopenvino.so.2630 等），glob 兜底
                matches = sorted(glob.glob(os.path.join(d, name + ".*")))
                path = matches[0] if matches else path
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path, mode=mode)
                except OSError:
                    pass


def load_library() -> ctypes.CDLL:
    """加载核心动态库；libs/ 缺失时给出可操作的错误提示。"""
    # 运行时搜索目录：包内 libs/ 优先；精简布局追加 openvino 包的 libs/
    dirs = [_LIBS_DIR]
    if not _has_bundled_runtime():
        ov_libs = _openvino_package_libs()
        if ov_libs is not None:
            dirs.append(ov_libs)

    if sys.platform == "win32":
        # Python 3.8+；失败（如极老的系统）不影响后续显式路径加载
        for d in dirs:
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
    _preload_runtime(dirs)

    for name in _candidate_names():
        path = os.path.join(_LIBS_DIR, name)
        if not os.path.exists(path):
            continue
        # 防瞬时占用：Windows Defender 等可能在新写入的大体积运行时 DLL
        # （如 82MB 的 openvino_intel_npu_compiler.dll）上扫描数十秒，
        # 期间 LoadLibrary 会报 WinError 32；指数退避重试 ~30s
        last_err: OSError | None = None
        for attempt in range(6):
            try:
                return ctypes.CDLL(path)
            except OSError as e:  # WinError 32 等
                last_err = e
                time.sleep(0.5 * (2 ** attempt))
        raise last_err

    raise RuntimeError(
        "未找到 libFastYolo 动态库（%s）。\n"
        "该包需要先构建：Windows 运行 python/scripts/build_windows.bat"
        "（OpenVINO 后端），Linux 运行 python/scripts/build_linux.sh"
        "（ONNX 后端），产物会自动收集到本目录。" % _LIBS_DIR
    )

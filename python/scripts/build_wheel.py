#!/usr/bin/env python3
"""构建可分发的预编译 wheel —— 用户机器上无需 cmake/编译器/VS。

开发机（或 CI）上先运行 build_windows.bat / build_linux.sh 完成 C++ 构建，
再运行本脚本产出平台 wheel：

  python scripts/build_wheel.py            # 精简版（默认）：仅 FastYolo + 后端库，
                                           # OpenVINO 运行时从 openvino pip 包加载（~2MB）
  python scripts/build_wheel.py --full     # 全捆绑：含 OpenVINO 运行时（~170MB，离线可装）
  python scripts/build_wheel.py --out DIR  # 指定输出目录（默认 python/dist）

输出: dist/fastyolo-<ver>-py3-none-<platform>.whl
      （py3-none 标签：同一 wheel 覆盖所有 Python 3.x，无需按版本重编）

分发方式（用户侧均为纯拷贝安装，无任何编译步骤）：
  1. 直接安装 wheel 文件:  pip install fastyolo-0.1.0-py3-none-win_amd64.whl
  2. 上传内部 PyPI 后:      pip install fastyolo -i <内部源>
  3. 挂到 Web/网盘后:       pip install <URL>
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

# 各平台 wheel 保留的运行时文件（精简模式）
_CORE_LIBS = {
    "win32": ("FastYolo.dll", "yolo_openvino.dll"),
    "linux": ("libFastYolo.so", "libyolo_onnx.so"),
}
_PLAT_TAG = {
    "win32": "win_amd64",
    "linux": "manylinux_2_17_x86_64",  # 静态链接的 ONNX 后端，理论上兼容更老的 glibc
}

_PACKAGE_FILES = ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md")


def _find_openvino_libs() -> str | None:
    """定位 openvino pip 包的运行时目录（--full 补齐运行时用）。"""
    import importlib.util
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="全捆绑模式（含 OpenVINO 运行时）")
    ap.add_argument("--out", default=None, help="wheel 输出目录（默认 python/dist）")
    args = ap.parse_args()

    if sys.platform not in _CORE_LIBS:
        print("[ERROR] 当前平台 %s 不支持 wheel 构建（仅 win32/linux）" % sys.platform,
              file=sys.stderr)
        return 1

    py_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_dir = os.path.join(py_dir, "fastyolo")
    libs_dir = os.path.join(pkg_dir, "libs")
    core_libs = _CORE_LIBS[sys.platform]

    missing = [f for f in core_libs if not os.path.exists(os.path.join(libs_dir, f))]
    if missing:
        print("[ERROR] libs/ 缺少构建产物: %s；请先运行构建脚本" % ", ".join(missing),
              file=sys.stderr)
        return 1

    out_dir = os.path.abspath(args.out or os.path.join(py_dir, "dist"))
    os.makedirs(out_dir, exist_ok=True)

    tmp = tempfile.mkdtemp(prefix="fastyolo_wheel_")
    build_out = os.path.join(tmp, "wheels")
    os.makedirs(build_out, exist_ok=True)
    try:
        # 1. 打包文件（pyproject/setup/MANIFEST/README）
        for f in _PACKAGE_FILES:
            src = os.path.join(py_dir, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp, f))

        # 2. 拷贝包；libs 按模式过滤（精简删运行时 / 全捆绑保留）
        dst_pkg = os.path.join(tmp, "fastyolo")
        shutil.copytree(pkg_dir, dst_pkg, ignore=shutil.ignore_patterns("__pycache__"))
        if not args.full:
            for f in os.listdir(os.path.join(dst_pkg, "libs")):
                if f not in core_libs:
                    os.remove(os.path.join(dst_pkg, "libs", f))

        # 3. 全捆绑模式：运行时已自带，去掉 openvino 依赖，保证目标机可完全离线。
        #    项目 libs/ 默认是精简的（运行时复用 openvino pip 包），此时从
        #    openvino 包补齐运行时，不依赖 collect --runtime full
        if args.full:
            ov_libs = _find_openvino_libs()
            if ov_libs is None:
                print("[ERROR] --full 需要已安装 openvino 包（python -m pip install openvino）",
                      file=sys.stderr)
                return 1
            dst_libs = os.path.join(dst_pkg, "libs")
            for f in os.listdir(ov_libs):
                low = f.lower()
                if not (low.endswith(".dll") or low.endswith(".so") or ".so." in low):
                    continue
                if not os.path.exists(os.path.join(dst_libs, f)):
                    shutil.copy2(os.path.join(ov_libs, f), os.path.join(dst_libs, f))

            pyproject = os.path.join(tmp, "pyproject.toml")
            with open(pyproject, encoding="utf-8") as f:
                text = f.read()
            text = text.replace(', "openvino>=2024.0"', "")
            with open(pyproject, "w", encoding="utf-8") as f:
                f.write(text)

        # 4. 构建 wheel（--plat-name 强制平台 tag，避免产出 py3-none-any）
        plat = _PLAT_TAG[sys.platform]
        subprocess.check_call(
            [sys.executable, "setup.py", "bdist_wheel", "--plat-name", plat, "-d", build_out],
            cwd=tmp,
        )

        # 5. 统一文件名，区分精简/全捆绑（避免同名覆盖）
        wheels = sorted(f for f in os.listdir(build_out) if f.endswith(".whl"))
        if not wheels:
            print("[ERROR] 未生成 wheel", file=sys.stderr)
            return 1
        version = wheels[-1].split("-")[1]
        new_name = "fastyolo-%s-py3-none-%s%s.whl" % (version, plat, "-full" if args.full else "")
        latest = os.path.join(out_dir, new_name)
        if os.path.exists(latest):
            os.remove(latest)  # Windows 上 rename 不覆盖已存在文件
        os.rename(os.path.join(build_out, wheels[-1]), latest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    size_mb = os.path.getsize(latest) / 1024 / 1024
    print("[OK] wheel 已生成: %s (%.1f MB, %s)" % (latest, size_mb,
          "全捆绑" if args.full else "精简"))
    print("分发: pip install %s" % latest)
    return 0


if __name__ == "__main__":
    sys.exit(main())

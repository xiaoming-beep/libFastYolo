"""把 engine/src/kernels/*.cl 嵌进 C++ 源码，生成 kernels_generated.cpp。

这样产出的后端库是自包含的：运行时不需要旁边放一个 kernels 目录，单个 dll 就是完整的。

调试内核时可以用环境变量 ONNXCL_KERNELS 指向磁盘目录覆盖嵌入版本，免去重新编译
（见 engine/src/cl/cl_runtime.cpp 的 lookupSource）。

用法: python gen_kernels.py <kernels_dir> <out_cpp>
"""
import glob
import os
import sys

DELIM = "ONNXCL"

# 单个字符串字面量的切分长度。
#
# 【为什么必须切】MSVC 对单个字符串字面量有长度上限（C2026: 字符串太大、
# 已截断尾部字符，阀值约 16380 字节）。把整个 .cl 源码塞进一个 R"(...)" 里，
# 一旦内核变大（比如加了 conv2d_gemm_small）MSVC 就编不过，而 MinGW/GCC 没这个
# 限制 —— 所以这个坑只在 MSVC 下暴露。
# C++ 会把相邻的字面量在编译期拼成一个 const char*，切开不影响功能。
LITERAL_CHUNK = 6000


def emit_literal(text):
    """把一段源码切成多段相邻的 raw string 字面量（规避 MSVC 的单字面量长度上限）。"""
    parts = []
    for i in range(0, len(text), LITERAL_CHUNK):
        parts.append('R"%s(%s)%s"' % (DELIM, text[i:i + LITERAL_CHUNK], DELIM))
    return "\n        ".join(parts)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src_dir = sys.argv[1]
    out_cpp = sys.argv[2]

    entries = []
    for path in sorted(glob.glob(os.path.join(src_dir, "*.cl"))):
        name = os.path.basename(path)[:-3]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        if ')' + DELIM + '"' in text:
            raise SystemExit(f"{path} 里含有分隔符 ){DELIM}\"，换一个 DELIM")
        entries.append((name, text))

    if not entries:
        raise SystemExit(f"{src_dir} 里没有 .cl 文件")

    # 输出目录可能不存在（CMake 的 binary dir 首次构建时）
    out_dir = os.path.dirname(os.path.abspath(out_cpp))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_cpp, "w", encoding="utf-8", newline="\n") as f:
        f.write("// 由 gen_kernels.py 自动生成，请勿手改\n")
        f.write("#include <map>\n#include <string>\n\n")
        f.write("namespace onnxcl {\n\n")
        f.write("static const std::map<std::string, const char*> kEmbeddedSources = {\n")
        for name, text in entries:
            f.write('    {"%s",\n        %s},\n' % (name, emit_literal(text)))
        f.write("};\n\n")
        f.write("const char* embeddedKernelSource(const std::string& name) {\n")
        f.write("    auto it = kEmbeddedSources.find(name);\n")
        f.write("    return it == kEmbeddedSources.end() ? nullptr : it->second;\n")
        f.write("}\n\n")
        f.write("}  // namespace onnxcl\n")

    total = sum(len(t) for _, t in entries)
    print(f"[onnxcl] 嵌入 {len(entries)} 个内核 (共 {total} 字节) -> {out_cpp}")
    for name, text in entries:
        nchunk = (len(text) + LITERAL_CHUNK - 1) // LITERAL_CHUNK
        print(f"  {name}.cl  {len(text)} 字节, 切成 {nchunk} 段")


if __name__ == "__main__":
    main()

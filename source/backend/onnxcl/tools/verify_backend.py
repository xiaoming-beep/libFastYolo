"""onnxcl 后端集成验证

把 test_onnxcl dump 出来的 Letterbox 像素喂给 onnxruntime，和 onnxcl 后端吐出的
原始输出张量对拍。这样：
  · 前处理对不对 -> 由 Letterbox 的输出像素间接体现（onnxruntime 用的是同一份像素）
  · 引擎算得对不对 -> 由输出张量逐元素对拍判定
  · 数据通路（模型字节加载 / 输出暴露）对不对 -> 输出形状和数值都能对上才算过

用法: python verify_backend.py <模型.onnx> <输出前缀>
      例: python verify_backend.py yolo11n.onnx build/bin/onnxcl_test
"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort


DT = {1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64", 10: "float16"}
NP = {1: np.float32, 2: np.uint8, 3: np.int8, 6: np.int32, 7: np.int64}


# ONNX 元素类型码 -> numpy dtype（读 test_onnxcl 写出的二进制时需要它）
def onnx_elem_size(et):
    return {1: 4, 2: 1, 3: 1, 6: 4, 7: 8}.get(et, 4)


def compare(tag, got, ref, atol=1e-4, rtol=1e-4):
    """判据用数值计算的标准形式 diff <= atol + rtol*|ref|。

    不能用单纯的 max_rel —— 参考值接近 0 时相对误差没有意义。
    """
    g = got.astype(np.float64)
    r = ref.astype(np.float64)
    diff = np.abs(g - r)
    tol = atol + rtol * np.abs(r)
    norm = diff / tol
    worst = int(np.argmax(norm))
    nbad = int((norm > 1.0).sum())
    cos = float(np.dot(g, r) / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-12))

    passed = nbad == 0 and np.isfinite(g).all()
    print(f"  {tag} {'✅' if passed else '❌'}  shape={list(got.shape)}")
    print(f"      最大绝对误差 : {diff.max():.3e}   (该处参考值 {r[worst]:.4f})")
    print(f"      平均绝对误差 : {diff.mean():.3e}")
    print(f"      归一化误差   : {norm.max():.3f} (需 < 1.0)   超阈元素: {nbad}/{diff.size}")
    print(f"      余弦相似度   : {cos:.8f}")
    print(f"      数值范围     : 参考 [{r.min():.4f}, {r.max():.4f}]"
          f"  我们 [{g.min():.4f}, {g.max():.4f}]")
    if not passed:
        for b in np.argsort(-norm)[:5]:
            print(f"        最差 [{b}] 我们={g[b]:.6f} 参考={r[b]:.6f} "
                  f"diff={diff[b]:.3e} 容差={tol[b]:.3e}")
    return passed


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    model_path, prefix = sys.argv[1], sys.argv[2]

    pixels_file = prefix + "_pixels.bin"
    if not os.path.exists(pixels_file):
        print(f"找不到 {pixels_file}，先跑 test_onnxcl")
        return 1

    m = onnx.load(model_path, load_external_data=False)
    inp = m.graph.input[0]
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    pixels = np.fromfile(pixels_file, dtype=np.float32)
    expect = int(np.prod(shape))
    if pixels.size != expect:
        print(f"像素数不符: 文件 {pixels.size}, 模型要 {expect} {shape}")
        return 1
    arr = np.ascontiguousarray(pixels.reshape(shape), dtype=np.float32)
    print(f"模型     : {os.path.basename(model_path)}")
    print(f"前处理输出: {shape}  (来自 libFastYolo 的 Letterbox)")
    print(f"  像素统计: 范围 [{arr.min():.4f}, {arr.max():.4f}] 均值 {arr.mean():.4f}")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    refs = sess.run(None, {sess.get_inputs()[0].name: arr})
    # onnxruntime 的输出类型字符串 -> ONNX 元素类型码
    # （必须按实际类型读文件：uint8 的类别图当 float32 读会越界）
    ORT_TYPE = {"tensor(float)": 1, "tensor(uint8)": 2, "tensor(int8)": 3,
                "tensor(int32)": 6, "tensor(int64)": 7, "tensor(float16)": 10}
    out_types = [ORT_TYPE.get(o.type, 1) for o in sess.get_outputs()]
    print(f"onnxruntime 输出: {[(list(r.shape), DT.get(t, t)) for r, t in zip(refs, out_types)]}")
    print()

    ok = True
    for i, ref in enumerate(refs):
        f = f"{prefix}_out{i}.bin"
        if not os.path.exists(f):
            print(f"  输出[{i}] 缺失: {f}")
            ok = False
            continue
        et = out_types[i] if i < len(out_types) else 1
        npdt = NP.get(et, np.float32)
        got = np.fromfile(f, dtype=npdt)
        ref_flat = np.asarray(ref).ravel().astype(npdt)
        if got.size != ref_flat.size:
            print(f"  输出[{i}] ❌ 元素数不符: 我们 {got.size}, 参考 {ref_flat.size}")
            ok = False
            continue
        ok = compare(f"输出[{i}]", got.reshape(-1), ref_flat.reshape(-1)) and ok

    print()
    print("✅ 集成验证通过：onnxcl 后端接进 libFastYolo 后数据通路与数值都正确"
          if ok else "❌ 集成验证不通过")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

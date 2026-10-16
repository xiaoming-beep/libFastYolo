#include "shape_infer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace onnxcl {
namespace {

// ---------------------------------------------------------------------------
// 取值小工具
// ---------------------------------------------------------------------------

const std::vector<int64_t>* inShape(const Node& n, const Graph& g, size_t i) {
    if (i >= n.inputs.size() || n.inputs[i] < 0) return nullptr;
    return &g.at(n.inputs[i]).shape;
}

const std::vector<int64_t>* inConstI64(const Node& n, const Graph& g, size_t i) {
    if (i >= n.inputs.size() || n.inputs[i] < 0) return nullptr;
    const TensorInfo& t = g.at(n.inputs[i]);
    if (!t.isConst || t.constI64.empty()) return nullptr;
    return &t.constI64;
}

DType inDtype(const Node& n, const Graph& g, size_t i) {
    if (i >= n.inputs.size() || n.inputs[i] < 0) return DType::F32;
    return g.at(n.inputs[i]).dtype;
}

void fail(std::string* err, const Node& n, const char* why) {
    if (err) {
        *err = "形状推导失败: op=" + n.opType + " name=" + n.name + " :: " + why;
    }
}

// 设置第 idx 个输出的 shape / dtype
bool setOut(Graph& g, const Node& n, size_t idx, std::vector<int64_t> shape, DType dt,
            std::string* err) {
    if (idx >= n.outputs.size() || n.outputs[idx] < 0) {
        fail(err, n, "输出下标越界");
        return false;
    }
    TensorInfo& t = g.at(n.outputs[idx]);
    t.shape = std::move(shape);
    t.dtype = dt;
    return true;
}

bool setAllOutsSameAsInput0(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* s = inShape(n, g, 0);
    if (s == nullptr) {
        fail(err, n, "缺少输入 0 的 shape");
        return false;
    }
    for (size_t i = 0; i < n.outputs.size(); ++i) {
        if (!setOut(g, n, i, *s, inDtype(n, g, 0), err)) return false;
    }
    return true;
}

int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// 空间算子（Conv / Pool）：把 padding 统一成 800 字节内的 4 元素数组
// ---------------------------------------------------------------------------

struct SpatialAttrs {
    int64_t kh = 1, kw = 1;
    int64_t sh = 1, sw = 1;
    int64_t dh = 1, dw = 1;
    int64_t pt = 0, pl = 0, pb = 0, pr = 0;
    bool ceilMode = false;
};

bool fillSpatialAttrs(const Node& n, const Graph& g, int64_t khDefault, int64_t kwDefault,
                      SpatialAttrs& a, std::string* err) {
    std::vector<int64_t> k = n.attrInts("kernel_shape");
    if (k.size() >= 2) {
        a.kh = k[0];
        a.kw = k[1];
    } else {
        a.kh = khDefault;
        a.kw = kwDefault;
    }
    std::vector<int64_t> s = n.attrInts("strides");
    if (s.size() >= 2) {
        a.sh = s[0];
        a.sw = s[1];
    }
    std::vector<int64_t> d = n.attrInts("dilations");
    if (d.size() >= 2) {
        a.dh = d[0];
        a.dw = d[1];
    }
    std::vector<int64_t> p = n.attrInts("pads");
    if (p.size() >= 4) {
        a.pt = p[0];
        a.pl = p[1];
        a.pb = p[2];
        a.pr = p[3];
    }
    a.ceilMode = n.attrInt("ceil_mode", 0) != 0;
    if (a.kh <= 0 || a.kw <= 0 || a.sh <= 0 || a.sw <= 0) {
        fail(err, n, "kernel/strides 非法");
        return false;
    }
    return true;
}

int64_t spatialOut(int64_t in, int64_t k, int64_t s, int64_t d, int64_t p0, int64_t p1, bool ceilMode) {
    const int64_t effective = d * (k - 1) + 1;
    const int64_t numer = in + p0 + p1 - effective;
    int64_t out = ceilMode ? ceilDiv(numer, s) : (numer / s);
    return out + 1;
}

// ---------------------------------------------------------------------------
// 各算子的形状规则
// ---------------------------------------------------------------------------

bool inferConv(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    const std::vector<int64_t>* ws = inShape(n, g, 1);
    if (xs == nullptr || ws == nullptr || xs->size() != 4 || ws->size() != 4) {
        fail(err, n, "Conv 期望 4 维输入");
        return false;
    }
    SpatialAttrs a;
    if (!fillSpatialAttrs(n, g, (*ws)[2], (*ws)[3], a, err)) return false;

    const int64_t N = (*xs)[0];
    const int64_t C = (*xs)[1];
    const int64_t H = (*xs)[2];
    const int64_t W = (*xs)[3];
    const int64_t K = (*ws)[0];
    const int64_t Cg = (*ws)[1];
    const int64_t group = n.attrInt("group", 1);
    if (Cg * group != C) {
        fail(err, n, "Conv group 与通道数不匹配");
        return false;
    }
    const int64_t OH = spatialOut(H, a.kh, a.sh, a.dh, a.pt, a.pb, false);
    const int64_t OW = spatialOut(W, a.kw, a.sw, a.dw, a.pl, a.pr, false);
    return setOut(g, n, 0, {N, K, OH, OW}, inDtype(n, g, 0), err);
}

bool inferConvTranspose(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    const std::vector<int64_t>* ws = inShape(n, g, 1);
    if (xs == nullptr || ws == nullptr || xs->size() != 4 || ws->size() != 4) {
        fail(err, n, "ConvTranspose 期望 4 维输入");
        return false;
    }
    SpatialAttrs a;
    if (!fillSpatialAttrs(n, g, (*ws)[2], (*ws)[3], a, err)) return false;

    const int64_t N = (*xs)[0];
    const int64_t H = (*xs)[2];
    const int64_t W = (*xs)[3];
    const int64_t K = (*ws)[1] * n.attrInt("group", 1);  // 输出通道
    // ONNX: out = (in-1)*stride - p0 - p1 + dilation*(k-1) + 1 + output_padding
    const int64_t OH = (H - 1) * a.sh - a.pt - a.pb + a.dh * (a.kh - 1) + 1;
    const int64_t OW = (W - 1) * a.sw - a.pl - a.pr + a.dw * (a.kw - 1) + 1;
    return setOut(g, n, 0, {N, K, OH, OW}, inDtype(n, g, 0), err);
}

bool inferPool(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    if (xs == nullptr || xs->size() != 4) {
        fail(err, n, "Pool 期望 4 维输入");
        return false;
    }
    SpatialAttrs a;
    if (!fillSpatialAttrs(n, g, 1, 1, a, err)) return false;
    const int64_t OH = spatialOut((*xs)[2], a.kh, a.sh, a.dh, a.pt, a.pb, a.ceilMode);
    const int64_t OW = spatialOut((*xs)[3], a.kw, a.sw, a.dw, a.pl, a.pr, a.ceilMode);
    return setOut(g, n, 0, {(*xs)[0], (*xs)[1], OH, OW}, inDtype(n, g, 0), err);
}

bool inferBinary(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* a = inShape(n, g, 0);
    const std::vector<int64_t>* b = inShape(n, g, 1);
    if (a == nullptr || b == nullptr) {
        fail(err, n, "BinaryOp 缺少输入 shape");
        return false;
    }
    std::vector<int64_t> out;
    if (!broadcastShapes(*a, *b, out)) {
        fail(err, n, ("无法广播 " + shapeToString(*a) + " 与 " + shapeToString(*b)).c_str());
        return false;
    }
    return setOut(g, n, 0, out, inDtype(n, g, 0), err);
}

bool inferConcat(const Node& n, Graph& g, std::string* err) {
    if (n.inputs.empty()) {
        fail(err, n, "Concat 没有输入");
        return false;
    }
    const std::vector<int64_t>* first = inShape(n, g, 0);
    if (first == nullptr) {
        fail(err, n, "Concat 缺少输入 shape");
        return false;
    }
    std::vector<int64_t> out = *first;
    int64_t axis = n.attrInt("axis", 0);
    if (axis < 0) axis += static_cast<int64_t>(out.size());
    if (axis < 0 || axis >= static_cast<int64_t>(out.size())) {
        fail(err, n, "Concat axis 越界");
        return false;
    }
    int64_t sum = 0;
    for (size_t i = 0; i < n.inputs.size(); ++i) {
        const std::vector<int64_t>* s = inShape(n, g, i);
        if (s == nullptr || s->size() != out.size()) {
            fail(err, n, "Concat 各输入 rank 不一致");
            return false;
        }
        sum += (*s)[axis];
    }
    out[axis] = sum;
    return setOut(g, n, 0, out, inDtype(n, g, 0), err);
}

bool inferSplit(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    if (xs == nullptr) {
        fail(err, n, "Split 缺少输入 shape");
        return false;
    }
    std::vector<int64_t> out = *xs;
    int64_t axis = n.attrInt("axis", 0);
    if (axis < 0) axis += static_cast<int64_t>(out.size());

    std::vector<int64_t> sizes;
    const std::vector<int64_t>* c = inConstI64(n, g, 1);
    if (c != nullptr && !c->empty()) {
        sizes = *c;
    } else {
        sizes = n.attrInts("split");
    }
    if (sizes.empty()) {
        // 均分
        const int64_t parts = static_cast<int64_t>(n.outputs.size());
        if (parts <= 0 || out[axis] % parts != 0) {
            fail(err, n, "Split 无法均分且没有给出 split 参数");
            return false;
        }
        sizes.assign(parts, out[axis] / parts);
    }
    if (sizes.size() != n.outputs.size()) {
        fail(err, n, "Split 的 split 个数与输出个数不一致");
        return false;
    }
    for (size_t i = 0; i < sizes.size(); ++i) {
        out[axis] = sizes[i];
        if (!setOut(g, n, i, out, inDtype(n, g, 0), err)) return false;
    }
    return true;
}

bool inferReshape(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    const std::vector<int64_t>* target = inConstI64(n, g, 1);
    if (xs == nullptr || target == nullptr) {
        fail(err, n, "Reshape 缺少输入或 shape 不是常量");
        return false;
    }
    if (n.attrInt("allowzero", 0) != 0) {
        fail(err, n, "Reshape allowzero=1 未支持");
        return false;
    }
    int64_t total = 1;
    for (auto d : *xs) total *= d;

    std::vector<int64_t> out(target->size(), 0);
    int64_t negIdx = -1;
    int64_t known = 1;
    for (size_t i = 0; i < target->size(); ++i) {
        int64_t d = (*target)[i];
        if (d == 0) {
            // 0 表示「沿用输入对应维」
            d = (i < xs->size()) ? (*xs)[i] : 1;
        }
        if (d == -1) {
            if (negIdx >= 0) {
                fail(err, n, "Reshape 出现多个 -1");
                return false;
            }
            negIdx = static_cast<int64_t>(i);
            out[i] = 1;  // 占位
            continue;
        }
        known *= d;
        out[i] = d;
    }
    if (negIdx >= 0) {
        if (known == 0) {
            fail(err, n, "Reshape 推导 -1 时除零");
            return false;
        }
        out[negIdx] = total / known;
    } else if (known != total) {
        fail(err, n, ("Reshape 元素总数不匹配: 输入 " + std::to_string(total) + " 目标 " +
                      std::to_string(known))
                         .c_str());
        return false;
    }
    return setOut(g, n, 0, out, inDtype(n, g, 0), err);
}

bool inferFlatten(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    if (xs == nullptr) {
        fail(err, n, "Flatten 缺少输入 shape");
        return false;
    }
    int64_t axis = n.attrInt("axis", 1);
    const int64_t r = static_cast<int64_t>(xs->size());
    if (axis < 0) axis += r;
    int64_t d0 = 1, d1 = 1;
    for (int64_t i = 0; i < axis; ++i) d0 *= (*xs)[i];
    for (int64_t i = axis; i < r; ++i) d1 *= (*xs)[i];
    return setOut(g, n, 0, {d0, d1}, inDtype(n, g, 0), err);
}

bool inferUnsqueeze(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    if (xs == nullptr) {
        fail(err, n, "Unsqueeze 缺少输入 shape");
        return false;
    }
    std::vector<int64_t> axes;
    const std::vector<int64_t>* c = inConstI64(n, g, 1);
    if (c != nullptr && !c->empty()) axes = *c;
    else axes = n.attrInts("axes");
    if (axes.empty()) {
        fail(err, n, "Unsqueeze 没有 axes");
        return false;
    }
    const int64_t outRank = static_cast<int64_t>(xs->size()) + static_cast<int64_t>(axes.size());
    std::vector<bool> isNew(outRank, false);
    for (auto& a : axes) {
        int64_t ax = a < 0 ? a + outRank : a;
        if (ax < 0 || ax >= outRank) {
            fail(err, n, "Unsqueeze axes 越界");
            return false;
        }
        isNew[ax] = true;
    }
    std::vector<int64_t> out;
    size_t src = 0;
    for (int64_t i = 0; i < outRank; ++i) {
        out.push_back(isNew[i] ? 1 : (*xs)[src++]);
    }
    return setOut(g, n, 0, out, inDtype(n, g, 0), err);
}

bool inferSqueeze(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    if (xs == nullptr) {
        fail(err, n, "Squeeze 缺少输入 shape");
        return false;
    }
    std::vector<int64_t> axes;
    const std::vector<int64_t>* c = inConstI64(n, g, 1);
    if (c != nullptr && !c->empty()) axes = *c;
    else axes = n.attrInts("axes");

    std::vector<int64_t> out;
    if (axes.empty()) {
        for (auto d : *xs) {
            if (d != 1) out.push_back(d);
        }
    } else {
        const int64_t r = static_cast<int64_t>(xs->size());
        std::vector<bool> drop(r, false);
        for (auto& a : axes) {
            int64_t ax = a < 0 ? a + r : a;
            if (ax < 0 || ax >= r) {
                fail(err, n, "Squeeze axes 越界");
                return false;
            }
            drop[ax] = true;
        }
        for (int64_t i = 0; i < r; ++i) {
            if (!drop[i]) out.push_back((*xs)[i]);
        }
    }
    return setOut(g, n, 0, out, inDtype(n, g, 0), err);
}

bool inferTranspose(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    if (xs == nullptr) {
        fail(err, n, "Transpose 缺少输入 shape");
        return false;
    }
    const int64_t r = static_cast<int64_t>(xs->size());
    std::vector<int64_t> perm = n.attrInts("perm");
    if (perm.empty()) {
        for (int64_t i = r - 1; i >= 0; --i) perm.push_back(i);
    }
    if (static_cast<int64_t>(perm.size()) != r) {
        fail(err, n, "Transpose perm 长度不对");
        return false;
    }
    std::vector<int64_t> out(r);
    for (int64_t i = 0; i < r; ++i) {
        int64_t p = perm[i] < 0 ? perm[i] + r : perm[i];
        if (p < 0 || p >= r) {
            fail(err, n, "Transpose perm 越界");
            return false;
        }
        out[i] = (*xs)[p];
    }
    return setOut(g, n, 0, out, inDtype(n, g, 0), err);
}

bool inferMatMul(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* a = inShape(n, g, 0);
    const std::vector<int64_t>* b = inShape(n, g, 1);
    if (a == nullptr || b == nullptr || a->empty() || b->empty()) {
        fail(err, n, "MatMul 缺少输入 shape");
        return false;
    }
    // numpy matmul 语义
    std::vector<int64_t> A = *a, B = *b;
    const bool aVec = A.size() == 1;
    const bool bVec = B.size() == 1;
    if (aVec) A.insert(A.begin(), 1);
    if (bVec) B.push_back(1);

    const int64_t M = A[A.size() - 2];
    const int64_t Ka = A[A.size() - 1];
    const int64_t Kb = B[B.size() - 2];
    const int64_t Nn = B[B.size() - 1];
    if (Ka != Kb) {
        fail(err, n, "MatMul 内维不匹配");
        return false;
    }

    std::vector<int64_t> Abatch(A.begin(), A.end() - 2);
    std::vector<int64_t> Bbatch(B.begin(), B.end() - 2);
    std::vector<int64_t> batch;
    if (!broadcastShapes(Abatch, Bbatch, batch)) {
        fail(err, n, "MatMul batch 维无法广播");
        return false;
    }
    std::vector<int64_t> out = batch;
    out.push_back(M);
    out.push_back(Nn);
    if (aVec) out.erase(out.end() - 2);
    if (bVec) out.erase(out.end() - 1);
    return setOut(g, n, 0, out, inDtype(n, g, 0), err);
}

bool inferGemm(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* a = inShape(n, g, 0);
    const std::vector<int64_t>* b = inShape(n, g, 1);
    if (a == nullptr || b == nullptr || a->size() != 2 || b->size() != 2) {
        fail(err, n, "Gemm 期望 2 维输入");
        return false;
    }
    const bool transA = n.attrInt("transA", 0) != 0;
    const bool transB = n.attrInt("transB", 0) != 0;
    const int64_t M = transA ? (*a)[1] : (*a)[0];
    const int64_t N = transB ? (*b)[0] : (*b)[1];
    return setOut(g, n, 0, {M, N}, inDtype(n, g, 0), err);
}

bool inferSlice(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    if (xs == nullptr) {
        fail(err, n, "Slice 缺少输入 shape");
        return false;
    }
    std::vector<int64_t> starts, ends, axes, steps;
    if (inConstI64(n, g, 1)) starts = *inConstI64(n, g, 1);
    if (inConstI64(n, g, 2)) ends = *inConstI64(n, g, 2);
    if (inConstI64(n, g, 3)) axes = *inConstI64(n, g, 3);
    if (inConstI64(n, g, 4)) steps = *inConstI64(n, g, 4);
    if (starts.empty() || ends.empty()) {
        fail(err, n, "Slice 的 starts/ends 不是常量（老 opset 的 attribute 形式也未支持）");
        return false;
    }
    const int64_t r = static_cast<int64_t>(xs->size());
    if (axes.empty()) {
        for (int64_t i = 0; i < static_cast<int64_t>(starts.size()); ++i) axes.push_back(i);
    }
    if (steps.empty()) steps.assign(starts.size(), 1);

    std::vector<int64_t> out = *xs;
    for (size_t k = 0; k < axes.size(); ++k) {
        int64_t ax = axes[k] < 0 ? axes[k] + r : axes[k];
        if (ax < 0 || ax >= r) {
            fail(err, n, "Slice axes 越界");
            return false;
        }
        const int64_t dim = (*xs)[ax];
        const int64_t step = steps[k];
        if (step == 0) {
            fail(err, n, "Slice step 不能为 0");
            return false;
        }
        int64_t s = starts[k];
        int64_t e = ends[k];
        if (s < 0) s += dim;
        if (e < 0) e += dim;
        if (step > 0) {
            s = std::max<int64_t>(0, std::min(s, dim));
            e = std::max<int64_t>(0, std::min(e, dim));
            out[ax] = e > s ? ceilDiv(e - s, step) : 0;
        } else {
            s = std::max<int64_t>(0, std::min(s, dim - 1));
            e = std::max<int64_t>(-1, std::min(e, dim - 1));
            out[ax] = s > e ? ceilDiv(s - e, -step) : 0;
        }
    }
    return setOut(g, n, 0, out, inDtype(n, g, 0), err);
}

bool inferResize(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    if (xs == nullptr) {
        fail(err, n, "Resize 缺少输入 shape");
        return false;
    }
    // sizes 优先（有些导出会填全 0 表示不用）
    const std::vector<int64_t>* sizes = inConstI64(n, g, 3);
    if (sizes != nullptr && sizes->size() == xs->size()) {
        bool allZero = true;
        for (auto v : *sizes) {
            if (v != 0) allZero = false;
        }
        if (!allZero) return setOut(g, n, 0, *sizes, inDtype(n, g, 0), err);
    }
    const std::vector<int64_t>* scales = inConstI64(n, g, 2);
    // scales 可能是 float 常量，这里两种都试
    if (n.inputs.size() > 2 && n.inputs[2] >= 0) {
        const TensorInfo& st = g.at(n.inputs[2]);
        if (st.isConst && st.constF32.size() == xs->size()) {
            std::vector<int64_t> out(xs->size());
            for (size_t i = 0; i < xs->size(); ++i) {
                out[i] = static_cast<int64_t>(std::floor(static_cast<double>((*xs)[i]) * st.constF32[i]));
            }
            return setOut(g, n, 0, out, inDtype(n, g, 0), err);
        }
    }
    if (scales != nullptr && scales->size() == xs->size()) {
        std::vector<int64_t> out(xs->size());
        for (size_t i = 0; i < xs->size(); ++i) out[i] = (*xs)[i] * (*scales)[i];
        return setOut(g, n, 0, out, inDtype(n, g, 0), err);
    }
    fail(err, n, "Resize 既没有可用的 sizes 也没有 scales");
    return false;
}

bool inferReduce(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    if (xs == nullptr) {
        fail(err, n, "Reduce 缺少输入 shape");
        return false;
    }
    std::vector<int64_t> axes;
    const std::vector<int64_t>* c = inConstI64(n, g, 1);
    if (c != nullptr && !c->empty()) axes = *c;
    else axes = n.attrInts("axes");
    const bool keepdims = n.attrInt("keepdims", 1) != 0;
    const int64_t r = static_cast<int64_t>(xs->size());

    std::vector<int64_t> out = *xs;
    if (axes.empty()) {
        if (keepdims) {
            for (auto& d : out) d = 1;
        } else {
            out.clear();
        }
    } else {
        for (auto& a : axes) {
            int64_t ax = a < 0 ? a + r : a;
            if (ax < 0 || ax >= r) {
                fail(err, n, "Reduce axes 越界");
                return false;
            }
            if (keepdims) out[ax] = 1;
            else out[ax] = 0;  // 标记待删
        }
        if (!keepdims) {
            std::vector<int64_t> filtered;
            for (auto d : out) {
                if (d != 0) filtered.push_back(d);
            }
            out = filtered;
        }
    }
    return setOut(g, n, 0, out, inDtype(n, g, 0), err);
}

bool inferArgMax(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    if (xs == nullptr) {
        fail(err, n, "ArgMax 缺少输入 shape");
        return false;
    }
    int64_t axis = n.attrInt("axis", 0);
    const int64_t r = static_cast<int64_t>(xs->size());
    if (axis < 0) axis += r;
    const bool keepdims = n.attrInt("keepdims", 1) != 0;
    std::vector<int64_t> out = *xs;
    if (keepdims) out[axis] = 1;
    else out.erase(out.begin() + axis);
    return setOut(g, n, 0, out, DType::I64, err);
}

bool inferTopK(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    const std::vector<int64_t>* kt = inConstI64(n, g, 1);
    if (xs == nullptr || kt == nullptr || kt->empty()) {
        fail(err, n, "TopK 缺少输入或 K 不是常量");
        return false;
    }
    int64_t axis = n.attrInt("axis", -1);
    const int64_t r = static_cast<int64_t>(xs->size());
    if (axis < 0) axis += r;
    std::vector<int64_t> out = *xs;
    out[axis] = (*kt)[0];
    if (!setOut(g, n, 0, out, inDtype(n, g, 0), err)) return false;
    return setOut(g, n, 1, out, DType::I64, err);
}

bool inferGatherElements(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* idx = inShape(n, g, 1);
    if (idx == nullptr) {
        fail(err, n, "GatherElements 缺少 indices shape");
        return false;
    }
    return setOut(g, n, 0, *idx, inDtype(n, g, 0), err);
}

// Tile：各维分别放大 repeats[i] 倍
bool inferTile(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    const std::vector<int64_t>* reps = inConstI64(n, g, 1);
    if (xs == nullptr || reps == nullptr) {
        fail(err, n, "Tile 缺少输入或 repeats 不是常量");
        return false;
    }
    if (reps->size() != xs->size()) {
        fail(err, n, "Tile 的 repeats 长度与输入 rank 不一致");
        return false;
    }
    std::vector<int64_t> out(xs->size());
    for (size_t i = 0; i < xs->size(); ++i) {
        out[i] = (*xs)[i] * (*reps)[i];
    }
    return setOut(g, n, 0, out, inDtype(n, g, 0), err);
}

// Gather（通用轴）：
//   out.shape = data.shape[0..axis-1] + indices.shape + data.shape[axis+1..]
bool inferGather(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* ds = inShape(n, g, 0);
    const std::vector<int64_t>* is = inShape(n, g, 1);
    if (ds == nullptr || is == nullptr) {
        fail(err, n, "Gather 缺少输入 shape");
        return false;
    }
    const int64_t r = static_cast<int64_t>(ds->size());
    int64_t axis = n.attrInt("axis", 0);
    if (axis < 0) axis += r;
    if (axis < 0 || axis >= r) {
        fail(err, n, "Gather axis 越界");
        return false;
    }
    std::vector<int64_t> out;
    out.insert(out.end(), ds->begin(), ds->begin() + axis);
    out.insert(out.end(), is->begin(), is->end());
    out.insert(out.end(), ds->begin() + axis + 1, ds->end());
    return setOut(g, n, 0, out, inDtype(n, g, 0), err);
}

bool inferExpand(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    const std::vector<int64_t>* target = inConstI64(n, g, 1);
    if (xs == nullptr || target == nullptr) {
        fail(err, n, "Expand 缺少输入或 shape 不是常量");
        return false;
    }
    std::vector<int64_t> out;
    if (!broadcastShapes(*xs, *target, out)) {
        fail(err, n, "Expand 无法广播到目标 shape");
        return false;
    }
    return setOut(g, n, 0, out, inDtype(n, g, 0), err);
}

bool inferCast(const Node& n, Graph& g, std::string* err) {
    const std::vector<int64_t>* xs = inShape(n, g, 0);
    if (xs == nullptr) {
        fail(err, n, "Cast 缺少输入 shape");
        return false;
    }
    const int to = static_cast<int>(n.attrInt("to", ONNX_FLOAT));
    return setOut(g, n, 0, *xs, dtypeFromOnnx(to), err);
}

}  // namespace

// ---------------------------------------------------------------------------

bool inferNodeShape(const Node& n, Graph& g, std::string* err) {
    const std::string& op = n.opType;

    // 1) 逐算子规则
    if (op == "Conv") return inferConv(n, g, err);
    if (op == "ConvTranspose") return inferConvTranspose(n, g, err);
    if (op == "MaxPool" || op == "AveragePool") return inferPool(n, g, err);
    if (op == "GlobalAveragePool" || op == "GlobalMaxPool") {
        const std::vector<int64_t>* xs = inShape(n, g, 0);
        if (xs == nullptr) {
            fail(err, n, "GlobalPool 缺少输入 shape");
            return false;
        }
        std::vector<int64_t> out = *xs;
        for (size_t i = 2; i < out.size(); ++i) out[i] = 1;
        return setOut(g, n, 0, out, inDtype(n, g, 0), err);
    }
    if (op == "Add" || op == "Sub" || op == "Mul" || op == "Div" || op == "Pow" ||
        op == "Mod" || op == "Max" || op == "Min" || op == "PRelu") {
        return inferBinary(n, g, err);
    }
    if (op == "Concat") return inferConcat(n, g, err);
    if (op == "Split") return inferSplit(n, g, err);
    if (op == "Reshape") return inferReshape(n, g, err);
    if (op == "Flatten") return inferFlatten(n, g, err);
    if (op == "Unsqueeze") return inferUnsqueeze(n, g, err);
    if (op == "Squeeze") return inferSqueeze(n, g, err);
    if (op == "Transpose") return inferTranspose(n, g, err);
    if (op == "MatMul") return inferMatMul(n, g, err);
    if (op == "Gemm") return inferGemm(n, g, err);
    if (op == "Slice") return inferSlice(n, g, err);
    if (op == "Resize" || op == "Upsample") return inferResize(n, g, err);
    if (op == "ReduceMax" || op == "ReduceMin" || op == "ReduceMean" || op == "ReduceSum" ||
        op == "ReduceProd") {
        return inferReduce(n, g, err);
    }
    if (op == "ArgMax" || op == "ArgMin") return inferArgMax(n, g, err);
    if (op == "TopK") return inferTopK(n, g, err);
    if (op == "GatherElements") return inferGatherElements(n, g, err);
    if (op == "Gather") return inferGather(n, g, err);
    if (op == "Tile") return inferTile(n, g, err);
    if (op == "Expand") return inferExpand(n, g, err);
    if (op == "Cast") return inferCast(n, g, err);

    // 2) 形状不变的算子
    if (op == "Sigmoid" || op == "Tanh" || op == "Relu" || op == "Exp" || op == "Log" ||
        op == "Sqrt" || op == "Cos" || op == "Sin" || op == "Neg" || op == "Abs" ||
        op == "Softmax" || op == "Softplus" || op == "Erf" || op == "Floor" || op == "Ceil" ||
        op == "Identity" || op == "Clip" || op == "LeakyRelu" || op == "Elu" || op == "Not") {
        if (op == "Softmax") {
            if (inConstI64(n, g, 1) == nullptr && n.inputs.size() > 1 && n.inputs[1] >= 0) {
                // yolo26n 的 Softmax 带 axis 输入，形状不变，直接过
            }
        }
        return setAllOutsSameAsInput0(n, g, err);
    }

    fail(err, n, "未支持的算子（形状推导）");
    return false;
}

}  // namespace onnxcl

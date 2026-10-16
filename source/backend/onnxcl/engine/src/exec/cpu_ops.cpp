// CPU 回退算子实现
//
// 【为什么这些算子走 CPU 而不是 GPU】
//   1. 张量都很小：TopK/Expand/Tile 最大 24K 元素、Gather 只有 300 个，
//      GatherElements 最大 672K（2.6MB）—— 一次 host<->device 往返的量级
//   2. 大量用 int64 索引：ONNX 里 TopK/Gather/GatherElements 的索引都是 int64，
//      在 GPU 上要么走 int64 内存访问（AMD 上很慢），要么自己截断成 int32
//      （有溢出风险）。CPU 上这些都不是问题。
//   3. 这些算子在后处理子图里，只占 GPU 时间的 <1%（实测 profile）
//
// 代价是每帧多几次 device<->host 同步（每次几微秒），完全被上面省下的复杂度抵消。
//
// 【一个已知的简化】引擎内部所有张量统一按 float32 存（包括 ONNX 里声明为
// int64 的索引张量）。对本批模型是安全的：索引值都远小于 2^24，float32 能精确
// 表示。但如果有模型的索引会超过 2^24，这里就会出错。
#include "onnxcl/executor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

#include "onnxcl/cl_runtime.h"

namespace onnxcl {

#define CPU_FAILF(...)                            \
    do {                                          \
        if (err != nullptr) {                     \
            char _b[1024];                        \
            std::snprintf(_b, sizeof(_b), __VA_ARGS__); \
            *err = _b;                            \
        }                                         \
        return false;                             \
    } while (0)

namespace {

// 行主序步长（元素为单位）
void cpuStrides(const std::vector<int64_t>& shape, std::vector<int64_t>& strides) {
    const size_t r = shape.size();
    strides.assign(r, 1);
    for (int i = static_cast<int>(r) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

int64_t cpuProd(const std::vector<int64_t>& shape, size_t from, size_t to) {
    int64_t p = 1;
    for (size_t i = from; i < to; ++i) p *= shape[i];
    return p;
}

// 把线性下标分解成 n-D 坐标
void cpuUnravel(int64_t lin, const std::vector<int64_t>& shape, std::vector<int64_t>& coord) {
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        coord[i] = lin % shape[i];
        lin /= shape[i];
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// 主机 <-> 设备搬运
// ---------------------------------------------------------------------------

bool Executor::readTensor(int idx, std::vector<float>& out, std::string* err) {
    const TensorInfo& t = g_.at(idx);
    const size_t cnt = static_cast<size_t>(t.numElements());
    out.resize(cnt == 0 ? 1 : cnt);
    if (cnt == 0) return true;
    if (!ClRuntime::instance().download(buf_[idx], out.data(), cnt * sizeof(float), err)) {
        return false;
    }
    return true;
}

bool Executor::writeTensor(int idx, const std::vector<float>& in, std::string* err) {
    const TensorInfo& t = g_.at(idx);
    const size_t cnt = static_cast<size_t>(t.numElements());
    if (in.size() != cnt) {
        CPU_FAILF("写张量 %s 元素数不符: 给了 %zu, 需要 %zu", t.name.c_str(), in.size(), cnt);
    }
    if (cnt == 0) return true;
    return ClRuntime::instance().upload(buf_[idx], in.data(), cnt * sizeof(float), err);
}

// ---------------------------------------------------------------------------
// TopK（opset 10+：K 是输入；模型里都是 axis=-1 / largest=1）
//   in : X, K
//   out: Values, Indices
// ---------------------------------------------------------------------------

bool Executor::opTopK(const Node& n, std::string* err) {
    const TensorInfo& x = g_.at(n.inputs[0]);

    // K：优先取输入[1] 的常量（opset 10+ 都是这样传）
    int64_t k = 0;
    if (n.inputs.size() > 1 && n.inputs[1] >= 0) {
        const TensorInfo& kt = g_.at(n.inputs[1]);
        if (!kt.constI64.empty()) {
            k = kt.constI64[0];
        } else if (!kt.constF32.empty()) {
            k = static_cast<int64_t>(kt.constF32[0]);
        }
    }
    if (k <= 0) {
        CPU_FAILF("TopK 的 K 必须是常量且 > 0: %s", n.name.c_str());
    }

    const int64_t r = static_cast<int64_t>(x.shape.size());
    int64_t axis = n.attrInt("axis", -1);
    if (axis < 0) axis += r;
    if (axis != r - 1) {
        // 模型里都是最后一维；其它轴等真需要了再补（实现不复杂，但没必要先写）
        CPU_FAILF("TopK 目前只支持最后一维 (axis=%lld): %s", static_cast<long long>(axis),
                  n.name.c_str());
    }
    const bool largest = n.attrInt("largest", 1) != 0;

    const int64_t rowLen = x.shape[axis];
    const int64_t rows = rowLen > 0 ? x.numElements() / rowLen : 0;
    if (k > rowLen) {
        CPU_FAILF("TopK 的 K=%lld 超过该维长度 %lld: %s", static_cast<long long>(k),
                  static_cast<long long>(rowLen), n.name.c_str());
    }

    std::vector<float> src;
    if (!readTensor(n.inputs[0], src, err)) return false;

    std::vector<float> vals(static_cast<size_t>(rows * k));
    std::vector<float> idx(static_cast<size_t>(rows * k));
    std::vector<int> order(static_cast<size_t>(rowLen));

    for (int64_t ri = 0; ri < rows; ++ri) {
        const float* p = src.data() + ri * rowLen;
        for (int64_t i = 0; i < rowLen; ++i) order[static_cast<size_t>(i)] = static_cast<int>(i);
        // partial_sort：只要前 K 个，比全排快。
        //
        // 【并列的 tie-break】分数完全相同时排序结果是实现相关的。模型里低置信度
        // 区域会有大量相等的分数（实测 yolo26n 有一组分重复了 200 次），
        // 不固定的话同一模型两次跑都可能得到不同结果。这里统一按「索引升序」
        // 作为次关键字，保证同一输入永远得到同一输出。
        std::partial_sort(order.begin(), order.begin() + k, order.end(),
                          [&](int a, int b) {
                              if (p[a] != p[b]) {
                                  return largest ? (p[a] > p[b]) : (p[a] < p[b]);
                              }
                              return a < b;
                          });
        for (int64_t i = 0; i < k; ++i) {
            const int oi = order[static_cast<size_t>(i)];
            vals[static_cast<size_t>(ri * k + i)] = p[oi];
            idx[static_cast<size_t>(ri * k + i)] = static_cast<float>(oi);
        }
    }

    if (!writeTensor(n.outputs[0], vals, err)) return false;
    // 索引张量在 ONNX 里是 int64，但引擎内部统一按 float32 存
    // （见文件头的说明），下游 GatherElements 再转回整数
    return writeTensor(n.outputs[1], idx, err);
}

// ---------------------------------------------------------------------------
// GatherElements
//   out[i0..][j0..][i_{axis+1}..] = data[..., indices[...], ...]
// 即：输出与 indices 同形，只有 axis 那一维的坐标换成 indices 的值
// ---------------------------------------------------------------------------

bool Executor::opGatherElements(const Node& n, std::string* err) {
    const TensorInfo& d = g_.at(n.inputs[0]);
    const TensorInfo& ix = g_.at(n.inputs[1]);
    const int64_t r = static_cast<int64_t>(d.shape.size());
    int64_t axis = n.attrInt("axis", 0);
    if (axis < 0) axis += r;
    if (axis < 0 || axis >= r) {
        CPU_FAILF("GatherElements axis 越界: %s", n.name.c_str());
    }

    // 按 outer / mid / inner 三段切分，避免逐元素的除法
    const int64_t outer = cpuProd(ix.shape, 0, static_cast<size_t>(axis));
    const int64_t mid = ix.shape[static_cast<size_t>(axis)];
    const int64_t inner = cpuProd(ix.shape, static_cast<size_t>(axis) + 1, ix.shape.size());
    const int64_t dAxisLen = d.shape[static_cast<size_t>(axis)];

    std::vector<float> data, idxF, out;
    if (!readTensor(n.inputs[0], data, err)) return false;
    if (!readTensor(n.inputs[1], idxF, err)) return false;
    out.resize(idxF.size());

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t m = 0; m < mid; ++m) {
            for (int64_t in = 0; in < inner; ++in) {
                const int64_t outPos = (o * mid + m) * inner + in;
                int64_t ii = static_cast<int64_t>(idxF[static_cast<size_t>(outPos)]);
                if (ii < 0) ii += dAxisLen;  // ONNX 允许负索引
                if (ii < 0 || ii >= dAxisLen) {
                    CPU_FAILF("GatherElements 索引 %lld 越界 [0,%lld): %s",
                              static_cast<long long>(ii), static_cast<long long>(dAxisLen),
                              n.name.c_str());
                }
                out[static_cast<size_t>(outPos)] =
                    data[static_cast<size_t>((o * dAxisLen + ii) * inner + in)];
            }
        }
    }
    return writeTensor(n.outputs[0], out, err);
}

// ---------------------------------------------------------------------------
// Gather（通用轴）
//   output 形状 = data.shape[0..axis-1] + indices.shape + data.shape[axis+1..]
// ---------------------------------------------------------------------------

bool Executor::opGather(const Node& n, std::string* err) {
    const TensorInfo& d = g_.at(n.inputs[0]);
    const TensorInfo& ix = g_.at(n.inputs[1]);
    const int64_t r = static_cast<int64_t>(d.shape.size());
    int64_t axis = n.attrInt("axis", 0);
    if (axis < 0) axis += r;
    if (axis < 0 || axis >= r) {
        CPU_FAILF("Gather axis 越界: %s", n.name.c_str());
    }

    const int64_t outer = cpuProd(d.shape, 0, static_cast<size_t>(axis));
    const int64_t dAxisLen = d.shape[static_cast<size_t>(axis)];
    const int64_t inner = cpuProd(d.shape, static_cast<size_t>(axis) + 1, d.shape.size());
    const int64_t numIdx = ix.numElements();

    std::vector<float> data, idxF, out;
    if (!readTensor(n.inputs[0], data, err)) return false;
    if (!readTensor(n.inputs[1], idxF, err)) return false;
    out.resize(static_cast<size_t>(outer * numIdx * inner));

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t k = 0; k < numIdx; ++k) {
            int64_t ii = static_cast<int64_t>(idxF[static_cast<size_t>(k)]);
            if (ii < 0) ii += dAxisLen;
            if (ii < 0 || ii >= dAxisLen) {
                CPU_FAILF("Gather 索引 %lld 越界 [0,%lld): %s", static_cast<long long>(ii),
                          static_cast<long long>(dAxisLen), n.name.c_str());
            }
            const float* src = data.data() + (o * dAxisLen + ii) * inner;
            float* dst = out.data() + (o * numIdx + k) * inner;
            std::memcpy(dst, src, static_cast<size_t>(inner) * sizeof(float));
        }
    }
    return writeTensor(n.outputs[0], out, err);
}

// ---------------------------------------------------------------------------
// Expand：把 input 广播到目标形状（沿长度为 1 的维复制）
// ---------------------------------------------------------------------------

bool Executor::opExpand(const Node& n, std::string* err) {
    const TensorInfo& x = g_.at(n.inputs[0]);
    const TensorInfo& o = g_.at(n.outputs[0]);

    std::vector<float> src, out;
    if (!readTensor(n.inputs[0], src, err)) return false;

    const size_t rOut = o.shape.size();
    std::vector<int64_t> inPad(rOut, 1);
    const size_t off = rOut - x.shape.size();
    for (size_t i = 0; i < x.shape.size(); ++i) inPad[off + i] = x.shape[i];

    // 广播语义：源张量长度为 1 的维，步长当作 0（等价于沿该维重复）
    std::vector<int64_t> nat, srcStride(rOut, 0);
    cpuStrides(inPad, nat);
    for (size_t i = 0; i < rOut; ++i) {
        srcStride[i] = (inPad[i] == o.shape[i]) ? nat[i] : 0;
    }

    out.resize(static_cast<size_t>(o.numElements()));
    std::vector<int64_t> coord(rOut);
    for (size_t lin = 0; lin < out.size(); ++lin) {
        cpuUnravel(static_cast<int64_t>(lin), o.shape, coord);
        int64_t si = 0;
        for (size_t i = 0; i < rOut; ++i) si += coord[i] * srcStride[i];
        out[lin] = src[static_cast<size_t>(si)];
    }
    return writeTensor(n.outputs[0], out, err);
}

// ---------------------------------------------------------------------------
// Tile：沿各维重复 N 次
// ---------------------------------------------------------------------------

bool Executor::opTile(const Node& n, std::string* err) {
    const TensorInfo& x = g_.at(n.inputs[0]);
    const TensorInfo& o = g_.at(n.outputs[0]);

    std::vector<float> src, out;
    if (!readTensor(n.inputs[0], src, err)) return false;

    const size_t r = x.shape.size();
    std::vector<int64_t> nat;
    cpuStrides(x.shape, nat);

    out.resize(static_cast<size_t>(o.numElements()));
    std::vector<int64_t> coord(r);
    for (size_t lin = 0; lin < out.size(); ++lin) {
        cpuUnravel(static_cast<int64_t>(lin), o.shape, coord);
        int64_t si = 0;
        for (size_t i = 0; i < r; ++i) {
            si += (coord[i] % x.shape[i]) * nat[i];  // 取模即重复
        }
        out[lin] = src[static_cast<size_t>(si)];
    }
    return writeTensor(n.outputs[0], out, err);
}

// ---------------------------------------------------------------------------
// ArgMax（axis + keepdims）
//   输出是 int64 索引，引擎内部按 float32 存
// ---------------------------------------------------------------------------

bool Executor::opArgMax(const Node& n, std::string* err) {
    const TensorInfo& x = g_.at(n.inputs[0]);
    const int64_t r = static_cast<int64_t>(x.shape.size());
    int64_t axis = n.attrInt("axis", 0);
    if (axis < 0) axis += r;
    if (axis < 0 || axis >= r) {
        CPU_FAILF("ArgMax axis 越界: %s", n.name.c_str());
    }
    const bool keepdims = n.attrInt("keepdims", 1) != 0;
    const bool lastIndex = n.attrInt("select_last_index", 0) != 0;

    const int64_t outer = cpuProd(x.shape, 0, static_cast<size_t>(axis));
    const int64_t axisLen = x.shape[static_cast<size_t>(axis)];
    const int64_t inner = cpuProd(x.shape, static_cast<size_t>(axis) + 1, x.shape.size());

    std::vector<float> src;
    if (!readTensor(n.inputs[0], src, err)) return false;

    std::vector<float> out(static_cast<size_t>(outer * inner));

    // 外层按 (outer, inner) 遍历、内层扫 axis —— 这样对输入是顺序访问
    // （反过来的话每个输出元素都要跨 axisLen 个长度为 inner 的块跳读，cache 全废）
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t in = 0; in < inner; ++in) {
            const int64_t base = o * axisLen * inner + in;
            float best = src[static_cast<size_t>(base)];
            int64_t bestIdx = 0;
            for (int64_t a = 1; a < axisLen; ++a) {
                const float v = src[static_cast<size_t>(base + a * inner)];
                // select_last_index=0 时严格大于才换（保留第一个最大值）
                const bool better = lastIndex ? (v >= best) : (v > best);
                if (better) {
                    best = v;
                    bestIdx = a;
                }
            }
            out[static_cast<size_t>(o * inner + in)] = static_cast<float>(bestIdx);
        }
    }
    return writeTensor(n.outputs[0], out, err);
}

// ---------------------------------------------------------------------------
// ConvTranspose：实现见 executor.cpp（它走 GPU，不是 CPU 回退算子）
// ---------------------------------------------------------------------------

}  // namespace onnxcl

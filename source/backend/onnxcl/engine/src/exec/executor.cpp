#include "onnxcl/executor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "onnxcl/cl_runtime.h"

namespace onnxcl {

#define FAILF(...)                              \
    do {                                        \
        if (err != nullptr) {                   \
            char _buf[1024];                    \
            std::snprintf(_buf, sizeof(_buf), __VA_ARGS__); \
            *err = _buf;                        \
        }                                       \
        return false;                           \
    } while (0)

// ---------------------------------------------------------------------------
// 形状小工具：全部统一到 4 维（左补 1），步长以「元素」为单位
// ---------------------------------------------------------------------------

static void padShape4(const std::vector<int64_t>& s, int out[4]) {
    for (int i = 0; i < 4; ++i) out[i] = 1;
    const int off = 4 - static_cast<int>(s.size());
    for (size_t i = 0; i < s.size(); ++i) out[off + i] = static_cast<int>(s[i]);
}

// 连续布局的自然步长
static void naturalStride4(const std::vector<int64_t>& s, int out[4]) {
    int p[4];
    padShape4(s, p);
    out[3] = 1;
    for (int i = 2; i >= 0; --i) out[i] = out[i + 1] * p[i + 1];
}

static long totalOf(const std::vector<int64_t>& s) {
    long n = 1;
    for (auto d : s) n *= static_cast<long>(d);
    return n;
}

// 二元算子：算输出形状 + 两侧的 4 维广播步长（被广播的维步长给 0）
static void prepBinary4(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
                        const std::vector<int64_t>& o, int outShape[4], int aSt[4], int bSt[4]) {
    int ap[4], bp[4], op[4], aNat[4], bNat[4];
    padShape4(a, ap);
    padShape4(b, bp);
    padShape4(o, op);
    naturalStride4(a, aNat);
    naturalStride4(b, bNat);
    for (int i = 0; i < 4; ++i) {
        outShape[i] = op[i];
        aSt[i] = (ap[i] == op[i]) ? aNat[i] : 0;
        bSt[i] = (bp[i] == op[i]) ? bNat[i] : 0;
    }
}

static bool isViewOp(const std::string& op) {
    // Alias 是融合 pass 产物（Sigmoid/Mul 被融进前面的 Conv 之后剩下的空壳）：
    // 输出直接复用输入的显存，不分配内存、不启动内核。
    //
    // Cast 也算视图：引擎内部所有张量统一按 float32 存，没有真正的类型转换，
    // 所以 Cast 只是把数据原样交给下游（详见 cpu_ops.cpp 文件头的说明）。
    return op == "Reshape" || op == "Flatten" || op == "Unsqueeze" || op == "Squeeze" ||
           op == "Identity" || op == "Alias" || op == "Cast";
}

// 一元算子名 -> op code（要和 elementwise.cl 里的 switch 对齐）
static int unaryCode(const std::string& op) {
    if (op == "Sigmoid") return 0;
    if (op == "Exp") return 1;
    if (op == "Sqrt") return 2;
    if (op == "Log") return 3;
    if (op == "Cos") return 4;
    if (op == "Sin") return 5;
    if (op == "Tanh") return 6;
    if (op == "Relu") return 7;
    if (op == "Abs") return 8;
    if (op == "Neg") return 9;
    if (op == "Floor") return 10;
    if (op == "Ceil") return 11;
    if (op == "Erf") return 12;
    if (op == "SiLU" || op == "Swish") return 13;
    if (op == "Relu6") return 14;
    if (op == "Sinh") return 15;
    if (op == "Cosh") return 16;
    return -1;
}

static int binaryCode(const std::string& op) {
    if (op == "Add") return 0;
    if (op == "Sub") return 1;
    if (op == "Mul") return 2;
    if (op == "Div") return 3;
    if (op == "Pow") return 4;
    if (op == "Max") return 5;
    if (op == "Min") return 6;
    if (op == "Mod") return 7;
    return -1;
}

// 小通道卷积内核一次能装完的归约维上限。
// 【必须和 conv.cl 里的 CS_BK 保持一致】，否则会落到多轮 chunk 的逻辑上
const int kConvSmallBK = 32;

// 卷积内核选择
enum class ConvKernel {
    Naive,      // 幼稚版：depthwise / 小到不值得分块的情况
    Gemm,       // 大分块 GEMM (BM=64)：主路径，适合 C 大
    GemmSmall,  // 小分块 GEMM (BM=16，归约维一次装完)：适合 cin 极小，如首层 cin=3
};

// 选核。
//
// init（决定要不要重排它的权重）和 opConv（决定启动哪个内核）必须用同一套判据，
// 否则会出现「该层要走 GEMM 但权重没重排」这种低级错误，所以抽成独立函数。
ConvKernel chooseConvKernel(const Node& n, const Graph& g) {
    if (n.opType != "Conv" || n.inputs.size() < 2) return ConvKernel::Naive;
    if (n.inputs[0] < 0 || n.inputs[1] < 0 || n.outputs.empty() || n.outputs[0] < 0) {
        return ConvKernel::Naive;
    }
    const TensorInfo& x = g.at(n.inputs[0]);
    const TensorInfo& w = g.at(n.inputs[1]);
    const TensorInfo& o = g.at(n.outputs[0]);
    if (x.shape.size() != 4 || w.shape.size() != 4 || o.shape.size() != 4) {
        return ConvKernel::Naive;
    }
    if (n.attrInt("group", 1) != 1) return ConvKernel::Naive;
    const std::vector<int64_t> d = n.attrInts("dilations");
    if (d.size() >= 2 && (d[0] != 1 || d[1] != 1)) return ConvKernel::Naive;

    const int kOut = static_cast<int>(o.shape[1]);
    const int cIn = static_cast<int>(x.shape[1]);
    if (kOut <= 0 || cIn <= 0) return ConvKernel::Naive;

    std::vector<int64_t> k = n.attrInts("kernel_shape");
    if (k.size() < 2) k = {w.shape[2], w.shape[3]};
    const int ktot = static_cast<int>(cIn * k[0] * k[1]);

    // ---- 两种分块的浪费比 ----
    //   大分块 GEMM : BM=64, BK=16
    //   小分块 GEMM : BM=16, BK=32（归约维一次装完）
    const int bigMBlocks = (kOut + 63) / 64;
    const int bigCChunks = (cIn + 15) / 16;
    const double bigWaste = (static_cast<double>(bigMBlocks * 64) / kOut) *
                            (static_cast<double>(bigCChunks * 16) / cIn);

    const int smMBlocks = (kOut + 15) / 16;
    const int smChunks = (ktot + kConvSmallBK - 1) / kConvSmallBK;
    const double smallWaste = (static_cast<double>(smMBlocks * 16) / kOut) *
                              (static_cast<double>(smChunks * kConvSmallBK) / ktot);

    // ---- 小分块只在「明显赚」的时候用 ----
    //
    // 三重条件：
    //   kOut >= 16 : 否则 BM=16 本身就在浪费（DFL 那层 cout=1，两种 GEMM 都不如幼稚版）
    //   kOut <= 128: 输出通道太多时根本用不着小分块
    //   smallWaste * 2 < bigWaste : 必须有大分块两倍以上的浪费优势才值得换 ——
    //       两个内核 work-group 尺寸不同（64 vs 256），小分块每 work-group 还要多填一张 k 解码表。
    if (kOut >= 16 && kOut <= 128 && smallWaste * 2.0 < bigWaste) {
        return ConvKernel::GemmSmall;
    }

    // 【已试过的失败方案，别重蹈】曾加过一条「并行度不足就换小分块」的判据：
    // 当大分块的 work-group 数 < 64（比如 [1,256,20,20] 只有 7 个）时改用小分块，
    // 想把通道方向多出的 4 倍分块拿来换并行度。
    // 实测【慢了 60%】（Conv 13.19ms -> 21.19ms）。原因是小分块的 work-group
    // 只有 64 线程，每个线程要扛 32 次 B 子块加载（大分块只要 4 次），
    // 而 B 子块的 im2col gather 每元素都有索引计算 + 边界检查 ——
    // 加载开销远超并行度带来的收益。
    //
    // 小空间尺寸层真正的问题是【work-group 太少】与【每 work-group 的固定开销
    // （loaders + barrier）占比高】的叠加，正确解法是减小固定开销（比如把输入
    // 驻留到 local、或改成不依赖 im2col 的直接卷积），而不是单纯改分块尺寸。

    // ---- 大分块 GEMM ----
    // BM=64 对 K=16 浪费 4 倍，但 GEMM 的绝对吞吐比幼稚版高一个数量级，
    // 浪费 4 倍仍然大幅胜出；浪费超过 12 倍就留给幼稚版。
    return (bigWaste <= 12.0) ? ConvKernel::Gemm : ConvKernel::Naive;
}

// 是否需要重排权重（两种 GEMM 共用同一套打包布局 [KH*KW][C][K]）
bool convNeedsPackedWeight(const Node& n, const Graph& g) {
    const ConvKernel kc = chooseConvKernel(n, g);
    return kc == ConvKernel::Gemm || kc == ConvKernel::GemmSmall;
}

// 把 ONNX 的权重 [K][C][KH*KW] 重排成 [KH*KW][C][K]。
// 目的是让 GEMM 的 A 子块全局读取沿 k 方向连续 —— 原来的布局下相邻线程的
// 地址相差 C*KH*KW 个 float，每个线程一个 cache line，完全不合并。
void packConvWeightHost(const std::vector<float>& src, int kOut, int cIn, int rs,
                        std::vector<float>& dst) {
    dst.assign(static_cast<size_t>(kOut) * cIn * rs, 0.0f);
    for (int r = 0; r < rs; ++r) {
        for (int c = 0; c < cIn; ++c) {
            for (int k = 0; k < kOut; ++k) {
                dst[(static_cast<size_t>(r) * cIn + c) * kOut + k] =
                    src[(static_cast<size_t>(k) * cIn + c) * rs + r];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 构造 / 析构
// ---------------------------------------------------------------------------

Executor::Executor(const Graph& g) : g_(g) {}

Executor::~Executor() {
    for (auto& kv : rootBuf_) ClRuntime::instance().release(kv.second);
    for (auto& kv : packedWeight_) ClRuntime::instance().release(kv.second);
    for (auto& kv : scratchPool_) ClRuntime::instance().release(kv.second);
    for (size_t i = 0; i < buf_.size(); ++i) {
        // 单独上传的 const 张量
        if (owned_[i] != 0 && buf_[i] != nullptr) ClRuntime::instance().release(buf_[i]);
    }
}

// 取一个至少 bytes 字节的临时缓冲（按尺寸缓存，稳态下不会真的分配）
//
// 【所有权契约 —— 之前在这里迹过一个很隐蔽的 bug】
//   返回的 cl_mem 【归缓冲池所有】，调用方【不要】自己 clRelease。
//   池子会在 Executor 析构时统一释放。
//
//   之前的写法是调用方用完调 rt.release(mid)，但【忘了把 scratchPool_ 里的条目一起删掉】。
//   后果：第一次推理把缓冲释放了，池子里却还留着那个已失效的句柄；
//   第二次推理 acquireScratch 直接命中池子，把脏句柄当成可用缓冲返回
//   -> use-after-free -> 崩溃。
//   而且这也完全违背了缓冲池的初衷（“按尺寸缓存、稳态零分配”），
//   每次释放等于池子白建。
//   特征：带 ConvTranspose 的模型（seg/depth 系列）跑第二次必崩。
cl_mem Executor::acquireScratch(size_t bytes, std::string* err) {
    auto it = scratchPool_.find(bytes);
    if (it != scratchPool_.end()) {
        return it->second;
    }
    cl_mem m = ClRuntime::instance().alloc(bytes, err);
    if (m == nullptr) {
        return nullptr;
    }
    scratchPool_[bytes] = m;
    deviceBytes_ += bytes;
    return m;
}

int Executor::findRoot(int i) {
    while (aliasRoot_[i] != i) {
        aliasRoot_[i] = aliasRoot_[aliasRoot_[i]];
        i = aliasRoot_[i];
    }
    return i;
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

bool Executor::init(std::string* err, int deviceIndex) {
    if (initialized_) return true;
    ClRuntime& rt = ClRuntime::instance();
    if (!rt.init(err, deviceIndex)) return false;
    rt.setProfiling(profiling_);

    const size_t n = g_.tensors.size();
    buf_.assign(n, nullptr);
    owned_.assign(n, 0);

    // --- 1. 视图算子做 alias 归并（Reshape 等零成本）---
    aliasRoot_.resize(n);
    for (size_t i = 0; i < n; ++i) aliasRoot_[i] = static_cast<int>(i);
    for (const Node& nd : g_.nodes) {
        if (!isViewOp(nd.opType)) continue;
        if (nd.inputs.empty() || nd.outputs.empty()) continue;
        if (nd.inputs[0] < 0 || nd.outputs[0] < 0) continue;
        aliasRoot_[findRoot(nd.outputs[0])] = findRoot(nd.inputs[0]);
    }

    // --- 2. 上传常量（权重 + 各类常量张量）---
    for (size_t i = 0; i < n; ++i) {
        const TensorInfo& t = g_.tensors[i];
        if (!t.isConst) continue;
        size_t cnt = static_cast<size_t>(t.numElements());
        if (cnt == 0) cnt = 1;
        std::vector<float> data = t.constF32;
        if (data.size() < cnt) {
            // int64 常量在 tensorToFloat 里已经转过；到这里说明确实没有数据
            data.resize(cnt, 0.0f);
        }
        cl_mem m = rt.allocWithData(data.data(), cnt * sizeof(float), err);
        if (m == nullptr) {
            FAILF("上传常量失败: %s (%zu 元素)", t.name.c_str(), cnt);
        }
        buf_[i] = m;
        owned_[i] = 1;
        deviceBytes_ += cnt * sizeof(float);
    }

    // --- 3. 分配激活值缓冲（每个 alias 根一个）---
    for (size_t i = 0; i < n; ++i) {
        const TensorInfo& t = g_.tensors[i];
        if (t.isConst) continue;
        if (findRoot(static_cast<int>(i)) != static_cast<int>(i)) continue;
        if (rootBuf_.count(static_cast<int>(i)) != 0) continue;
        size_t bytes = t.bytes;
        if (bytes == 0) bytes = sizeof(float);
        cl_mem m = rt.alloc(bytes, err);
        if (m == nullptr) {
            FAILF("分配显存失败: %s (%zu 字节, 累计 %.1f MB)", t.name.c_str(), bytes,
                  deviceBytes_ / 1048576.0);
        }
        rootBuf_[static_cast<int>(i)] = m;
        deviceBytes_ += bytes;
    }

    // --- 4. 绑定：每个张量指向它的根缓冲 ---
    for (size_t i = 0; i < n; ++i) {
        if (g_.tensors[i].isConst) continue;
        const int root = findRoot(static_cast<int>(i));
        if (buf_[root] == nullptr) {
            auto it = rootBuf_.find(root);
            if (it == rootBuf_.end()) {
                FAILF("张量 %s 的根缓冲不存在", g_.tensors[i].name.c_str());
            }
            buf_[i] = it->second;
        } else {
            buf_[i] = buf_[root];  // 根是常量
        }
    }

    // --- 5. 给走 GEMM 的 Conv 重排权重 ---
    // ONNX 的权重是 [K][C][KH*KW]，GEMM 需要 [KH*KW][C][K] 才能让 A 子块的
    // 全局读取沿 k 连续（完全合并）。这只是一次性的内存拷贝，
    // onnx 文件本身依然是直读的，没有格式转换步骤。
    {
        int nPacked = 0;
        for (size_t i = 0; i < g_.nodes.size(); ++i) {
            const Node& n = g_.nodes[i];
            if (!convNeedsPackedWeight(n, g_)) continue;
            const TensorInfo& w = g_.at(n.inputs[1]);
            if (!w.isConst || w.constF32.empty()) continue;
            const int wK = static_cast<int>(w.shape[0]);
            const int wC = static_cast<int>(w.shape[1]);
            const int wRS = static_cast<int>(w.shape[2] * w.shape[3]);
            std::vector<float> packed;
            packConvWeightHost(w.constF32, wK, wC, wRS, packed);
            cl_mem m = rt.allocWithData(packed.data(), packed.size() * sizeof(float), err);
            if (m == nullptr) {
                FAILF("重排卷积权重失败: %s", n.name.c_str());
            }
            packedWeight_[static_cast<int>(i)] = m;
            deviceBytes_ += packed.size() * sizeof(float);
            nPacked++;
        }
        if (nPacked > 0) {
            std::fprintf(stderr, "[onnxcl] 重排了 %d 层卷积权重\n", nPacked);
        }
    }

    // --- 6. 预编译本次图用到的内核 ---
    // 提前编译能让首次推理不包含编译耗时，也方便一次性报出所有编译错误
    const char* needed[][2] = {
        {"conv", "conv2d"},
        {"conv", "conv2d_gemm"},
        {"conv", "conv2d_gemm_small"},
        {"elementwise", "binary_op"},
        {"elementwise", "unary_op"},
        {"elementwise", "clip_op"},
        {"copy", "copy_strided"},
        {"copy", "pixel_shuffle2x"},
        {"pool", "maxpool"},
        {"pool", "global_avgpool"},
        {"resize", "resize_nearest"},
        {"softmax", "softmax"},
        {"matmul", "batched_matmul"},
    };
    for (const auto& pair : needed) {
        if (rt.getKernel(pair[0], pair[1], err) == nullptr) return false;
    }
    // 内核都编译/加载完了，这时才能报告缓存命中情况
    rt.reportCacheStat();

    nodeProfile_.resize(g_.nodes.size());
    for (size_t i = 0; i < g_.nodes.size(); ++i) {
        NodeProfile& np = nodeProfile_[i];
        const Node& n = g_.nodes[i];
        np.name = n.name;
        np.opType = n.opType;
        if (!n.inputs.empty() && n.inputs[0] >= 0) {
            np.shape = shapeToString(g_.at(n.inputs[0]).shape) + " -> ";
        }
        if (!n.outputs.empty() && n.outputs[0] >= 0) {
            np.shape += shapeToString(g_.at(n.outputs[0]).shape);
        }
    }

    initialized_ = true;
    return true;
}

void Executor::setProfiling(bool on) {
    profiling_ = on;
    ClRuntime::instance().setProfiling(on);
}

void Executor::resetProfile() {
    profile_.clear();
    for (auto& np : nodeProfile_) {
        np.ms = 0.0;
        np.count = 0;
    }
}

void Executor::printSlowNodes(FILE* out, int limit) const {
    std::vector<const NodeProfile*> v;
    for (const auto& np : nodeProfile_) {
        if (np.ms > 0.0) v.push_back(&np);
    }
    std::sort(v.begin(), v.end(),
              [](const NodeProfile* a, const NodeProfile* b) { return a->ms > b->ms; });
    if (static_cast<int>(v.size()) > limit) v.resize(limit);

    std::fprintf(out, "\n最慢的 %zu 个节点:\n", v.size());
    std::fprintf(out, "%-14s %-24s %-26s %9s\n", "总耗时(ms)", "算子", "输入 -> 输出", "次数");
    std::fprintf(out, "%s\n", std::string(80, '-').c_str());
    for (const NodeProfile* np : v) {
        std::fprintf(out, "%-14.3f %-24s %-26s %9d\n", np->ms, np->opType.c_str(),
                     np->shape.c_str(), np->count);
    }
}

void Executor::printProfile(FILE* out) const {
    std::vector<std::pair<std::string, std::pair<int, double>>> v(profile_.begin(), profile_.end());
    std::sort(v.begin(), v.end(),
              [](const std::pair<std::string, std::pair<int, double>>& a,
                 const std::pair<std::string, std::pair<int, double>>& b) {
                  return a.second.second > b.second.second;
              });
    double total = 0;
    for (auto& kv : v) total += kv.second.second;
    std::fprintf(out, "\n%-16s %8s %12s %8s\n", "算子", "次数", "GPU(ms)", "占比");
    std::fprintf(out, "------------------------------------------------\n");
    for (auto& kv : v) {
        std::fprintf(out, "%-16s %8d %12.3f %7.1f%%\n", kv.first.c_str(), kv.second.first,
                     kv.second.second, 100.0 * kv.second.second / (total > 0 ? total : 1));
    }
    std::fprintf(out, "------------------------------------------------\n");
    std::fprintf(out, "%-16s %8s %12.3f\n", "合计", "", total);
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

bool Executor::run(const float* input, std::vector<std::vector<float>>& outputs, std::string* err) {
    if (!initialized_) {
        FAILF("Executor 还没 init");
    }
    ClRuntime& rt = ClRuntime::instance();

    const int inIdx = g_.inputs[0];
    const TensorInfo& it = g_.at(inIdx);
    const size_t inBytes = static_cast<size_t>(it.numElements()) * sizeof(float);
    if (!rt.upload(buf_[inIdx], input, inBytes, err)) return false;

    for (size_t i = 0; i < g_.nodes.size(); ++i) {
        if (!runNode(g_.nodes[i], static_cast<int>(i), err)) return false;
    }

    if (!rt.finish(err)) return false;

    outputs.resize(g_.outputs.size());
    for (size_t i = 0; i < g_.outputs.size(); ++i) {
        const TensorInfo& t = g_.at(g_.outputs[i]);
        const size_t cnt = static_cast<size_t>(t.numElements());
        outputs[i].resize(cnt);
        if (!rt.download(buf_[g_.outputs[i]], outputs[i].data(), cnt * sizeof(float), err)) {
            return false;
        }
    }
    return true;
}

bool Executor::runNode(const Node& n, int nodeIdx, std::string* err) {
    const double t0 = profiling_ ? ClRuntime::instance().totalKernelMs() : 0.0;

    const std::string& op = n.opType;
    bool ok = false;

    if (isViewOp(op) || op == "Constant") {
        // 【Constant 必须显式跳过】它在构图阶段就已经物化成了常量张量
        // （init 时把数据上传到显存），运行时不需要做任何事。
        // 但它仍然是图里的一个节点，会被派发到这里 —— 不跳过就会报
        // 「不支持算子 Constant」。
        //
        // 真实模型里很少踩到（那几个 YOLO 都用 initializer 传常量），
        // 但很多 ONNX 导出工具习惯用 Constant 节点，所以这是个通用性缺陷。
        ok = true;  // 已 alias / 已物化，什么都不用做
    } else if (op == "Conv") {
        ok = opConv(n, nodeIdx, err);
    } else if (binaryCode(op) >= 0) {
        ok = opBinary(n, op.c_str(), err);
    } else if (unaryCode(op) >= 0 || op == "Clip") {
        ok = opUnary(n, err);
    } else if (op == "Concat") {
        ok = opConcat(n, err);
    } else if (op == "Split") {
        ok = opSplit(n, err);
    } else if (op == "Slice") {
        ok = opSlice(n, err);
    } else if (op == "Transpose") {
        ok = opTranspose(n, err);
    } else if (op == "Resize" || op == "Upsample") {
        ok = opResize(n, err);
    } else if (op == "MaxPool" || op == "AveragePool") {
        ok = opPool(n, err);
    } else if (op == "Softmax") {
        ok = opSoftmax(n, err);
    } else if (op == "MatMul") {
        ok = opMatMul(n, err);
    } else if (op == "Gemm") {
        // Gemm 有 transB / 偏置 C / alpha / beta，不能复用 MatMul 的实现
        ok = opGemm(n, err);
    } else if (op == "GlobalAveragePool") {
        ok = opGlobalPool(n, err);
    } else if (op == "ReduceMax" || op == "ReduceMean" || op == "ReduceSum") {
        ok = opReduce(n, err);
    } else if (op == "ConvTranspose") {
        ok = opConvTranspose(n, nodeIdx, err);
    } else if (op == "TopK") {
        ok = opTopK(n, err);
    } else if (op == "GatherElements") {
        ok = opGatherElements(n, err);
    } else if (op == "Gather") {
        ok = opGather(n, err);
    } else if (op == "Expand") {
        ok = opExpand(n, err);
    } else if (op == "Tile") {
        ok = opTile(n, err);
    } else if (op == "ArgMax" || op == "ArgMin") {
        ok = opArgMax(n, err);
    } else {
        FAILF("执行器还不支持算子 %s (节点 %s)", op.c_str(), n.name.c_str());
    }

    if (profiling_) {
        const double dt = ClRuntime::instance().totalKernelMs() - t0;
        auto& p = profile_[op];
        p.first++;
        p.second += dt;
        if (nodeIdx >= 0 && nodeIdx < static_cast<int>(nodeProfile_.size())) {
            nodeProfile_[nodeIdx].ms += dt;
            nodeProfile_[nodeIdx].count++;
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// 各算子实现
// ---------------------------------------------------------------------------

bool Executor::opConv(const Node& n, int nodeIdx, std::string* err) {
    ClRuntime& rt = ClRuntime::instance();
    const TensorInfo& x = g_.at(n.inputs[0]);
    const TensorInfo& w = g_.at(n.inputs[1]);
    const TensorInfo& o = g_.at(n.outputs[0]);

    if (x.shape.size() != 4 || w.shape.size() != 4) {
        FAILF("Conv 只支持 4 维: %s", n.name.c_str());
    }
    std::vector<int64_t> k = n.attrInts("kernel_shape");
    if (k.size() < 2) k = {w.shape[2], w.shape[3]};
    std::vector<int64_t> s = n.attrInts("strides");
    if (s.size() < 2) s = {1, 1};
    std::vector<int64_t> p = n.attrInts("pads");
    if (p.size() < 4) p = {0, 0, 0, 0};
    std::vector<int64_t> d = n.attrInts("dilations");
    if (d.size() < 2) d = {1, 1};
    const int group = static_cast<int>(n.attrInt("group", 1));
    const bool hasBias = n.inputs.size() > 2 && n.inputs[2] >= 0;
    // 融合 pass 打的标记：0=无 1=SiLU 2=ReLU 3=ReLU6
    const int activation = static_cast<int>(n.attrInt("onnxcl_activation", 0));

    // ---- 选内核 ----
    // ONNXCL_CONV=naive 强制走幼稚版（对拍用），=gemm 强制走 GEMM，不设则自动
    static const int convMode = []() {
        const char* e = std::getenv("ONNXCL_CONV");
        if (e != nullptr && std::strcmp(e, "naive") == 0) return 1;
        if (e != nullptr && std::strcmp(e, "gemm") == 0) return 2;
        return 0;
    }();

    // 走不走 GEMM 由 shouldUseGemmConv 决定（init 用同一套判据决定要不要重排权重），
    // 这里只负责确认权重真的重排好了
    const int kOut = static_cast<int>(o.shape[1]);
    const int cIn = static_cast<int>(x.shape[1]);
    const bool packedOk = packedWeight_.count(nodeIdx) != 0;
    // 用哪个 GEMM 变体由 chooseConvKernel 决定（init 已按同一判据决定过是否重排权重）
    const bool smallKernel = (chooseConvKernel(n, g_) == ConvKernel::GemmSmall);
    bool useGemm;
    if (convMode == 1) {
        useGemm = false;  // 强制幼稚版（对拍用）
    } else if (convMode == 2) {
        useGemm = packedOk;  // 强制 GEMM
        if (!packedOk) {
            FAILF("ONNXCL_CONV=gemm 但该层没有重排权重: %s", n.name.c_str());
        }
    } else {
        useGemm = packedOk;  // 自动
    }

    if (useGemm) {
        const char* kname = smallKernel ? "conv2d_gemm_small" : "conv2d_gemm";
        // 两个内核都硬编码了 work-group 尺寸（大分块 16x16=256，小分块 4x16=64）；
        // 如果驱动允许的上限更小，自动缩容会静默算出错的结果 —— 宁可报错
        const size_t wgThreads = smallKernel ? 64 : 256;
        cl_kernel kern = rt.getKernel("conv", kname, err);
        if (kern == nullptr) return false;
        const size_t kmax = rt.kernelMaxWG(kern);
        if (kmax < wgThreads) {
            FAILF("%s 需要 work-group=%zu，但驱动只允许 %zu (%s)", kname, wgThreads, kmax,
                  n.name.c_str());
        }

        const int N = static_cast<int>(x.shape[0]);
        const int C = cIn;
        const int H = static_cast<int>(x.shape[2]);
        const int Wd = static_cast<int>(x.shape[3]);
        const int K = kOut;
        const int OH = static_cast<int>(o.shape[2]);
        const int OW = static_cast<int>(o.shape[3]);
        const long SP = static_cast<long>(N) * OH * OW;

        Kernel kk(kern, kname);
        kk.set(0, buf_[n.inputs[0]]);
        kk.set(1, packedWeight_[nodeIdx]);  // 重排后的权重 [KH*KW][C][K]
        kk.set(2, hasBias ? buf_[n.inputs[2]] : buf_[n.inputs[1]]);
        kk.set(3, buf_[n.outputs[0]]);
        kk.set(4, N);
        kk.set(5, C);
        kk.set(6, H);
        kk.set(7, Wd);
        kk.set(8, K);
        kk.set(9, OH);
        kk.set(10, OW);
        kk.set(11, static_cast<int>(k[0]));
        kk.set(12, static_cast<int>(k[1]));
        kk.set(13, static_cast<int>(s[0]));
        kk.set(14, static_cast<int>(s[1]));
        kk.set(15, static_cast<int>(p[0]));
        kk.set(16, static_cast<int>(p[1]));
        kk.set(17, hasBias ? 1 : 0);
        kk.set(18, activation);

        if (smallKernel) {
            // 小分块：BM=16, BN=64, TM=4, TN=4 -> work-group (4, 16) = 64 线程
            // grid: (空间块 x 4, 输出通道块 x 16)
            const size_t gwsSpatial = ((static_cast<size_t>(SP) + 63) / 64) * 4;
            const size_t gwsChannel = ((static_cast<size_t>(K) + 15) / 16) * 16;
            return kk.run(gwsSpatial, gwsChannel, 0, 4, 16, 0, err);
        }
        // 大分块：BM=64, BN=64, TM=4, TN=4 -> work-group (16, 16) = 256 线程
        // grid: (空间块 x 16, 输出通道块 x 16)
        const size_t gwsSpatial = ((static_cast<size_t>(SP) + 63) / 64) * 16;
        const size_t gwsChannel = ((static_cast<size_t>(K) + 63) / 64) * 16;
        return kk.run(gwsSpatial, gwsChannel, 0, 16, 16, 0, err);
    }

    // ---- 幼稚版 ----
    cl_kernel kern = rt.getKernel("conv", "conv2d", err);
    if (kern == nullptr) return false;

    const int inShape[4] = {static_cast<int>(x.shape[0]), static_cast<int>(x.shape[1]),
                            static_cast<int>(x.shape[2]), static_cast<int>(x.shape[3])};
    const int outShape[4] = {static_cast<int>(o.shape[0]), static_cast<int>(o.shape[1]),
                             static_cast<int>(o.shape[2]), static_cast<int>(o.shape[3])};
    const int ksize[2] = {static_cast<int>(k[0]), static_cast<int>(k[1])};
    const int strd[2] = {static_cast<int>(s[0]), static_cast<int>(s[1])};
    const int pad2[2] = {static_cast<int>(p[0]), static_cast<int>(p[1])};
    const int dil2[2] = {static_cast<int>(d[0]), static_cast<int>(d[1])};

    Kernel kk(kern, "conv2d");
    kk.set(0, buf_[n.inputs[0]]);
    kk.set(1, buf_[n.inputs[1]]);
    // 没有 bias 时也要传一个合法 buffer（OpenCL 不允许给 __global 传 NULL）
    kk.set(2, hasBias ? buf_[n.inputs[2]] : buf_[n.inputs[1]]);
    kk.set(3, buf_[n.outputs[0]]);
    kk.setRaw(4, sizeof(inShape), inShape);
    kk.setRaw(5, sizeof(outShape), outShape);
    kk.setRaw(6, sizeof(ksize), ksize);
    kk.setRaw(7, sizeof(strd), strd);
    kk.setRaw(8, sizeof(pad2), pad2);
    kk.setRaw(9, sizeof(dil2), dil2);
    kk.set(10, static_cast<int>(w.shape[1]));  // cinG
    kk.set(11, group);
    kk.set(12, hasBias ? 1 : 0);
    kk.set(13, activation);

    const size_t total = static_cast<size_t>(o.numElements());
    return kk.run(total, 0, 0, 256, 0, 0, err);
}

bool Executor::opBinary(const Node& n, const char* opName, std::string* err) {
    ClRuntime& rt = ClRuntime::instance();
    int code = binaryCode(opName);
    if (code < 0) FAILF("未知二元算子 %s", opName);

    // ONNX 的 Div 对整数输入做的是【整数除法】（向零截断），不是浮点除法。
    // 典型用途：`展平索引 / 类别数` 还原出 anchor 下标 —— 浮点除会得到 188.6
    // 这种非整数，下游再看一眼就不一样了。引擎内部统一按 f32 存，所以只能
    // 靠输入张量的 dtype 判断该走哪条路。
    if (code == 3) {  // Div
        const DType dt = g_.at(n.inputs[0]).dtype;
        if (dt != DType::F32 && dt != DType::F16) {
            code = 8;  // 整数除法
        }
    }

    const TensorInfo& a = g_.at(n.inputs[0]);
    const TensorInfo& b = g_.at(n.inputs[1]);
    const TensorInfo& o = g_.at(n.outputs[0]);

    int outShape[4], aSt[4], bSt[4];
    prepBinary4(a.shape, b.shape, o.shape, outShape, aSt, bSt);
    const long total = totalOf(o.shape);

    cl_kernel kern = rt.getKernel("elementwise", "binary_op", err);
    if (kern == nullptr) return false;

    Kernel kk(kern, "binary_op");
    kk.set(0, buf_[n.inputs[0]]);
    kk.set(1, buf_[n.inputs[1]]);
    kk.set(2, buf_[n.outputs[0]]);
    kk.setRaw(3, sizeof(outShape), outShape);
    kk.setRaw(4, sizeof(aSt), aSt);
    kk.setRaw(5, sizeof(bSt), bSt);
    kk.set(6, static_cast<int>(total));
    kk.set(7, code);
    return kk.run(static_cast<size_t>(total), 0, 0, 256, 0, 0, err);
}

bool Executor::opUnary(const Node& n, std::string* err) {
    ClRuntime& rt = ClRuntime::instance();
    const TensorInfo& o = g_.at(n.outputs[0]);
    const long total = totalOf(o.shape);

    if (n.opType == "Clip") {
        float lo = -INFINITY, hi = INFINITY;
        // opset 11+ 的 Clip 用输入传 min/max
        if (n.inputs.size() > 1 && n.inputs[1] >= 0) {
            const TensorInfo& t = g_.at(n.inputs[1]);
            if (!t.constF32.empty()) lo = t.constF32[0];
        }
        if (n.inputs.size() > 2 && n.inputs[2] >= 0) {
            const TensorInfo& t = g_.at(n.inputs[2]);
            if (!t.constF32.empty()) hi = t.constF32[0];
        }
        // 老 opset 用 attribute
        if (const AttributeProto* a = n.attr("min")) lo = a->f;
        if (const AttributeProto* a = n.attr("max")) hi = a->f;

        cl_kernel kern = rt.getKernel("elementwise", "clip_op", err);
        if (kern == nullptr) return false;
        Kernel kk(kern, "clip_op");
        kk.set(0, buf_[n.inputs[0]]);
        kk.set(1, buf_[n.outputs[0]]);
        kk.set(2, static_cast<int>(total));
        kk.set(3, lo);
        kk.set(4, hi);
        return kk.run(static_cast<size_t>(total), 0, 0, 256, 0, 0, err);
    }

    const int code = unaryCode(n.opType);
    if (code < 0) FAILF("未知一元算子 %s", n.opType.c_str());

    cl_kernel kern = rt.getKernel("elementwise", "unary_op", err);
    if (kern == nullptr) return false;
    Kernel kk(kern, "unary_op");
    kk.set(0, buf_[n.inputs[0]]);
    kk.set(1, buf_[n.outputs[0]]);
    kk.set(2, static_cast<int>(total));
    kk.set(3, code);
    return kk.run(static_cast<size_t>(total), 0, 0, 256, 0, 0, err);
}

// Concat：对每个输入跑一次 strided copy，写进输出对应位置
bool Executor::opConcat(const Node& n, std::string* err) {
    ClRuntime& rt = ClRuntime::instance();
    const TensorInfo& o = g_.at(n.outputs[0]);
    int64_t axis = n.attrInt("axis", 0);
    if (axis < 0) axis += static_cast<int64_t>(o.shape.size());
    if (axis < 0 || axis >= static_cast<int64_t>(o.shape.size())) {
        FAILF("Concat axis 越界: %s", n.name.c_str());
    }

    // ---- 快速路径：整块连续拷贝，不启动内核 ----
    //
    // 当 concat 轴之前的所有维都是 1（也就是 concat 轴就是最外层非单位维）时，
    // 每个输入在自身显存里是连续的，在输出里也落在一段连续区间上 ——
    // 直接 clEnqueueCopyBuffer 就行，省掉通用 strided copy 的逐元素索引计算。
    // 通道拼接（[1,C,H,W] 沿 axis=1）正是这种情况，是 YOLO 里最普遍的形态。
    {
        bool contiguous = true;
        for (int64_t d = 0; d < axis; ++d) {
            if (o.shape[d] != 1) {
                contiguous = false;
                break;
            }
        }
        if (contiguous) {
            size_t dstOffset = 0;
            for (size_t i = 0; i < n.inputs.size(); ++i) {
                if (n.inputs[i] < 0) continue;
                const TensorInfo& in = g_.at(n.inputs[i]);
                const size_t bytes = static_cast<size_t>(in.numElements()) * sizeof(float);
                if (!rt.copyBuffer(buf_[n.inputs[i]], buf_[n.outputs[0]], bytes, 0, dstOffset,
                                   err)) {
                    return false;
                }
                dstOffset += bytes;
            }
            return true;
        }
    }

    int outNat[4];
    naturalStride4(o.shape, outNat);
    // 注意：outNat 是「左补 1 到 4 维」后的步长数组，原始 rank 的 axis
    // 必须映射到 4 维下标 4-rank+axis 才能取对（否则会差几十倍）
    const int axis4 = 4 - static_cast<int>(o.shape.size()) + static_cast<int>(axis);

    cl_kernel kern = rt.getKernel("copy", "copy_strided", err);
    if (kern == nullptr) return false;

    int64_t axisOffset = 0;
    for (size_t i = 0; i < n.inputs.size(); ++i) {
        if (n.inputs[i] < 0) continue;
        const TensorInfo& in = g_.at(n.inputs[i]);
        int shp[4], srcNat[4];
        padShape4(in.shape, shp);
        naturalStride4(in.shape, srcNat);
        const long total = totalOf(in.shape);

        // 本输入在输出里的起始偏移 = 沿 axis 已累计的大小 * 输出在 axis 上的步长
        const int dstOffset = static_cast<int>(axisOffset * outNat[axis4]);

        Kernel kk(kern, "copy_strided");
        kk.set(0, buf_[n.inputs[i]]);
        kk.set(1, buf_[n.outputs[0]]);
        kk.setRaw(2, sizeof(shp), shp);
        kk.setRaw(3, sizeof(srcNat), srcNat);
        kk.setRaw(4, sizeof(outNat), outNat);
        kk.set(5, 0);
        kk.set(6, dstOffset);
        kk.set(7, static_cast<int>(total));
        if (!kk.run(static_cast<size_t>(total), 0, 0, 256, 0, 0, err)) return false;

        axisOffset += in.shape[axis];
    }
    return true;
}

bool Executor::opSplit(const Node& n, std::string* err) {
    ClRuntime& rt = ClRuntime::instance();
    const TensorInfo& in = g_.at(n.inputs[0]);
    int64_t axis = n.attrInt("axis", 0);
    if (axis < 0) axis += static_cast<int64_t>(in.shape.size());

    // split 尺寸：优先输入[1] 常量，其次 attribute，最后均分
    std::vector<int64_t> sizes;
    if (n.inputs.size() > 1 && n.inputs[1] >= 0) {
        const TensorInfo& t = g_.at(n.inputs[1]);
        if (!t.constI64.empty()) sizes = t.constI64;
    }
    if (sizes.empty()) sizes = n.attrInts("split");
    if (sizes.empty()) {
        const int64_t parts = static_cast<int64_t>(n.outputs.size());
        if (parts <= 0) FAILF("Split 输出个数为 0: %s", n.name.c_str());
        sizes.assign(parts, in.shape[axis] / parts);
    }
    if (sizes.size() != n.outputs.size()) {
        FAILF("Split 尺寸个数(%zu)与输出个数(%zu)不符: %s", sizes.size(), n.outputs.size(),
              n.name.c_str());
    }

    int srcNat[4];
    naturalStride4(in.shape, srcNat);
    // 同 Concat：axis 要映射到 4 维下标
    const int axis4 = 4 - static_cast<int>(in.shape.size()) + static_cast<int>(axis);

    cl_kernel kern = rt.getKernel("copy", "copy_strided", err);
    if (kern == nullptr) return false;

    int64_t axisOffset = 0;
    for (size_t i = 0; i < n.outputs.size(); ++i) {
        if (n.outputs[i] < 0) continue;
        const TensorInfo& o = g_.at(n.outputs[i]);
        int shp[4], dstNat[4];
        padShape4(o.shape, shp);
        naturalStride4(o.shape, dstNat);
        const long total = totalOf(o.shape);

        const int srcOffset = static_cast<int>(axisOffset * srcNat[axis4]);

        Kernel kk(kern, "copy_strided");
        kk.set(0, buf_[n.inputs[0]]);
        kk.set(1, buf_[n.outputs[i]]);
        kk.setRaw(2, sizeof(shp), shp);
        kk.setRaw(3, sizeof(srcNat), srcNat);
        kk.setRaw(4, sizeof(dstNat), dstNat);
        kk.set(5, srcOffset);
        kk.set(6, 0);
        kk.set(7, static_cast<int>(total));
        if (!kk.run(static_cast<size_t>(total), 0, 0, 256, 0, 0, err)) return false;

        axisOffset += sizes[i];
    }
    return true;
}

bool Executor::opSlice(const Node& n, std::string* err) {
    ClRuntime& rt = ClRuntime::instance();
    const TensorInfo& o = g_.at(n.outputs[0]);
    const TensorInfo& in = g_.at(n.inputs[0]);

    auto getI64 = [&](size_t idx) -> std::vector<int64_t> {
        if (idx >= n.inputs.size() || n.inputs[idx] < 0) return {};
        const TensorInfo& t = g_.at(n.inputs[idx]);
        return t.constI64;
    };
    std::vector<int64_t> starts = getI64(1), ends = getI64(2), axes = getI64(3), steps = getI64(4);
    if (starts.empty() || ends.empty()) {
        FAILF("Slice 的 starts/ends 必须是常量: %s", n.name.c_str());
    }
    const int64_t r = static_cast<int64_t>(in.shape.size());
    if (axes.empty()) {
        for (size_t i = 0; i < starts.size(); ++i) axes.push_back(static_cast<int64_t>(i));
    }
    if (steps.empty()) steps.assign(starts.size(), 1);

    int srcNatRaw[4];
    naturalStride4(in.shape, srcNatRaw);

    // 计算起点偏移 + 按步长调整该维的源步长。
    //
    // 【步长支持基本是白送的】copy_strided 内核本来就是「按步长寻址的块拷贝」：
    // 输出第 k 个元素取 src[offset + k*stride]。所以非单位步长（含负步长）
    // 只要把该维的源步长乘上 step 就行，不需要改内核。
    // 负步长也天然成立：stride 取负后寻址方向自动反过来。
    int64_t off = 0;
    for (size_t i = 0; i < axes.size(); ++i) {
        int64_t ax = axes[i] < 0 ? axes[i] + r : axes[i];
        if (ax < 0 || ax >= r) FAILF("Slice axes 越界: %s", n.name.c_str());
        // 4 维自然步长是按左补 1 的 4 维算的，这里映射一下
        const int dim4 = 4 - static_cast<int>(r) + static_cast<int>(ax);
        int64_t s = starts[i];
        if (s < 0) s += in.shape[ax];
        off += s * srcNatRaw[dim4];
        if (steps[i] != 1) {
            srcNatRaw[dim4] *= static_cast<int>(steps[i]);
        }
    }

    int shp[4];
    padShape4(o.shape, shp);
    int dstNat[4];
    naturalStride4(o.shape, dstNat);
    const long total = totalOf(o.shape);

    cl_kernel kern = rt.getKernel("copy", "copy_strided", err);
    if (kern == nullptr) return false;

    Kernel kk(kern, "copy_strided");
    kk.set(0, buf_[n.inputs[0]]);
    kk.set(1, buf_[n.outputs[0]]);
    kk.setRaw(2, sizeof(shp), shp);
    kk.setRaw(3, sizeof(srcNatRaw), srcNatRaw);
    kk.setRaw(4, sizeof(dstNat), dstNat);
    kk.set(5, static_cast<int>(off));
    kk.set(6, 0);
    kk.set(7, static_cast<int>(total));
    return kk.run(static_cast<size_t>(total), 0, 0, 256, 0, 0, err);
}

bool Executor::opTranspose(const Node& n, std::string* err) {
    ClRuntime& rt = ClRuntime::instance();
    const TensorInfo& in = g_.at(n.inputs[0]);
    const TensorInfo& o = g_.at(n.outputs[0]);
    const int64_t r = static_cast<int64_t>(in.shape.size());

    std::vector<int64_t> perm = n.attrInts("perm");
    if (perm.empty()) {
        for (int64_t i = r - 1; i >= 0; --i) perm.push_back(i);
    }
    if (static_cast<int64_t>(perm.size()) != r) FAILF("Transpose perm 长度不对: %s", n.name.c_str());

    int inNat[4];
    naturalStride4(in.shape, inNat);
    const int off4 = 4 - static_cast<int>(r);

    // 输出第 i 维对应输入第 perm[i] 维
    int shp[4], srcNat[4];
    padShape4(o.shape, shp);
    for (int i = 0; i < 4; ++i) srcNat[i] = 0;
    for (int64_t i = 0; i < r; ++i) {
        int64_t pm = perm[i] < 0 ? perm[i] + r : perm[i];
        if (pm < 0 || pm >= r) FAILF("Transpose perm 越界: %s", n.name.c_str());
        srcNat[off4 + i] = inNat[off4 + pm];
    }
    int dstNat[4];
    naturalStride4(o.shape, dstNat);
    const long total = totalOf(o.shape);

    cl_kernel kern = rt.getKernel("copy", "copy_strided", err);
    if (kern == nullptr) return false;

    Kernel kk(kern, "copy_strided");
    kk.set(0, buf_[n.inputs[0]]);
    kk.set(1, buf_[n.outputs[0]]);
    kk.setRaw(2, sizeof(shp), shp);
    kk.setRaw(3, sizeof(srcNat), srcNat);
    kk.setRaw(4, sizeof(dstNat), dstNat);
    kk.set(5, 0);
    kk.set(6, 0);
    kk.set(7, static_cast<int>(total));
    return kk.run(static_cast<size_t>(total), 0, 0, 256, 0, 0, err);
}

bool Executor::opResize(const Node& n, std::string* err) {
    ClRuntime& rt = ClRuntime::instance();
    const TensorInfo& in = g_.at(n.inputs[0]);
    const TensorInfo& o = g_.at(n.outputs[0]);
    if (in.shape.size() != 4) FAILF("Resize 只支持 4 维: %s", n.name.c_str());

    const std::string mode = n.attrString("mode", "nearest");
    const std::string ctm = n.attrString("coordinate_transformation_mode", "half_pixel");
    const std::string nm = n.attrString("nearest_mode", "round_prefer_floor");

    const int inShape[4] = {static_cast<int>(in.shape[0]), static_cast<int>(in.shape[1]),
                            static_cast<int>(in.shape[2]), static_cast<int>(in.shape[3])};
    const int outShape[4] = {static_cast<int>(o.shape[0]), static_cast<int>(o.shape[1]),
                             static_cast<int>(o.shape[2]), static_cast<int>(o.shape[3])};
    const float scaleH = static_cast<float>(o.shape[2]) / static_cast<float>(in.shape[2]);
    const float scaleW = static_cast<float>(o.shape[3]) / static_cast<float>(in.shape[3]);
    const size_t total = static_cast<size_t>(o.numElements());

    if (mode == "nearest") {
        if (ctm != "asymmetric" || (nm != "floor" && nm != "round_prefer_floor")) {
            FAILF("Resize nearest 目前只支持 asymmetric+floor，收到 ctm=%s nearest=%s (%s)",
                  ctm.c_str(), nm.c_str(), n.name.c_str());
        }
        cl_kernel kern = rt.getKernel("resize", "resize_nearest", err);
        if (kern == nullptr) return false;
        Kernel kk(kern, "resize_nearest");
        kk.set(0, buf_[n.inputs[0]]);
        kk.set(1, buf_[n.outputs[0]]);
        kk.setRaw(2, sizeof(inShape), inShape);
        kk.setRaw(3, sizeof(outShape), outShape);
        kk.set(4, scaleH);
        kk.set(5, scaleW);
        return kk.run(total, 0, 0, 256, 0, 0, err);
    }

    if (mode == "linear") {
        cl_kernel kern = rt.getKernel("resize", "resize_bilinear", err);
        if (kern == nullptr) return false;
        Kernel kk(kern, "resize_bilinear");
        kk.set(0, buf_[n.inputs[0]]);
        kk.set(1, buf_[n.outputs[0]]);
        kk.setRaw(2, sizeof(inShape), inShape);
        kk.setRaw(3, sizeof(outShape), outShape);
        kk.set(4, scaleH);
        kk.set(5, scaleW);
        kk.set(6, ctm == "align_corners" ? 1 : 0);
        return kk.run(total, 0, 0, 256, 0, 0, err);
    }

    FAILF("Resize 不支持 mode=%s (%s)", mode.c_str(), n.name.c_str());
}

bool Executor::opPool(const Node& n, std::string* err) {
    ClRuntime& rt = ClRuntime::instance();
    const TensorInfo& in = g_.at(n.inputs[0]);
    const TensorInfo& o = g_.at(n.outputs[0]);
    if (n.opType == "AveragePool") FAILF("AveragePool 还没实现: %s", n.name.c_str());
    if (in.shape.size() != 4) FAILF("MaxPool 只支持 4 维: %s", n.name.c_str());

    std::vector<int64_t> k = n.attrInts("kernel_shape");
    std::vector<int64_t> s = n.attrInts("strides");
    if (s.size() < 2) s = {1, 1};
    std::vector<int64_t> p = n.attrInts("pads");
    if (p.size() < 4) p = {0, 0, 0, 0};
    if (k.size() < 2) FAILF("MaxPool 缺少 kernel_shape: %s", n.name.c_str());

    const int inShape[4] = {static_cast<int>(in.shape[0]), static_cast<int>(in.shape[1]),
                            static_cast<int>(in.shape[2]), static_cast<int>(in.shape[3])};
    const int outShape[4] = {static_cast<int>(o.shape[0]), static_cast<int>(o.shape[1]),
                             static_cast<int>(o.shape[2]), static_cast<int>(o.shape[3])};
    const int ksize[2] = {static_cast<int>(k[0]), static_cast<int>(k[1])};
    const int strd[2] = {static_cast<int>(s[0]), static_cast<int>(s[1])};
    const int pad2[2] = {static_cast<int>(p[0]), static_cast<int>(p[1])};

    cl_kernel kern = rt.getKernel("pool", "maxpool", err);
    if (kern == nullptr) return false;

    Kernel kk(kern, "maxpool");
    kk.set(0, buf_[n.inputs[0]]);
    kk.set(1, buf_[n.outputs[0]]);
    kk.setRaw(2, sizeof(inShape), inShape);
    kk.setRaw(3, sizeof(outShape), outShape);
    kk.setRaw(4, sizeof(ksize), ksize);
    kk.setRaw(5, sizeof(strd), strd);
    kk.setRaw(6, sizeof(pad2), pad2);
    kk.set(7, 0);
    return kk.run(static_cast<size_t>(o.numElements()), 0, 0, 256, 0, 0, err);
}

bool Executor::opSoftmax(const Node& n, std::string* err) {
    ClRuntime& rt = ClRuntime::instance();
    const TensorInfo& in = g_.at(n.inputs[0]);
    const int64_t r = static_cast<int64_t>(in.shape.size());

    int64_t axis = n.attrInt("axis", -2);  // ONNX 默认 axis 会随 opset 变，这里给个保守默认
    // opset 13+ 的 Softmax 可能把 axis 放在输入[1]
    if (n.inputs.size() > 1 && n.inputs[1] >= 0) {
        const TensorInfo& t = g_.at(n.inputs[1]);
        if (!t.constI64.empty()) axis = t.constI64[0];
    }
    if (axis < 0) axis += r;
    if (axis < 0 || axis >= r) FAILF("Softmax axis 越界: %s", n.name.c_str());

    long outer = 1, axisSize = in.shape[axis], inner = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= in.shape[i];
    for (int64_t i = axis + 1; i < r; ++i) inner *= in.shape[i];

    cl_kernel kern = rt.getKernel("softmax", "softmax", err);
    if (kern == nullptr) return false;

    Kernel kk(kern, "softmax");
    kk.set(0, buf_[n.inputs[0]]);
    kk.set(1, buf_[n.outputs[0]]);
    kk.set(2, static_cast<int>(outer));
    kk.set(3, static_cast<int>(axisSize));
    kk.set(4, static_cast<int>(inner));
    return kk.run(static_cast<size_t>(outer), static_cast<size_t>(inner), 0, 0, 0, 0, err);
}

bool Executor::opMatMul(const Node& n, std::string* err) {
    ClRuntime& rt = ClRuntime::instance();
    const TensorInfo& a = g_.at(n.inputs[0]);
    const TensorInfo& b = g_.at(n.inputs[1]);
    if (a.shape.size() < 2 || b.shape.size() < 2) {
        FAILF("MatMul 只支持 >=2 维: %s", n.name.c_str());
    }

    // 只处理 batch 维完全相同的情况（yolo11n 就是）
    long batch = 1;
    for (size_t i = 0; i + 2 < a.shape.size(); ++i) batch *= a.shape[i];
    const long M = a.shape[a.shape.size() - 2];
    const long K = a.shape[a.shape.size() - 1];
    const long N = b.shape[b.shape.size() - 1];

    cl_kernel kern = rt.getKernel("matmul", "batched_matmul", err);
    if (kern == nullptr) return false;

    Kernel kk(kern, "batched_matmul");
    kk.set(0, buf_[n.inputs[0]]);
    kk.set(1, buf_[n.inputs[1]]);
    kk.set(2, buf_[n.inputs[0]]);  // C 占位（MatMul 没有偏置）
    kk.set(3, buf_[n.outputs[0]]);
    kk.set(4, static_cast<int>(batch));
    kk.set(5, static_cast<int>(M));
    kk.set(6, static_cast<int>(K));
    kk.set(7, static_cast<int>(N));
    kk.set(8, 0);      // transB
    kk.set(9, 0);      // hasC
    kk.set(10, 1.0f);  // alpha
    kk.set(11, 1.0f);  // beta
    return kk.run(static_cast<size_t>(batch * M * N), 0, 0, 256, 0, 0, err);
}

// ---------------------------------------------------------------------------
// Gemm：Y = alpha * op(A) x op(B) + beta * C，op(X) = transX ? X^T : X
//
// 和 MatMul 的关键差异（也是这个算子最容易写错的地方）：
//
//   1. transB=1 时权重按 [N, K] 存，N 要取 B.shape[0] ——
//      取错（习惯性地拿 B.shape[-1]）不会“算得不准”，而是把输出宽度算大
//      1280 而不是 1000，往输出缓冲里多写元素 → 越界 + 数据错乱，
//      最终表现为【输出全是 NaN】。分类头就是这个坑。
//   2. 第三个输入 C 是偏置（形状 [N]，按列广播），不能当它不存在。
// ---------------------------------------------------------------------------
bool Executor::opGemm(const Node& n, std::string* err) {
    ClRuntime& rt = ClRuntime::instance();
    const TensorInfo& a = g_.at(n.inputs[0]);
    const TensorInfo& b = g_.at(n.inputs[1]);
    const TensorInfo& o = g_.at(n.outputs[0]);
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        FAILF("Gemm 只支持 2 维: %s", n.name.c_str());
    }

    const bool transA = n.attrInt("transA", 0) != 0;
    const bool transB = n.attrInt("transB", 0) != 0;
    const float alpha = n.attrFloat("alpha", 1.0f);
    const float beta = n.attrFloat("beta", 1.0f);
    const bool hasC = n.inputs.size() > 2 && n.inputs[2] >= 0;

    if (transA) {
        FAILF("Gemm 暂不支持 transA=1（本批模型没用到）: %s", n.name.c_str());
    }
    const long M = a.shape[0];
    const long K = a.shape[1];
    const long N = transB ? b.shape[0] : b.shape[1];
    if (o.shape.size() != 2 || o.shape[0] != M || o.shape[1] != N) {
        FAILF("Gemm 输出尺寸不符（transB=%d，期望 [%ld,%ld]）: %s", transB ? 1 : 0, M, N,
              n.name.c_str());
    }
    if (hasC && g_.at(n.inputs[2]).numElements() != N) {
        FAILF("Gemm 的偏置 C 长度应为 %ld: %s", N, n.name.c_str());
    }

    cl_kernel kern = rt.getKernel("matmul", "batched_matmul", err);
    if (kern == nullptr) return false;

    Kernel kk(kern, "batched_matmul");
    kk.set(0, buf_[n.inputs[0]]);
    kk.set(1, buf_[n.inputs[1]]);
    kk.set(2, hasC ? buf_[n.inputs[2]] : buf_[n.inputs[0]]);  // 无偏置时传占位
    kk.set(3, buf_[n.outputs[0]]);
    kk.set(4, 1);  // batch
    kk.set(5, static_cast<int>(M));
    kk.set(6, static_cast<int>(K));
    kk.set(7, static_cast<int>(N));
    kk.set(8, transB ? 1 : 0);
    kk.set(9, hasC ? 1 : 0);
    kk.set(10, alpha);
    kk.set(11, beta);
    return kk.run(static_cast<size_t>(M * N), 0, 0, 256, 0, 0, err);
}

bool Executor::opGlobalPool(const Node& n, std::string* err) {
    ClRuntime& rt = ClRuntime::instance();
    const TensorInfo& in = g_.at(n.inputs[0]);
    if (in.shape.size() != 4) FAILF("GlobalAveragePool 只支持 4 维: %s", n.name.c_str());
    const int inShape[4] = {static_cast<int>(in.shape[0]), static_cast<int>(in.shape[1]),
                            static_cast<int>(in.shape[2]), static_cast<int>(in.shape[3])};

    cl_kernel kern = rt.getKernel("pool", "global_avgpool", err);
    if (kern == nullptr) return false;
    Kernel kk(kern, "global_avgpool");
    kk.set(0, buf_[n.inputs[0]]);
    kk.set(1, buf_[n.outputs[0]]);
    kk.setRaw(2, sizeof(inShape), inShape);
    return kk.run(static_cast<size_t>(in.shape[0] * in.shape[1]), 0, 0, 0, 0, 0, err);
}

bool Executor::opReduce(const Node& n, std::string* err) {
    ClRuntime& rt = ClRuntime::instance();
    const TensorInfo& in = g_.at(n.inputs[0]);

    // 只支持最后一维归约（yolo11n 没有 Reduce，yolo26n 用不到这条路径）
    if (in.shape.empty()) FAILF("Reduce 输入为空: %s", n.name.c_str());
    long outer = 1;
    for (size_t i = 0; i + 1 < in.shape.size(); ++i) outer *= in.shape[i];
    const long axisSize = in.shape[in.shape.size() - 1];
    const int op = (n.opType == "ReduceMax") ? 0 : 1;

    cl_kernel kern = rt.getKernel("matmul", "reduce_last_dim", err);
    if (kern == nullptr) return false;
    Kernel kk(kern, "reduce_last_dim");
    kk.set(0, buf_[n.inputs[0]]);
    kk.set(1, buf_[n.outputs[0]]);
    kk.set(2, static_cast<int>(outer));
    kk.set(3, static_cast<int>(axisSize));
    kk.set(4, op);
    return kk.run(static_cast<size_t>(outer), 0, 0, 0, 0, 0, err);
}

// ---------------------------------------------------------------------------
// ConvTranspose
//
// YOLO 的 seg / depth 头只有一种形态：k=2, s=2, p=0, d=1, group=1
// （实测 yolo11n-seg / yolo26n-seg / yolo26n-depth 三个模型参数完全一致）。
//
// 这种形态等价于「先做一次 1x1 卷积把通道数放大 4 倍，再做一次 pixel shuffle 2x」：
//
//   out[n][oc][2h+dh][2w+dw] = sum_ic in[n][ic][h][w] * W[ic][oc][dh][dw]
//
// 令 k = oc*4 + dh*2 + dw，就变成了一个普通的 1x1 卷积：
//   mid[n][k][h][w] = sum_ic in[n][ic][h][w] * A[ic][k]
//   再 shuffle： out[n][oc][2h+dh][2w+dw] = mid[n][oc*4+dh*2+dw][h][w]
//
// 【权重不用重排】ONNX 的 ConvTranspose 权重布局是 [C_in, C_out, kh, kw]，
// 展平后就是 [C_in][C_out*4]；而 1x1 卷积（kh=kw=1 时 RS=1）的 A 子块要求的
// 正是 [C][K] —— 两者完全一致，直接用原始显存就行。
//
// 【为什么不用朴素反卷积】朴素实现里每个 (h,w) 位置都要为所有输出通道循环一遍，
// 输入会被重复读 C_out 次 —— 实测算下来访存量是这里的三倍以上。走 1x1 GEMM 的话，
// 输入每个空间块只读一次（GEMM 的 B 子块天然复用）。
//
// bias 在 shuffle 阶段按 oc 加：1x1 卷积有 C_out*4 个输出通道，但它们共用同一个
// oc 的 bias，在 shuffle 里加最方便（省掉一次 bias 重排）。
// ---------------------------------------------------------------------------
bool Executor::opConvTranspose(const Node& n, int /*nodeIdx*/, std::string* err) {
    ClRuntime& rt = ClRuntime::instance();
    const TensorInfo& x = g_.at(n.inputs[0]);
    const TensorInfo& w = g_.at(n.inputs[1]);
    const TensorInfo& o = g_.at(n.outputs[0]);
    const bool hasBias = n.inputs.size() > 2 && n.inputs[2] >= 0;

    if (x.shape.size() != 4 || w.shape.size() != 4 || o.shape.size() != 4) {
        FAILF("ConvTranspose 只支持 4 维: %s", n.name.c_str());
    }
    std::vector<int64_t> k = n.attrInts("kernel_shape");
    if (k.size() < 2) k = {w.shape[2], w.shape[3]};
    std::vector<int64_t> s = n.attrInts("strides");
    if (s.size() < 2) s = {1, 1};
    std::vector<int64_t> p = n.attrInts("pads");
    if (p.size() < 4) p = {0, 0, 0, 0};
    std::vector<int64_t> d = n.attrInts("dilations");
    if (d.size() < 2) d = {1, 1};
    const int group = static_cast<int>(n.attrInt("group", 1));

    if (group != 1 || k[0] != 2 || k[1] != 2 || s[0] != 2 || s[1] != 2 || p[0] != 0 ||
        p[1] != 0 || p[2] != 0 || p[3] != 0 || d[0] != 1 || d[1] != 1) {
        FAILF("ConvTranspose 目前只支持 k=2/s=2/p=0/d=1/group=1（YOLO seg/depth 头就是这种）: %s",
              n.name.c_str());
    }

    const int N = static_cast<int>(x.shape[0]);
    const int Cin = static_cast<int>(x.shape[1]);
    const int H = static_cast<int>(x.shape[2]);
    const int W = static_cast<int>(x.shape[3]);
    // ONNX ConvTranspose 权重布局是 [C_in, C_out/group, kh, kw]
    const int Cout = static_cast<int>(w.shape[1]);
    const int K4 = Cout * 4;
    if (static_cast<int>(o.shape[2]) != H * 2 || static_cast<int>(o.shape[3]) != W * 2) {
        FAILF("ConvTranspose 输出尺寸不符预期: %s", n.name.c_str());
    }

    // ---- 1) 1x1 卷积：[N, Cin, H, W] -> 中间张量 [N, K4, H, W] ----
    // 中间张量从缓冲池取。注意【所有权归池子】，函数返回前【不要】释放它 ——
    // 池子按尺寸缓存、在 Executor 析构时统一释放，这才是缓冲复用的意义。
    const size_t midBytes = static_cast<size_t>(N) * K4 * H * W * sizeof(float);
    cl_mem mid = acquireScratch(midBytes, err);
    if (mid == nullptr) {
        FAILF("ConvTranspose 取中间张量失败: %s", n.name.c_str());
    }

    cl_kernel gemm = rt.getKernel("conv", "conv2d_gemm", err);
    if (gemm == nullptr) {
        return false;
    }
    const size_t kmax = rt.kernelMaxWG(gemm);
    if (kmax < 256) {
        FAILF("conv2d_gemm 需要 work-group=256，但驱动只允许 %zu (%s)", kmax, n.name.c_str());
    }

    const long SP = static_cast<long>(N) * H * W;
    {
        Kernel kk(gemm, "conv2d_gemm");
        kk.set(0, buf_[n.inputs[0]]);
        // 直接用原始权重的显存（布局 [Cin][Cout*4] 正好等于 GEMM 要的 A[C][K]）
        kk.set(1, buf_[n.inputs[1]]);
        kk.set(2, buf_[n.inputs[1]]);  // bias 在 shuffle 阶段加，这里传个合法 buffer 占位
        kk.set(3, mid);
        kk.set(4, N);
        kk.set(5, Cin);
        kk.set(6, H);
        kk.set(7, W);
        kk.set(8, K4);   // 输出通道数
        kk.set(9, H);    // OH
        kk.set(10, W);   // OW
        kk.set(11, 1);   // KH
        kk.set(12, 1);   // KW
        kk.set(13, 1);   // SH
        kk.set(14, 1);   // SW
        kk.set(15, 0);   // PT
        kk.set(16, 0);   // PL
        kk.set(17, 0);   // hasBias
        kk.set(18, 0);   // activation
        const size_t gwsSpatial = ((static_cast<size_t>(SP) + 63) / 64) * 16;
        const size_t gwsChannel = ((static_cast<size_t>(K4) + 63) / 64) * 16;
        if (!kk.run(gwsSpatial, gwsChannel, 0, 16, 16, 0, err)) {
            return false;
        }
    }

    // ---- 2) pixel shuffle：[N, K4, H, W] -> [N, Cout, 2H, 2W]，顺便按 oc 加 bias ----
    cl_kernel shuf = rt.getKernel("copy", "pixel_shuffle2x", err);
    if (shuf == nullptr) {
        return false;
    }
    {
        Kernel sk(shuf, "pixel_shuffle2x");
        sk.set(0, mid);
        sk.set(1, buf_[n.outputs[0]]);
        sk.set(2, hasBias ? buf_[n.inputs[2]] : buf_[n.inputs[1]]);
        sk.set(3, hasBias ? 1 : 0);
        sk.set(4, Cout);
        sk.set(5, H);
        sk.set(6, W);
        sk.set(7, N);
        const size_t total = static_cast<size_t>(N) * Cout * 4 * H * W;
        if (!sk.run(total, 0, 0, 256, 0, 0, err)) {
            return false;
        }
    }

    // 【不要 release(mid)】缓冲池持有它，析构时统一释放（详见 acquireScratch 的说明）。
    return true;
}

}  // namespace onnxcl

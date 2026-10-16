// 图融合 pass。
//
// 目前只做一件最有价值的事：把 YOLO 里成吨的 SiLU 消掉。
//
//   Conv ──► Sigmoid ──► Mul(x, sig)      变成      Conv(activation=SiLU)
//     x ────────────────────┘
//
// 为什么值得：实测 yolo11n 上 Sigmoid + Mul 占 GPU 时间的 31%，
// 而且它们是纯显存带宽受限的 —— 一个 [1,64,80,80] 的张量，SiLU 要读 2 遍写 1 遍。
// 融进 Conv 的 epilogue 之后这部分显存往返直接归零（结果还在寄存器里就乘掉了）。
//
// 融合分两档：
//   · 前驱是 Conv 且 Conv 输出只被这一处消费 -> 融进 Conv 的 activation 参数（最优）
//   · 其它情况                              -> 两条节点合成一条 SiLU 一元算子（次优，仍省一次往返）
#include <string>
#include <vector>

#include "onnxcl/graph.h"

namespace onnxcl {
namespace {

// 给节点挂一个内部属性（用 onnxcl_ 前缀和 ONNX 官方属性区分开）
void setIntAttr(Node& n, const char* key, int64_t v) {
    for (auto& a : n.attributes) {
        if (a.name == key) {
            a.i = v;
            a.type = ATTR_INT;
            return;
        }
    }
    AttributeProto a;
    a.name = key;
    a.type = ATTR_INT;
    a.i = v;
    n.attributes.push_back(a);
}

}  // namespace

FusionStats fuseGraph(Graph& g) {
    FusionStats st;
    const size_t nNodes = g.nodes.size();
    const size_t nTensors = g.tensors.size();

    // ---- 生产者 / 消费者 ----
    std::vector<int> producer(nTensors, -1);
    std::vector<std::vector<int>> consumers(nTensors);
    for (size_t i = 0; i < nNodes; ++i) {
        for (int o : g.nodes[i].outputs) {
            if (o >= 0) producer[o] = static_cast<int>(i);
        }
        for (int in : g.nodes[i].inputs) {
            if (in >= 0) consumers[in].push_back(static_cast<int>(i));
        }
    }
    // 是图输出的张量不能被就地覆盖
    std::vector<char> isGraphOutput(nTensors, 0);
    for (int o : g.outputs) {
        if (o >= 0) isGraphOutput[o] = 1;
    }

    // ---- 扫描 SiLU 模式 ----
    for (size_t i = 0; i < nNodes; ++i) {
        Node& mul = g.nodes[i];
        if (mul.opType != "Mul" || mul.inputs.size() != 2) continue;

        // 找出「另一个输入正好是 Sigmoid(x)，而 x 是这第 k 个输入」的位置
        int sigIdx = -1;
        for (int k = 0; k < 2; ++k) {
            const int other = mul.inputs[1 - k];
            if (other < 0) continue;
            const int p = producer[other];
            if (p < 0) continue;
            const Node& sig = g.nodes[p];
            if (sig.opType == "Sigmoid" && sig.inputs.size() == 1 && sig.inputs[0] == mul.inputs[k]) {
                sigIdx = p;
                break;
            }
        }
        if (sigIdx < 0) continue;

        Node& sig = g.nodes[sigIdx];
        const int x = sig.inputs[0];
        const int convIdx = producer[x];

        // 能不能直接融进 Conv？
        //
        // 注意这里的「没有别的消费者」不能写成 consumers[x].size() == 1：
        // SiLU 的模式是 Mul(x, Sigmoid(x))，x 同时被 Sigmoid 和 Mul 读，本来就是 2 个消费者。
        // 正确的条件是这个两个节点之外没人再读 x。
        bool otherConsumer = false;
        for (int c : consumers[x]) {
            if (c != sigIdx && c != static_cast<int>(i)) {
                otherConsumer = true;
                break;
            }
        }
        const bool canFoldIntoConv = convIdx >= 0 && g.nodes[convIdx].opType == "Conv" &&
                                     !otherConsumer && isGraphOutput[x] == 0 &&
                                     g.at(x).dtype == DType::F32 &&
                                     g.at(mul.outputs[0]).dtype == DType::F32;

        if (canFoldIntoConv) {
            setIntAttr(g.nodes[convIdx], "onnxcl_activation", 1 /*SiLU*/);
            // Sigmoid 和 Mul 都退化成「视图」：输出直接复用 Conv 的输出显存，
            // 不分配内存、不启动内核。Mul 的结果本来就等于 Conv+SiLU 的结果。
            sig.opType = "Alias";
            mul.opType = "Alias";
            mul.inputs.assign(1, x);
            st.convSilu++;
        } else {
            // 融不进 Conv，但至少把 Sigmoid+Mul 合成一条 SiLU
            mul.opType = "SiLU";
            mul.inputs.assign(1, x);
            sig.opType = "Alias";
            st.unarySilu++;
        }
    }

    return st;
}

}  // namespace onnxcl

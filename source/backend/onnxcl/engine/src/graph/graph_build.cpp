#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <functional>
#include <set>

#include "onnxcl/graph.h"
#include "shape_infer.h"

namespace onnxcl {

// ---------------------------------------------------------------------------
// dtype
// ---------------------------------------------------------------------------

const char* dtypeName(DType t) {
    switch (t) {
        case DType::F32: return "float32";
        case DType::F16: return "float16";
        case DType::I32: return "int32";
        case DType::I64: return "int64";
        case DType::U8: return "uint8";
        default: return "unknown";
    }
}

int dtypeBytes(DType t) {
    switch (t) {
        case DType::F32: return 4;
        case DType::F16: return 2;
        case DType::I32: return 4;
        case DType::I64: return 8;
        case DType::U8: return 1;
        default: return 0;
    }
}

DType dtypeFromOnnx(int onnxType) {
    switch (onnxType) {
        case ONNX_FLOAT: return DType::F32;
        case ONNX_FLOAT16:
        case ONNX_BFLOAT16: return DType::F16;
        case ONNX_INT32: return DType::I32;
        case ONNX_INT64: return DType::I64;
        case ONNX_UINT8:
        case ONNX_INT8:
        case ONNX_BOOL: return DType::U8;
        default: return DType::Unknown;
    }
}

// ---------------------------------------------------------------------------
// shape 工具
// ---------------------------------------------------------------------------

bool broadcastShapes(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
                     std::vector<int64_t>& out) {
    const size_t r = std::max(a.size(), b.size());
    out.assign(r, 1);
    for (size_t i = 0; i < r; ++i) {
        const int64_t da = (i + a.size() < r) ? 1 : a[i + a.size() - r];
        const int64_t db = (i + b.size() < r) ? 1 : b[i + b.size() - r];
        if (da == db) {
            out[i] = da;
        } else if (da == 1) {
            out[i] = db;
        } else if (db == 1) {
            out[i] = da;
        } else {
            return false;
        }
    }
    return true;
}

std::string shapeToString(const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i) out += ",";
        out += std::to_string(s[i]);
    }
    return out + "]";
}

size_t Graph::totalTensorBytes() const {
    size_t n = 0;
    for (const auto& t : tensors) {
        if (t.isConst) continue;  // 常量单独算
        n += t.bytes;
    }
    return n;
}

// ---------------------------------------------------------------------------
// 常量物化
// ---------------------------------------------------------------------------

static bool tensorToI64(const TensorProto& t, std::vector<int64_t>& out) {
    const int64_t n = t.numElements();
    out.clear();
    out.reserve(static_cast<size_t>(n));
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(t.rawData.data());
    const size_t rawLen = t.rawData.size();

    switch (t.dataType) {
        case ONNX_INT64:
            if (!t.rawData.empty()) {
                if (rawLen < static_cast<size_t>(n) * 8) return false;
                const int64_t* p = reinterpret_cast<const int64_t*>(raw);
                out.assign(p, p + n);
            } else {
                out = t.int64Data;
            }
            return true;
        case ONNX_INT32:
            if (!t.rawData.empty()) {
                if (rawLen < static_cast<size_t>(n) * 4) return false;
                const int32_t* p = reinterpret_cast<const int32_t*>(raw);
                for (int64_t i = 0; i < n; ++i) out.push_back(p[i]);
            } else {
                for (int32_t v : t.int32Data) out.push_back(v);
            }
            return true;
        case ONNX_UINT64:
            if (!t.rawData.empty() && rawLen >= static_cast<size_t>(n) * 8) {
                const uint64_t* p = reinterpret_cast<const uint64_t*>(raw);
                for (int64_t i = 0; i < n; ++i) out.push_back(static_cast<int64_t>(p[i]));
                return true;
            }
            for (uint64_t v : t.uint64Data) out.push_back(static_cast<int64_t>(v));
            return true;
        case ONNX_UINT8:
        case ONNX_INT8:
        case ONNX_BOOL:
            if (rawLen < static_cast<size_t>(n)) return false;
            for (int64_t i = 0; i < n; ++i) out.push_back(raw[i]);
            return true;
        default: {
            // 浮点常量（比如 Resize 的 scales）—— 转成整数没有意义，但索引类用途需要它
            std::vector<float> f;
            if (!tensorToFloat(t, f)) return false;
            for (float v : f) out.push_back(static_cast<int64_t>(v));
            return true;
        }
    }
}

static void materializeConst(TensorInfo& ti, const TensorProto& t) {
    ti.isConst = true;
    ti.dtype = dtypeFromOnnx(t.dataType);
    if (!t.dims.empty()) {
        ti.shape.assign(t.dims.begin(), t.dims.end());
    } else {
        ti.shape.clear();  // 标量
    }
    tensorToI64(t, ti.constI64);
    tensorToFloat(t, ti.constF32);
    ti.bytes = static_cast<size_t>(ti.numElements()) * 4;  // 统一按 float32 上传给 GPU
}

// ---------------------------------------------------------------------------
// Graph::build
// ---------------------------------------------------------------------------

struct Builder {
    Graph& g;
    ModelProto const& m;
    std::string* err;

    int touch(const std::string& name) {
        auto it = g.nameToIndex.find(name);
        if (it != g.nameToIndex.end()) return it->second;
        const int idx = static_cast<int>(g.tensors.size());
        g.tensors.emplace_back();
        g.tensors.back().name = name;
        g.nameToIndex[name] = idx;
        return idx;
    }

    bool run() {
        g.irVersion = m.irVersion;
        g.opset = m.opsetVersion();
        const GraphProto& gp = m.graph;

        // --- 1. 先把所有张量登记上 ---
        for (const auto& vi : gp.input) touch(vi.name);
        for (const auto& vi : gp.output) touch(vi.name);
        for (const auto& vi : gp.valueInfo) touch(vi.name);
        for (const auto& t : gp.initializer) touch(t.name);
        for (const auto& n : gp.node) {
            for (const auto& in : n.input) {
                if (!in.empty()) touch(in);
            }
            for (const auto& out : n.output) {
                if (!out.empty()) touch(out);
            }
        }

        // --- 2. 已知的 shape / dtype ---
        auto applyValueInfo = [&](const ValueInfo& vi) {
            TensorInfo& ti = g.at(touch(vi.name));
            if (vi.type.elemType != ONNX_UNDEFINED) {
                ti.dtype = dtypeFromOnnx(vi.type.elemType);
            }
            if (vi.type.shape.hasShape) {
                ti.shape = vi.type.shape.dims;
            }
        };
        for (const auto& vi : gp.input) {
            applyValueInfo(vi);
            g.at(g.nameToIndex[vi.name]).isInput = true;
            g.inputs.push_back(g.nameToIndex[vi.name]);
        }
        for (const auto& vi : gp.output) {
            applyValueInfo(vi);
            g.at(g.nameToIndex[vi.name]).isOutput = true;
            g.outputs.push_back(g.nameToIndex[vi.name]);
        }
        for (const auto& vi : gp.valueInfo) applyValueInfo(vi);

        // --- 3. 初始化为常量 ---
        for (const auto& t : gp.initializer) {
            materializeConst(g.at(g.nameToIndex[t.name]), t);
        }

        // --- 4. 建节点 ---
        std::vector<std::vector<int>> nodeInputs;
        std::vector<std::vector<int>> nodeOutputs;
        for (const auto& n : gp.node) {
            Node node;
            node.opType = n.opType;
            node.name = n.name;
            node.attributes = n.attribute;
            // 带 "::" 的是自定义 domain，本批模型没有
            if (!n.domain.empty() && n.domain != "ai.onnx" && n.domain != "onnx") {
                *err = "不支持的 op domain: " + n.domain + " (op=" + n.opType + ")";
                return false;
            }
            for (const auto& in : n.input) {
                node.inputs.push_back(in.empty() ? -1 : g.nameToIndex[in]);
            }
            for (const auto& out : n.output) {
                node.outputs.push_back(out.empty() ? -1 : g.nameToIndex[out]);
            }
            g.nodes.push_back(std::move(node));
        }

        // --- 5. 常量节点（Constant）先落地 ---
        for (size_t i = 0; i < g.nodes.size(); ++i) {
            Node& n = g.nodes[i];
            if (n.opType != "Constant") continue;
            const AttributeProto* val = n.attr("value");
            const AttributeProto* valFloat = n.attr("value_float");
            const AttributeProto* valInt = n.attr("value_int");
            TensorInfo& ti = g.at(n.outputs[0]);
            if (val != nullptr) {
                materializeConst(ti, val->t);
            } else if (valFloat != nullptr) {
                ti.isConst = true;
                ti.shape.clear();
                ti.dtype = DType::F32;
                ti.constF32 = {valFloat->f};
                ti.constI64 = {static_cast<int64_t>(valFloat->f)};
                ti.bytes = 4;
            } else if (valInt != nullptr) {
                ti.isConst = true;
                ti.shape.clear();
                ti.dtype = DType::I64;
                ti.constI64 = {valInt->i};
                ti.constF32 = {static_cast<float>(valInt->i)};
                ti.bytes = 4;
            } else {
                *err = "Constant 节点缺少 value 属性: " + n.name;
                return false;
            }
        }

        // --- 6. 拓扑排序 ---
        // ONNX 规范要求 node 已按拓扑序排列，但显式排一次更稳（不依赖导出工具的自觉）
        std::vector<int> producerOf(g.tensors.size(), -1);
        for (size_t i = 0; i < g.nodes.size(); ++i) {
            for (int o : g.nodes[i].outputs) {
                if (o >= 0) producerOf[o] = static_cast<int>(i);
            }
        }

        std::vector<int> order;
        std::vector<char> state(g.nodes.size(), 0);  // 0=未访问 1=访问中 2=完成
        std::function<bool(int)> visit = [&](int ni) -> bool {
            if (state[ni] == 2) return true;
            if (state[ni] == 1) {
                *err = "图里有环，涉及节点: " + g.nodes[ni].name;
                return false;
            }
            state[ni] = 1;
            for (int in : g.nodes[ni].inputs) {
                if (in < 0) continue;
                const TensorInfo& ti = g.at(in);
                if (ti.isInput || ti.isConst) continue;  // 源头，形状已定
                const int p = producerOf[in];
                if (p < 0) {
                    *err = "张量 " + ti.name + " 没有生产者也不是输入/常量";
                    return false;
                }
                if (!visit(p)) return false;
            }
            state[ni] = 2;
            order.push_back(ni);
            return true;
        };
        for (size_t i = 0; i < g.nodes.size(); ++i) {
            if (!visit(static_cast<int>(i))) return false;
        }

        // --- 7. 按拓扑序推 shape ---
        std::vector<Node> sortedNodes;
        sortedNodes.reserve(order.size());
        for (int i : order) sortedNodes.push_back(g.nodes[i]);
        g.nodes = std::move(sortedNodes);

        for (const Node& n : g.nodes) {
            if (n.opType == "Constant") continue;  // 上面处理过了
            std::string shapeErr;
            if (!inferNodeShape(n, g, &shapeErr)) {
                *err = shapeErr;
                return false;
            }
        }

        // --- 8. 收尾：算字节数、检查 shape 完整性 ---
        for (size_t i = 0; i < g.tensors.size(); ++i) {
            TensorInfo& ti = g.tensors[i];
            if (ti.shape.empty() && !ti.isConst) {
                // 标量是合法的（ReduceMax keepdims=0 全归约）
                // 但不能有维度 <=0 的未知情况
            }
            for (auto d : ti.shape) {
                // 维度 0 是合法的 —— ONNX 用空张量（dims=[0]）表示"不指定"，
                // 典型是 Resize 的 roi 输入。但它只应出现在常量上：
                // 激活值为空说明图本身有问题，早点报出来比后面算出一堆空张量好。
                if (d < 0 || (d == 0 && !ti.isConst)) {
                    *err = "张量 " + ti.name + " 的 shape 非法: " + shapeToString(ti.shape);
                    return false;
                }
            }
            if (ti.dtype == DType::Unknown) {
                *err = "张量 " + ti.name + " 的 dtype 未知";
                return false;
            }
            if (!ti.isConst) {
                ti.bytes = static_cast<size_t>(ti.numElements()) * dtypeBytes(ti.dtype);
            }
        }
        return true;
    }
};

bool buildGraph(const ModelProto& model, Graph& graph, std::string* err) {
    std::string localErr;
    Builder b{graph, model, err ? err : &localErr};
    if (!b.run()) return false;
    // 形状推导完之后做融合（融合不改 shape，所以顺序不能反）
    fuseGraph(graph);
    return true;
}

// ---------------------------------------------------------------------------
// dump
// ---------------------------------------------------------------------------

void dumpGraph(const Graph& g, FILE* out, bool verbose) {
    std::fprintf(out, "ir_version %" PRId64 "\n", g.irVersion);
    std::fprintf(out, "opset %" PRId64 "\n", g.opset);
    std::fprintf(out, "tensors %zu\n", g.tensors.size());
    std::fprintf(out, "nodes %zu\n", g.nodes.size());

    for (int i : g.inputs) {
        const TensorInfo& t = g.at(i);
        std::fprintf(out, "input %s %s %s\n", t.name.c_str(), dtypeName(t.dtype),
                     shapeToString(t.shape).c_str());
    }
    for (int i : g.outputs) {
        const TensorInfo& t = g.at(i);
        std::fprintf(out, "output %s %s %s\n", t.name.c_str(), dtypeName(t.dtype),
                     shapeToString(t.shape).c_str());
    }

    size_t constBytes = 0, actBytes = 0;
    int nConst = 0;
    for (const auto& t : g.tensors) {
        if (t.isConst) {
            constBytes += t.bytes;
            nConst++;
        } else {
            actBytes += t.bytes;
        }
    }
    std::fprintf(out, "const_tensors %d\n", nConst);
    std::fprintf(out, "const_bytes %zu\n", constBytes);
    std::fprintf(out, "activation_bytes %zu\n", actBytes);

    // 算子直方图
    {
        std::vector<std::pair<std::string, int>> hist;
        for (const auto& n : g.nodes) {
            bool found = false;
            for (auto& h : hist) {
                if (h.first == n.opType) {
                    h.second++;
                    found = true;
                    break;
                }
            }
            if (!found) hist.push_back({n.opType, 1});
        }
        std::sort(hist.begin(), hist.end());
        for (const auto& h : hist) std::fprintf(out, "op %s %d\n", h.first.c_str(), h.second);
    }

    if (!verbose) return;

    for (size_t i = 0; i < g.nodes.size(); ++i) {
        const Node& n = g.nodes[i];
        std::fprintf(out, "node %zu %s name=%s\n", i, n.opType.c_str(), n.name.c_str());
        for (size_t k = 0; k < n.inputs.size(); ++k) {
            if (n.inputs[k] < 0) {
                std::fprintf(out, "  in[%zu] <缺失>\n", k);
            } else {
                const TensorInfo& t = g.at(n.inputs[k]);
                std::fprintf(out, "  in[%zu] %s %s %s%s\n", k, t.name.c_str(), dtypeName(t.dtype),
                             shapeToString(t.shape).c_str(), t.isConst ? " (const)" : "");
            }
        }
        for (size_t k = 0; k < n.outputs.size(); ++k) {
            if (n.outputs[k] < 0) {
                std::fprintf(out, "  out[%zu] <缺失>\n", k);
            } else {
                const TensorInfo& t = g.at(n.outputs[k]);
                std::fprintf(out, "  out[%zu] %s %s %s\n", k, t.name.c_str(), dtypeName(t.dtype),
                             shapeToString(t.shape).c_str());
            }
        }
    }
}

}  // namespace onnxcl

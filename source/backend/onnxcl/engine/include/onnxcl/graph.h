// 图 IR：ONNX 解析结果 → 可执行的算子图。
//
// 为什么不抄 MNN 那套：
//   MNN 有 source/shape（80 文件 / 6759 行）+ source/geometry（49 文件 / 10805 行）。
//   那是因为它要支持任意模型、任意动态 shape、并在图上做算子分解（Split→Slice→Raster）。
//   我们的模型是静态 shape、无控制流的，所以只需要：
//     · 一次拓扑序遍历，边走边推 shape（shape_infer.cpp）
//     · 不需要 geometry 层，算子直接落地成 kernel
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "onnxcl/onnx_model.h"

namespace onnxcl {

enum class DType { F32, F16, I32, I64, U8, Unknown };

const char* dtypeName(DType t);
int dtypeBytes(DType t);
DType dtypeFromOnnx(int onnxType);

// ---------------------------------------------------------------------------

struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    DType dtype = DType::F32;

    bool isInput = false;
    bool isOutput = false;
    bool isConst = false;   // initializer 或 Constant 节点产出

    // 常量值。索引类参数以 int64 保存，浮点参数以 float 保存（两者可能同时有）
    std::vector<int64_t> constI64;
    std::vector<float> constF32;

    // 内存规划结果（memory_plan.cpp 填）
    int slot = -1;         // 复用槽位号
    size_t slotOffset = 0;
    size_t bytes = 0;

    int64_t numElements() const {
        if (shape.empty()) return 1;
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    int dims() const { return static_cast<int>(shape.size()); }
};

// ---------------------------------------------------------------------------

struct Node {
    std::string opType;
    std::string name;
    std::vector<int> inputs;   // tensor 下标；-1 表示该输入缺失
    std::vector<int> outputs;
    std::vector<AttributeProto> attributes;

    const AttributeProto* attr(const std::string& key) const {
        for (const auto& a : attributes) {
            if (a.name == key) return &a;
        }
        return nullptr;
    }
    int64_t attrInt(const std::string& key, int64_t def) const {
        const AttributeProto* a = attr(key);
        return a ? a->i : def;
    }
    std::vector<int64_t> attrInts(const std::string& key) const {
        const AttributeProto* a = attr(key);
        return a ? a->ints : std::vector<int64_t>{};
    }
    float attrFloat(const std::string& key, float def) const {
        const AttributeProto* a = attr(key);
        return a ? a->f : def;
    }
    std::string attrString(const std::string& key, const std::string& def) const {
        const AttributeProto* a = attr(key);
        return a ? a->s : def;
    }
};

// ---------------------------------------------------------------------------

struct Graph {
    std::vector<TensorInfo> tensors;
    std::vector<Node> nodes;
    std::map<std::string, int> nameToIndex;
    std::vector<int> inputs;
    std::vector<int> outputs;

    int64_t irVersion = 0;
    int64_t opset = 0;

    TensorInfo* find(const std::string& name) {
        auto it = nameToIndex.find(name);
        return it == nameToIndex.end() ? nullptr : &tensors[it->second];
    }
    const TensorInfo* find(const std::string& name) const {
        auto it = nameToIndex.find(name);
        return it == nameToIndex.end() ? nullptr : &tensors[it->second];
    }
    TensorInfo& at(int i) { return tensors[i]; }
    const TensorInfo& at(int i) const { return tensors[i]; }

    // 统计信息
    size_t totalTensorBytes() const;
};

// ONNX ModelProto → Graph。失败时填 err。
bool buildGraph(const ModelProto& model, Graph& graph, std::string* err);

// 图融合 pass 的统计（用于验收：融合了多少处）
struct FusionStats {
    int convSilu = 0;   // 融进 Conv epilogue 的 SiLU 处数
    int unarySilu = 0;  // 退化成单条一元 SiLU 的处数
    int total() const { return convSilu + unarySilu; }
};

// 就地修改 g。需要在形状推导之后调用（融合不改 shape）。
// 融合后的节点会变成 opType == "Alias"（零成本视图），具体见 fusion.cpp
FusionStats fuseGraph(Graph& g);

// 把图打印成人可读的文本（dump_ir 用）
void dumpGraph(const Graph& g, FILE* out, bool verbose);

// ---------------------------------------------------------------------------
// 形状工具
// ---------------------------------------------------------------------------

// ONNX / numpy 风格的多向广播。返回 false 表示不可广播。
bool broadcastShapes(const std::vector<int64_t>& a, const std::vector<int64_t>& b,
                     std::vector<int64_t>& out);

// 把 shape 打印成 [1,3,640,640]
std::string shapeToString(const std::vector<int64_t>& s);

}  // namespace onnxcl

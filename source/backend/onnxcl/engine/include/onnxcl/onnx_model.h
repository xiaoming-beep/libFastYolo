// ONNX 模型的内存表示（只覆盖 yolo 系列实际用到的 6 个 message 类型）。
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace onnxcl {

// onnx.TensorProto.DataType
enum OnnxDataType : int {
    ONNX_UNDEFINED  = 0,
    ONNX_FLOAT      = 1,
    ONNX_UINT8      = 2,
    ONNX_INT8       = 3,
    ONNX_UINT16     = 4,
    ONNX_INT16      = 5,
    ONNX_INT32      = 6,
    ONNX_INT64      = 7,
    ONNX_STRING     = 8,
    ONNX_BOOL       = 9,
    ONNX_FLOAT16    = 10,
    ONNX_DOUBLE     = 11,
    ONNX_UINT32     = 12,
    ONNX_UINT64     = 13,
    ONNX_COMPLEX64  = 14,
    ONNX_COMPLEX128 = 15,
    ONNX_BFLOAT16   = 16,
};

const char* onnxDataTypeName(int t);
int onnxDataTypeBytes(int t);

// onnx.AttributeProto.AttributeType
enum OnnxAttrType : int {
    ATTR_UNDEFINED     = 0,
    ATTR_FLOAT         = 1,
    ATTR_INT           = 2,
    ATTR_STRING        = 3,
    ATTR_TENSOR        = 4,
    ATTR_GRAPH         = 5,
    ATTR_FLOATS        = 6,
    ATTR_INTS          = 7,
    ATTR_STRINGS       = 8,
    ATTR_TENSORS       = 9,
    ATTR_GRAPHS        = 10,
    ATTR_SPARSE_TENSOR = 11,
    ATTR_SPARSE_TENSORS = 12,
    ATTR_TYPE_PROTO    = 13,
    ATTR_TYPE_PROTOS   = 14,
};

struct TensorShape {
    std::vector<int64_t> dims;       // 已知的维度值；未知为 -1
    std::vector<std::string> dimParams;  // 符号维度名（如 "batch"）；已知时为空串
    bool hasShape = false;
    bool isDynamic() const {
        for (auto d : dims) {
            if (d < 0) return true;
        }
        return false;
    }
};

struct TensorType {
    int elemType = ONNX_UNDEFINED;
    TensorShape shape;
};

struct ValueInfo {
    std::string name;
    TensorType type;
};

struct TensorProto {
    std::string name;
    std::vector<int64_t> dims;
    int dataType = ONNX_UNDEFINED;

    // 未压缩的重复字段（老格式）
    std::vector<float> floatData;
    std::vector<int32_t> int32Data;
    std::vector<int64_t> int64Data;
    std::vector<uint64_t> uint64Data;
    std::vector<double> doubleData;

    // 新格式：原始字节
    std::string rawData;

    // external data（本批模型不用，但解析后要走同一条路）
    std::string externalLocation;
    int64_t externalOffset = 0;
    int64_t externalLength = -1;

    int64_t numElements() const {
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return dims.empty() ? 1 : n;
    }
};

struct AttributeProto {
    std::string name;
    int type = ATTR_UNDEFINED;
    float f = 0.0f;
    int64_t i = 0;
    std::string s;
    std::vector<float> floats;
    std::vector<int64_t> ints;
    std::vector<std::string> strings;
    TensorProto t;

    bool has(const std::string& key) const { return name == key; }
    // 便于按需取值
    int64_t intOr(int64_t def) const { return type == ATTR_INT ? i : def; }
};

struct NodeProto {
    std::string name;
    std::string opType;
    std::string domain;
    std::vector<std::string> input;
    std::vector<std::string> output;
    std::vector<AttributeProto> attribute;

    // 按名字找属性
    const AttributeProto* attr(const std::string& key) const {
        for (const auto& a : attribute) {
            if (a.name == key) return &a;
        }
        return nullptr;
    }
    int64_t attrInt(const std::string& key, int64_t def) const {
        const AttributeProto* a = attr(key);
        return a ? a->i : def;
    }
    int64_t attrIntIdx(const std::string& key, size_t idx, int64_t def) const {
        const AttributeProto* a = attr(key);
        if (!a || idx >= a->ints.size()) return def;
        return a->ints[idx];
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

struct GraphProto {
    std::string name;
    std::vector<NodeProto> node;
    std::vector<TensorProto> initializer;
    std::vector<ValueInfo> input;
    std::vector<ValueInfo> output;
    std::vector<ValueInfo> valueInfo;
};

struct ModelProto {
    int64_t irVersion = 0;
    std::string producerName;
    std::string producerVersion;
    std::string domain;
    int64_t modelVersion = 0;
    GraphProto graph;
    std::map<std::string, int64_t> opsetImport;  // domain -> version
    std::string docString;

    // 从文件/内存加载。返回 false 并填充 error。
    bool loadFromFile(const std::string& path, std::string* error);
    bool loadFromMemory(const uint8_t* data, size_t size, std::string* error);

    int64_t opsetVersion() const {
        auto it = opsetImport.find("");
        if (it != opsetImport.end()) return it->second;
        it = opsetImport.find("ai.onnx");
        if (it != opsetImport.end()) return it->second;
        return opsetImport.empty() ? 0 : opsetImport.begin()->second;
    }

    const TensorProto* findInitializer(const std::string& name) const {
        for (const auto& t : graph.initializer) {
            if (t.name == name) return &t;
        }
        return nullptr;
    }
};

// 把 TensorProto 的数值统一取成 float（供 dump / 后续常量折叠用）。
// 对 float16 / int32 / int64 / uint8 都做转换。失败返回 false。
bool tensorToFloat(const TensorProto& t, std::vector<float>& out);

}  // namespace onnxcl

#include "onnxcl/onnx_model.h"

#include <cstdio>
#include <cstring>
#include <fstream>

#include "onnxcl/pb_reader.h"

namespace onnxcl {

const char* onnxDataTypeName(int t) {
    switch (t) {
        case ONNX_FLOAT: return "float32";
        case ONNX_UINT8: return "uint8";
        case ONNX_INT8: return "int8";
        case ONNX_UINT16: return "uint16";
        case ONNX_INT16: return "int16";
        case ONNX_INT32: return "int32";
        case ONNX_INT64: return "int64";
        case ONNX_STRING: return "string";
        case ONNX_BOOL: return "bool";
        case ONNX_FLOAT16: return "float16";
        case ONNX_DOUBLE: return "float64";
        case ONNX_UINT32: return "uint32";
        case ONNX_UINT64: return "uint64";
        case ONNX_BFLOAT16: return "bfloat16";
        default: return "unknown";
    }
}

int onnxDataTypeBytes(int t) {
    switch (t) {
        case ONNX_FLOAT: return 4;
        case ONNX_UINT8:
        case ONNX_INT8:
        case ONNX_BOOL: return 1;
        case ONNX_UINT16:
        case ONNX_INT16:
        case ONNX_FLOAT16:
        case ONNX_BFLOAT16: return 2;
        case ONNX_INT32:
        case ONNX_UINT32: return 4;
        case ONNX_INT64:
        case ONNX_UINT64:
        case ONNX_DOUBLE: return 8;
        default: return 0;
    }
}

namespace pb_parse {

using pb::Reader;
using pb::WT_I32;
using pb::WT_I64;
using pb::WT_LEN;
using pb::WT_VARINT;

// ---------------------------------------------------------------------------
// 重复数值字段：ONNX 里既可能是 packed（LEN），也可能是逐个（VARINT/FIXED）
// ---------------------------------------------------------------------------

static void readRepeatedInt64(Reader& r, uint32_t wt, std::vector<int64_t>& out) {
    if (wt == WT_LEN) {
        Reader sub;
        if (!r.readMessage(sub)) return;
        while (!sub.empty()) {
            uint64_t v = 0;
            if (!sub.readVarint(v)) break;
            out.push_back(static_cast<int64_t>(v));
        }
    } else if (wt == WT_VARINT) {
        uint64_t v = 0;
        if (r.readVarint(v)) out.push_back(static_cast<int64_t>(v));
    } else {
        r.skipField(wt);
    }
}

static void readRepeatedUInt64(Reader& r, uint32_t wt, std::vector<uint64_t>& out) {
    if (wt == WT_LEN) {
        Reader sub;
        if (!r.readMessage(sub)) return;
        while (!sub.empty()) {
            uint64_t v = 0;
            if (!sub.readVarint(v)) break;
            out.push_back(v);
        }
    } else if (wt == WT_VARINT) {
        uint64_t v = 0;
        if (r.readVarint(v)) out.push_back(v);
    } else {
        r.skipField(wt);
    }
}

static void readRepeatedFloat(Reader& r, uint32_t wt, std::vector<float>& out) {
    if (wt == WT_LEN) {
        Reader sub;
        if (!r.readMessage(sub)) return;
        while (sub.remaining() >= 4) {
            uint32_t bits = 0;
            sub.readFixed32(bits);
            out.push_back(pb::asFloat(bits));
        }
    } else if (wt == WT_I32) {
        uint32_t bits = 0;
        if (r.readFixed32(bits)) out.push_back(pb::asFloat(bits));
    } else {
        r.skipField(wt);
    }
}

static void readRepeatedDouble(Reader& r, uint32_t wt, std::vector<double>& out) {
    if (wt == WT_LEN) {
        Reader sub;
        if (!r.readMessage(sub)) return;
        while (sub.remaining() >= 8) {
            uint64_t bits = 0;
            sub.readFixed64(bits);
            double d;
            std::memcpy(&d, &bits, 8);
            out.push_back(d);
        }
    } else if (wt == WT_I64) {
        uint64_t bits = 0;
        if (r.readFixed64(bits)) {
            double d;
            std::memcpy(&d, &bits, 8);
            out.push_back(d);
        }
    } else {
        r.skipField(wt);
    }
}

static void readRepeatedString(Reader& r, uint32_t wt, std::vector<std::string>& out) {
    std::string s;
    if (wt == WT_LEN && r.readString(s)) out.push_back(std::move(s));
}

// ---------------------------------------------------------------------------
// TensorShapeProto
// ---------------------------------------------------------------------------

static void parseDimension(Reader r, TensorShape& shape) {
    int64_t dimValue = -1;
    std::string dimParam;
    bool hasValue = false;
    uint32_t field = 0, wt = 0;
    while (!r.empty() && r.readTag(field, wt)) {
        if (field == 1 && wt == WT_VARINT) {
            uint64_t v = 0;
            r.readVarint(v);
            dimValue = static_cast<int64_t>(v);
            hasValue = true;
        } else if (field == 2 && wt == WT_LEN) {
            r.readString(dimParam);
        } else {
            r.skipField(wt);
        }
    }
    shape.dims.push_back(hasValue ? dimValue : -1);
    shape.dimParams.push_back(dimParam);
}

static void parseTensorShape(Reader r, TensorShape& shape) {
    shape.hasShape = true;
    uint32_t field = 0, wt = 0;
    while (!r.empty() && r.readTag(field, wt)) {
        if (field == 1 && wt == WT_LEN) {
            Reader sub;
            if (r.readMessage(sub)) parseDimension(sub, shape);
        } else {
            r.skipField(wt);
        }
    }
}

// TypeProto.Tensor
static void parseTypeProtoTensor(Reader r, TensorType& type) {
    uint32_t field = 0, wt = 0;
    while (!r.empty() && r.readTag(field, wt)) {
        if (field == 1 && wt == WT_VARINT) {
            uint64_t v = 0;
            r.readVarint(v);
            type.elemType = static_cast<int>(v);
        } else if (field == 2 && wt == WT_LEN) {
            Reader sub;
            if (r.readMessage(sub)) parseTensorShape(sub, type.shape);
        } else {
            r.skipField(wt);
        }
    }
}

// TypeProto
static void parseTypeProto(Reader r, TensorType& type) {
    uint32_t field = 0, wt = 0;
    while (!r.empty() && r.readTag(field, wt)) {
        if (field == 1 && wt == WT_LEN) {
            Reader sub;
            if (r.readMessage(sub)) parseTypeProtoTensor(sub, type);
        } else {
            r.skipField(wt);
        }
    }
}

static void parseValueInfo(Reader r, ValueInfo& vi) {
    uint32_t field = 0, wt = 0;
    while (!r.empty() && r.readTag(field, wt)) {
        if (field == 1 && wt == WT_LEN) {
            r.readString(vi.name);
        } else if (field == 2 && wt == WT_LEN) {
            Reader sub;
            if (r.readMessage(sub)) parseTypeProto(sub, vi.type);
        } else {
            r.skipField(wt);
        }
    }
}

// ---------------------------------------------------------------------------
// TensorProto
// ---------------------------------------------------------------------------

static void parseStringStringEntry(Reader r, std::string& key, std::string& value) {
    uint32_t field = 0, wt = 0;
    while (!r.empty() && r.readTag(field, wt)) {
        if (field == 1 && wt == WT_LEN) {
            r.readString(key);
        } else if (field == 2 && wt == WT_LEN) {
            r.readString(value);
        } else {
            r.skipField(wt);
        }
    }
}

static void parseTensorProto(Reader r, TensorProto& t) {
    uint32_t field = 0, wt = 0;
    while (!r.empty() && r.readTag(field, wt)) {
        switch (field) {
            case 1:
                readRepeatedInt64(r, wt, t.dims);
                break;
            case 2: {
                uint64_t v = 0;
                if (wt == WT_VARINT && r.readVarint(v)) t.dataType = static_cast<int>(v);
                else r.skipField(wt);
                break;
            }
            case 4:
                readRepeatedFloat(r, wt, t.floatData);
                break;
            case 5: {
                // int32_data：可能是 packed varint，也可能是 fixed32（负数的特殊编码）
                if (wt == WT_LEN) {
                    Reader sub;
                    if (r.readMessage(sub)) {
                        while (!sub.empty()) {
                            uint64_t v = 0;
                            if (!sub.readVarint(v)) break;
                            t.int32Data.push_back(static_cast<int32_t>(v));
                        }
                    } else {
                        r.skipField(wt);
                    }
                } else if (wt == WT_VARINT) {
                    uint64_t v = 0;
                    if (r.readVarint(v)) t.int32Data.push_back(static_cast<int32_t>(v));
                } else if (wt == WT_I32) {
                    uint32_t bits = 0;
                    if (r.readFixed32(bits)) t.int32Data.push_back(static_cast<int32_t>(bits));
                } else {
                    r.skipField(wt);
                }
                break;
            }
            case 7:
                readRepeatedInt64(r, wt, t.int64Data);
                break;
            case 8:
                if (wt == WT_LEN) r.readString(t.name);
                else r.skipField(wt);
                break;
            case 9: {
                const uint8_t* p = nullptr;
                size_t n         = 0;
                if (wt == WT_LEN && r.readBytes(p, n)) {
                    t.rawData.assign(reinterpret_cast<const char*>(p), n);
                } else {
                    r.skipField(wt);
                }
                break;
            }
            case 10: {
                Reader sub;
                if (wt == WT_LEN && r.readMessage(sub)) {
                    std::string k, v;
                    parseStringStringEntry(sub, k, v);
                    if (k == "location") {
                        t.externalLocation = v;
                    } else if (k == "offset") {
                        t.externalOffset = std::strtoll(v.c_str(), nullptr, 10);
                    } else if (k == "length") {
                        t.externalLength = std::strtoll(v.c_str(), nullptr, 10);
                    }
                } else {
                    r.skipField(wt);
                }
                break;
            }
            case 12:
                readRepeatedDouble(r, wt, t.doubleData);
                break;
            case 13:
                readRepeatedUInt64(r, wt, t.uint64Data);
                break;
            default:
                r.skipField(wt);
                break;
        }
    }
}

// ---------------------------------------------------------------------------
// AttributeProto
// ---------------------------------------------------------------------------

static void parseAttributeProto(Reader r, AttributeProto& a) {
    uint32_t field = 0, wt = 0;
    while (!r.empty() && r.readTag(field, wt)) {
        switch (field) {
            case 1:
                if (wt == WT_LEN) r.readString(a.name);
                else r.skipField(wt);
                break;
            case 2: {
                uint32_t bits = 0;
                if (wt == WT_I32 && r.readFixed32(bits)) a.f = pb::asFloat(bits);
                else r.skipField(wt);
                break;
            }
            case 3: {
                uint64_t v = 0;
                if (wt == WT_VARINT && r.readVarint(v)) a.i = static_cast<int64_t>(v);
                else r.skipField(wt);
                break;
            }
            case 4:
                if (wt == WT_LEN) r.readString(a.s);
                else r.skipField(wt);
                break;
            case 5: {
                Reader sub;
                if (wt == WT_LEN && r.readMessage(sub)) parseTensorProto(sub, a.t);
                else r.skipField(wt);
                break;
            }
            case 6: {
                // 嵌套 graph：本批模型没有。跳过。
                r.skipField(wt);
                break;
            }
            case 7:
                readRepeatedFloat(r, wt, a.floats);
                break;
            case 8:
                readRepeatedInt64(r, wt, a.ints);
                break;
            case 9:
                readRepeatedString(r, wt, a.strings);
                break;
            case 20: {
                uint64_t v = 0;
                if (wt == WT_VARINT && r.readVarint(v)) a.type = static_cast<int>(v);
                else r.skipField(wt);
                break;
            }
            default:
                r.skipField(wt);
                break;
        }
    }
}

// ---------------------------------------------------------------------------
// NodeProto
// ---------------------------------------------------------------------------

static void parseNodeProto(Reader r, NodeProto& n) {
    uint32_t field = 0, wt = 0;
    while (!r.empty() && r.readTag(field, wt)) {
        switch (field) {
            case 1:
                readRepeatedString(r, wt, n.input);
                break;
            case 2:
                readRepeatedString(r, wt, n.output);
                break;
            case 3:
                if (wt == WT_LEN) r.readString(n.name);
                else r.skipField(wt);
                break;
            case 4:
                if (wt == WT_LEN) r.readString(n.opType);
                else r.skipField(wt);
                break;
            case 5: {
                Reader sub;
                if (wt == WT_LEN && r.readMessage(sub)) {
                    AttributeProto a;
                    parseAttributeProto(sub, a);
                    n.attribute.push_back(std::move(a));
                } else {
                    r.skipField(wt);
                }
                break;
            }
            case 7:
                if (wt == WT_LEN) r.readString(n.domain);
                else r.skipField(wt);
                break;
            default:
                r.skipField(wt);
                break;
        }
    }
}

// ---------------------------------------------------------------------------
// GraphProto
// ---------------------------------------------------------------------------

static void parseGraphProto(Reader r, GraphProto& g) {
    uint32_t field = 0, wt = 0;
    while (!r.empty() && r.readTag(field, wt)) {
        switch (field) {
            case 1: {
                Reader sub;
                if (wt == WT_LEN && r.readMessage(sub)) {
                    NodeProto n;
                    parseNodeProto(sub, n);
                    g.node.push_back(std::move(n));
                } else {
                    r.skipField(wt);
                }
                break;
            }
            case 2:
                if (wt == WT_LEN) r.readString(g.name);
                else r.skipField(wt);
                break;
            case 5: {
                Reader sub;
                if (wt == WT_LEN && r.readMessage(sub)) {
                    TensorProto t;
                    parseTensorProto(sub, t);
                    g.initializer.push_back(std::move(t));
                } else {
                    r.skipField(wt);
                }
                break;
            }
            case 11:
            case 12:
            case 13: {
                Reader sub;
                if (wt == WT_LEN && r.readMessage(sub)) {
                    ValueInfo vi;
                    parseValueInfo(sub, vi);
                    if (field == 11) g.input.push_back(std::move(vi));
                    else if (field == 12) g.output.push_back(std::move(vi));
                    else g.valueInfo.push_back(std::move(vi));
                } else {
                    r.skipField(wt);
                }
                break;
            }
            default:
                r.skipField(wt);
                break;
        }
    }
}

// ---------------------------------------------------------------------------
// OperatorSetIdProto
// ---------------------------------------------------------------------------

static void parseOperatorSetId(Reader r, std::string& domain, int64_t& version) {
    uint32_t field = 0, wt = 0;
    while (!r.empty() && r.readTag(field, wt)) {
        if (field == 1 && wt == WT_LEN) {
            r.readString(domain);
        } else if (field == 2 && wt == WT_VARINT) {
            uint64_t v = 0;
            r.readVarint(v);
            version = static_cast<int64_t>(v);
        } else {
            r.skipField(wt);
        }
    }
}

// ---------------------------------------------------------------------------
// ModelProto
// ---------------------------------------------------------------------------

static void parseModelProto(Reader r, ModelProto& m) {
    uint32_t field = 0, wt = 0;
    while (!r.empty() && r.readTag(field, wt)) {
        switch (field) {
            case 1: {
                uint64_t v = 0;
                if (wt == WT_VARINT && r.readVarint(v)) m.irVersion = static_cast<int64_t>(v);
                else r.skipField(wt);
                break;
            }
            case 2:
                if (wt == WT_LEN) r.readString(m.producerName);
                else r.skipField(wt);
                break;
            case 3:
                if (wt == WT_LEN) r.readString(m.producerVersion);
                else r.skipField(wt);
                break;
            case 4:
                if (wt == WT_LEN) r.readString(m.domain);
                else r.skipField(wt);
                break;
            case 5: {
                uint64_t v = 0;
                if (wt == WT_VARINT && r.readVarint(v)) m.modelVersion = static_cast<int64_t>(v);
                else r.skipField(wt);
                break;
            }
            case 6:
                if (wt == WT_LEN) r.readString(m.docString);
                else r.skipField(wt);
                break;
            case 7: {
                Reader sub;
                if (wt == WT_LEN && r.readMessage(sub)) parseGraphProto(sub, m.graph);
                else r.skipField(wt);
                break;
            }
            case 8: {
                Reader sub;
                if (wt == WT_LEN && r.readMessage(sub)) {
                    std::string domain;
                    int64_t version = 0;
                    parseOperatorSetId(sub, domain, version);
                    m.opsetImport[domain] = version;
                } else {
                    r.skipField(wt);
                }
                break;
            }
            default:
                r.skipField(wt);
                break;
        }
    }
}

}  // namespace pb_parse

// ---------------------------------------------------------------------------
// 公开接口
// ---------------------------------------------------------------------------

bool ModelProto::loadFromMemory(const uint8_t* data, size_t size, std::string* error) {
    if (data == nullptr || size == 0) {
        if (error) *error = "空数据";
        return false;
    }
    // 检查首字段是不是 field 1 (ir_version)，做个粗校验
    pb::Reader r(data, size);
    uint32_t field = 0, wt = 0;
    if (!r.readTag(field, wt)) {
        if (error) *error = "不是合法的 protobuf 数据";
        return false;
    }
    pb::Reader r2(data, size);
    pb_parse::parseModelProto(r2, *this);
    if (graph.node.empty()) {
        if (error) *error = "解析成功但没有 node（可能不是 ONNX 文件）";
        return false;
    }
    return true;
}

bool ModelProto::loadFromFile(const std::string& path, std::string* error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        if (error) *error = "打不开文件: " + path;
        return false;
    }
    f.seekg(0, std::ios::end);
    const std::streamoff len = f.tellg();
    f.seekg(0, std::ios::beg);
    if (len <= 0) {
        if (error) *error = "文件为空: " + path;
        return false;
    }
    std::vector<uint8_t> buf(static_cast<size_t>(len));
    f.read(reinterpret_cast<char*>(buf.data()), len);
    if (!f) {
        if (error) *error = "读取失败: " + path;
        return false;
    }
    return loadFromMemory(buf.data(), buf.size(), error);
}

// ---------------------------------------------------------------------------

static float halfToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp       = (h >> 10) & 0x1F;
    uint32_t mant      = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;  // ±0
        } else {
            // 次正规数
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);  // Inf / NaN
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

static float bf16ToFloat(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

bool tensorToFloat(const TensorProto& t, std::vector<float>& out) {
    const int64_t n = t.numElements();
    out.clear();
    out.reserve(static_cast<size_t>(n));

    const uint8_t* raw = reinterpret_cast<const uint8_t*>(t.rawData.data());
    const size_t rawLen = t.rawData.size();

    switch (t.dataType) {
        case ONNX_FLOAT: {
            if (!t.rawData.empty()) {
                if (rawLen < static_cast<size_t>(n) * 4) return false;
                const float* p = reinterpret_cast<const float*>(raw);
                out.assign(p, p + n);
            } else {
                out = t.floatData;
            }
            return true;
        }
        case ONNX_FLOAT16: {
            if (rawLen < static_cast<size_t>(n) * 2) return false;
            const uint16_t* p = reinterpret_cast<const uint16_t*>(raw);
            for (int64_t i = 0; i < n; ++i) out.push_back(halfToFloat(p[i]));
            return true;
        }
        case ONNX_BFLOAT16: {
            if (rawLen < static_cast<size_t>(n) * 2) return false;
            const uint16_t* p = reinterpret_cast<const uint16_t*>(raw);
            for (int64_t i = 0; i < n; ++i) out.push_back(bf16ToFloat(p[i]));
            return true;
        }
        case ONNX_DOUBLE: {
            if (!t.rawData.empty()) {
                if (rawLen < static_cast<size_t>(n) * 8) return false;
                const double* p = reinterpret_cast<const double*>(raw);
                for (int64_t i = 0; i < n; ++i) out.push_back(static_cast<float>(p[i]));
            } else {
                for (double d : t.doubleData) out.push_back(static_cast<float>(d));
            }
            return true;
        }
        case ONNX_INT64: {
            if (!t.rawData.empty()) {
                if (rawLen < static_cast<size_t>(n) * 8) return false;
                const int64_t* p = reinterpret_cast<const int64_t*>(raw);
                for (int64_t i = 0; i < n; ++i) out.push_back(static_cast<float>(p[i]));
            } else {
                for (int64_t v : t.int64Data) out.push_back(static_cast<float>(v));
            }
            return true;
        }
        case ONNX_UINT64: {
            if (!t.rawData.empty()) {
                if (rawLen < static_cast<size_t>(n) * 8) return false;
                const uint64_t* p = reinterpret_cast<const uint64_t*>(raw);
                for (int64_t i = 0; i < n; ++i) out.push_back(static_cast<float>(p[i]));
            } else {
                for (uint64_t v : t.uint64Data) out.push_back(static_cast<float>(v));
            }
            return true;
        }
        case ONNX_INT32: {
            if (!t.rawData.empty()) {
                if (rawLen < static_cast<size_t>(n) * 4) return false;
                const int32_t* p = reinterpret_cast<const int32_t*>(raw);
                for (int64_t i = 0; i < n; ++i) out.push_back(static_cast<float>(p[i]));
            } else {
                for (int32_t v : t.int32Data) out.push_back(static_cast<float>(v));
            }
            return true;
        }
        case ONNX_UINT32: {
            if (rawLen < static_cast<size_t>(n) * 4) return false;
            const uint32_t* p = reinterpret_cast<const uint32_t*>(raw);
            for (int64_t i = 0; i < n; ++i) out.push_back(static_cast<float>(p[i]));
            return true;
        }
        case ONNX_INT16: {
            if (rawLen < static_cast<size_t>(n) * 2) return false;
            const int16_t* p = reinterpret_cast<const int16_t*>(raw);
            for (int64_t i = 0; i < n; ++i) out.push_back(static_cast<float>(p[i]));
            return true;
        }
        case ONNX_UINT16: {
            if (rawLen < static_cast<size_t>(n) * 2) return false;
            const uint16_t* p = reinterpret_cast<const uint16_t*>(raw);
            for (int64_t i = 0; i < n; ++i) out.push_back(static_cast<float>(p[i]));
            return true;
        }
        case ONNX_INT8: {
            if (rawLen < static_cast<size_t>(n)) return false;
            for (int64_t i = 0; i < n; ++i) out.push_back(static_cast<float>(static_cast<int8_t>(raw[i])));
            return true;
        }
        case ONNX_UINT8:
        case ONNX_BOOL: {
            if (rawLen < static_cast<size_t>(n)) return false;
            for (int64_t i = 0; i < n; ++i) out.push_back(static_cast<float>(raw[i]));
            return true;
        }
        default:
            return false;
    }
}

}  // namespace onnxcl

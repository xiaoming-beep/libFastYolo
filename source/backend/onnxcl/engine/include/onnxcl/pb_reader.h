// ONNX 文件解析用的最小 protobuf wire-format 读取器。
// 只读不写，零第三方依赖（不需要 libprotobuf / protoc）。
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace onnxcl {
namespace pb {

enum WireType : uint32_t {
    WT_VARINT = 0,
    WT_I64    = 1,
    WT_LEN    = 2,
    WT_SGROUP = 3,
    WT_EGROUP = 4,
    WT_I32    = 5,
};

class Reader {
public:
    Reader() : begin_(nullptr), cur_(nullptr), end_(nullptr) {}
    Reader(const uint8_t* data, size_t size) : begin_(data), cur_(data), end_(data + size) {}

    bool empty() const { return cur_ >= end_; }
    size_t remaining() const { return static_cast<size_t>(end_ - cur_); }
    const uint8_t* data() const { return begin_; }

    bool readVarint(uint64_t& out) {
        uint64_t result = 0;
        int shift = 0;
        while (cur_ < end_) {
            const uint8_t b = *cur_++;
            result |= static_cast<uint64_t>(b & 0x7F) << shift;
            if ((b & 0x80) == 0) {
                out = result;
                return true;
            }
            shift += 7;
            if (shift > 63) {
                return false;
            }
        }
        return false;
    }

    bool readTag(uint32_t& field, uint32_t& wireType) {
        uint64_t key = 0;
        if (!readVarint(key)) {
            return false;
        }
        field    = static_cast<uint32_t>(key >> 3);
        wireType = static_cast<uint32_t>(key & 0x7);
        return field != 0;
    }

    bool readFixed32(uint32_t& out) {
        if (remaining() < 4) {
            return false;
        }
        std::memcpy(&out, cur_, 4);
        cur_ += 4;
        return true;
    }

    bool readFixed64(uint64_t& out) {
        if (remaining() < 8) {
            return false;
        }
        std::memcpy(&out, cur_, 8);
        cur_ += 8;
        return true;
    }

    // 读一个长度前缀的字节块，返回指向数据的指针（不拷贝）。
    bool readBytes(const uint8_t*& outData, size_t& outLen) {
        uint64_t len = 0;
        if (!readVarint(len)) {
            return false;
        }
        if (len > remaining()) {
            return false;
        }
        outData = cur_;
        outLen  = static_cast<size_t>(len);
        cur_ += len;
        return true;
    }

    bool readString(std::string& out) {
        const uint8_t* p = nullptr;
        size_t n         = 0;
        if (!readBytes(p, n)) {
            return false;
        }
        out.assign(reinterpret_cast<const char*>(p), n);
        return true;
    }

    // 取一个嵌套 message 的 Reader。
    bool readMessage(Reader& out) {
        const uint8_t* p = nullptr;
        size_t n         = 0;
        if (!readBytes(p, n)) {
            return false;
        }
        out = Reader(p, n);
        return true;
    }

    bool skipField(uint32_t wireType) {
        switch (wireType) {
            case WT_VARINT: {
                uint64_t v = 0;
                return readVarint(v);
            }
            case WT_I64:
                return skip(8);
            case WT_LEN: {
                const uint8_t* p = nullptr;
                size_t n         = 0;
                return readBytes(p, n);
            }
            case WT_I32:
                return skip(4);
            default:
                return false;
        }
    }

    bool skip(size_t n) {
        if (remaining() < n) {
            return false;
        }
        cur_ += n;
        return true;
    }

private:
    const uint8_t* begin_;
    const uint8_t* cur_;
    const uint8_t* end_;
};

// 读一个 float（可能是 packed 里的单个元素）。
inline float asFloat(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// zigzag / 补码：protobuf 的 int32/int64 都是补码 varint。
inline int64_t asInt64(uint64_t v) { return static_cast<int64_t>(v); }
inline int32_t asInt32(uint64_t v) { return static_cast<int32_t>(v); }

}  // namespace pb
}  // namespace onnxcl

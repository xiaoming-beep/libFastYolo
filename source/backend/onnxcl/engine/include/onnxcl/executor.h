// 执行器：把图 IR 逐节点落到 OpenCL 内核调用上。
//
// 内存策略（M2/M3 正确性优先版）：
//   · 每个非 const 张量一个独立 cl_mem（还不做复用，yolo11n 峰值 281MB，先跑通）
//   · Reshape / Flatten / Unsqueeze / Squeeze 是纯视图操作 —— 用 alias 让输出
//     直接共享输入的显存，零拷贝、零内核
//   · const 张量（权重/常量）在 init 时一次性上传
#pragma once

#include <CL/cl.h>

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "onnxcl/graph.h"

namespace onnxcl {

class Executor {
public:
    explicit Executor(const Graph& g);
    ~Executor();

    // 编译全部内核 + 上传权重。耗时主要在编译内核（首次约几百 ms 到几秒）。
    // deviceIndex 是 OpenCL 设备序号（多设备机器上选卡用）
    bool init(std::string* err, int deviceIndex = 0);

    // input 必须包含 graph.inputs[0] 所需的元素个数（按 shape 展平的 float32）
    bool run(const float* input, std::vector<std::vector<float>>& outputs, std::string* err);

    void setProfiling(bool on);
    void printProfile(FILE* out) const;
    void resetProfile();

    // 逐节点耗时（用来定位到具体哪一层慢）。需要 profiling 打开。
    struct NodeProfile {
        std::string name;
        std::string opType;
        std::string shape;   // 输入形状，方便认层
        double ms = 0.0;
        int count = 0;
    };
    void printSlowNodes(FILE* out, int limit) const;

    // 显存占用（字节）
    size_t deviceBytes() const { return deviceBytes_; }

private:
    bool runNode(const Node& n, int nodeIdx, std::string* err);

    // 各算子的实现
    bool opConv(const Node& n, int nodeIdx, std::string* err);
    bool opBinary(const Node& n, const char* opName, std::string* err);
    bool opUnary(const Node& n, std::string* err);
    bool opConcat(const Node& n, std::string* err);
    bool opSplit(const Node& n, std::string* err);
    bool opSlice(const Node& n, std::string* err);
    bool opTranspose(const Node& n, std::string* err);
    bool opResize(const Node& n, std::string* err);
    bool opPool(const Node& n, std::string* err);
    bool opSoftmax(const Node& n, std::string* err);
    bool opMatMul(const Node& n, std::string* err);
    // Gemm 和 MatMul 的差异：transB（权重已转置）、第三个输入 C（偏置）、alpha/beta
    bool opGemm(const Node& n, std::string* err);
    bool opGlobalPool(const Node& n, std::string* err);
    bool opReduce(const Node& n, std::string* err);

    // -----------------------------------------------------------------------
    // CPU 回退路径
    //
    // 后处理里的算子（TopK / GatherElements / Gather / Expand / Tile / ArgMax）
    // 张量都很小（最大的 GatherElements 也就 2.5MB），而且大量用 int64 索引 ——
    // 在 GPU 上写这些内核不划算（还没算上调试成本）。所以走
    // 「下到主机 -> CPU 算 -> 传回去」这一步。
    //
    // 代价是每帧多几次 device<->host 同步，但实测这些算子只占 GPU 时间的 <1%。
    // -----------------------------------------------------------------------
    bool readTensor(int idx, std::vector<float>& out, std::string* err);
    bool writeTensor(int idx, const std::vector<float>& in, std::string* err);

    bool opTopK(const Node& n, std::string* err);
    bool opGatherElements(const Node& n, std::string* err);
    bool opGather(const Node& n, std::string* err);
    bool opExpand(const Node& n, std::string* err);
    bool opTile(const Node& n, std::string* err);
    bool opArgMax(const Node& n, std::string* err);

    // ConvTranspose：这是真卷积（不是后处理小算子），走 GPU
    bool opConvTranspose(const Node& n, int nodeIdx, std::string* err);

    const Graph& g_;
    std::vector<cl_mem> buf_;      // 每个张量一份（alias 后共享）
    std::vector<char> owned_;      // 该下标是否需要 release

    // alias 归并：view 算子的输出指向输入的根
    std::vector<int> aliasRoot_;
    int findRoot(int i);

    // 复用池：根下标 -> cl_mem
    std::map<int, cl_mem> rootBuf_;

    // 走 GEMM 的 Conv：节点下标 -> 重排后的权重 [KH*KW][C][K]
    // 重排让 A 子块的全局读取完全合并；这只是一次性的内存拷贝，不是文件格式转换
    std::map<int, cl_mem> packedWeight_;

    // -----------------------------------------------------------------------
    // 临时缓冲池（按字节数索引）
    //
    // 目前只有 ConvTranspose 用：它需要一个中间张量承载 1x1 卷积的输出。
    // 以前每帧 alloc + release 一次 —— 显存分配本身不贵，但每帧都做完全没必要，
    // 而且频繁分配/释放容易造成显存碎片。按尺寸缓存后稳态下零分配。
    // -----------------------------------------------------------------------
    std::map<size_t, cl_mem> scratchPool_;
    cl_mem acquireScratch(size_t bytes, std::string* err);

    size_t deviceBytes_ = 0;
    bool initialized_ = false;

    // 性能统计。默认关闭！
    // 打开后每次 kernel 启动都要 clWaitForEvents + 两次 clGetEventProfilingInfo，
    // 实测在 AMD 上会让每次启动多出 ~90us 的 host 同步开销（199 次启动就是 18ms），
    // 所以只在做性能分析时开。
    bool profiling_ = false;
    std::map<std::string, std::pair<int, double>> profile_;  // opType -> (次数, 总 ms)
    std::vector<NodeProfile> nodeProfile_;                   // 按节点下标索引
};

}  // namespace onnxcl

// 极简 OpenCL 运行时封装。
//
// 设计取舍（针对 gfx1035 / RDNA2 的桌面核显，只跑 YOLO）：
//   · 只用 buffer（cl_mem），不用 image —— 省掉一半代码量，AMD 上 buffer 路径已经够快
//   · OpenCL 通过运行时加载（cl_api.h），不依赖 OpenCL.lib / OpenCL SDK
//   · 内核源码在构建时嵌入；同时允许用环境变量 ONNXCL_KERNELS 指到磁盘目录，
//     方便改 kernel 时不用重新编译
//   · 单 context / 单 in-order queue，够用
#pragma once

#include "onnxcl/cl_api.h"

#include <map>
#include <string>
#include <vector>

namespace onnxcl {

class ClRuntime {
public:
    static ClRuntime& instance();

    bool init(std::string* err, int deviceIndex = 0);
    bool ok() const { return context_ != nullptr; }

    cl_platform_id platform() const { return platform_; }
    cl_device_id device() const { return device_; }
    cl_context context() const { return context_; }
    cl_command_queue queue() const { return queue_; }

    const std::string& platformName() const { return platformName_; }
    const std::string& deviceName() const { return deviceName_; }
    const std::string& deviceVersion() const { return deviceVersion_; }
    cl_ulong localMemBytes() const { return localMemBytes_; }
    cl_ulong maxWorkGroupSize() const { return maxWorkGroupSize_; }
    cl_uint computeUnits() const { return computeUnits_; }

    // ---- program / kernel ----
    // 一个 .cl 文件 = 一个 program（按 programName 缓存），里面可以有多个 kernel。
    // programName 同时是嵌入表里的键（不带 .cl 后缀）。
    cl_kernel getKernel(const std::string& programName, const std::string& kernelName, std::string* err);

    // 强制重新编译（调试内核时用）
    void invalidatePrograms() { programs_.clear(); }

    // ---- buffer ----
    cl_mem alloc(size_t bytes, std::string* err);
    cl_mem allocWithData(const void* data, size_t bytes, std::string* err);
    void release(cl_mem m);

    bool upload(cl_mem m, const void* data, size_t bytes, std::string* err);
    bool download(cl_mem m, void* data, size_t bytes, std::string* err);
    // 显存内拷贝（Concat 通道拼接走的快速路径：不启动内核，直接用 DMA）
    bool copyBuffer(cl_mem src, cl_mem dst, size_t bytes, size_t srcOffset, size_t dstOffset,
                    std::string* err);
    bool fillZero(cl_mem m, size_t bytes, std::string* err);
    bool finish(std::string* err);
    bool flush();

    // ---- 计时 ----
    // 打开后每次 kernel 启动都会带 event 并累计 GPU 时间（会强制同步，仅用于性能分析）
    void setProfiling(bool on) { profiling_ = on; }
    bool profiling() const { return profiling_; }
    double totalKernelMs() const { return totalMs_; }

    // 统计：launch 次数 / 总 GPU 时间
    int launchCount() const { return launchCount_; }
    void resetStats() { launchCount_ = 0; totalMs_ = 0; }
    void addKernelTime(double ms) {
        totalMs_ += ms;
        launchCount_++;
    }

    // 驱动对该内核允许的最大 work-group（可能小于设备上限，比如寄存器压力大的 conv）
    size_t kernelMaxWG(cl_kernel k);

    // 打印内核编译缓存命中统计。必须在内核真正编译完之后调 ——
    // 内核是 Executor::init 第 5 步预编译的，在 ClRuntime::init 末尾时计数还是 0。
    void reportCacheStat() const;

    ~ClRuntime();

private:
    ClRuntime() = default;
    ClRuntime(const ClRuntime&) = delete;
    ClRuntime& operator=(const ClRuntime&) = delete;

    cl_program buildFromSource(const std::string& source, const std::string& tag, std::string* err);

    // -----------------------------------------------------------------------
    // 内核编译产物的磁盘缓存
    //
    // AMD 的 OpenCL 编译器很慢（本机 8 个内核约 3.7 秒，占了整个初始化的 99%）。
    // 把 clBuildProgram 的结果（CL_PROGRAM_BINARIES）落盘，下次启动直接
    // clCreateProgramWithBinary，初始化从秒级降到百毫秒级。
    //
    // 缓存键包含源码 + 编译选项 + 设备名 + 设备版本 + 驱动版本 —— 驱动一升级
    // 键就变了，不会拿到过期的二进制。也可以用 ONNXCL_NO_CACHE=1 彻底关掉。
    // -----------------------------------------------------------------------
    std::string programCacheFile(const std::string& source, const std::string& options,
                                 const std::string& tag) const;
    cl_program loadCachedProgram(const std::string& path, const std::string& tag) const;
    void saveProgramBinary(cl_program prog, const std::string& path) const;

    // 日志：只在第一次真正编译/命中缓存时打一行，避免刷屏
    int cacheHits_ = 0;
    int cacheMisses_ = 0;
    const char* lookupSource(const std::string& name) const;

    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;

    std::string platformName_, deviceName_, deviceVersion_;
    cl_ulong localMemBytes_ = 0;
    cl_ulong maxWorkGroupSize_ = 0;
    cl_uint computeUnits_ = 0;

    std::map<std::string, cl_program> programs_;
    std::map<std::string, cl_kernel> kernels_;
    std::map<cl_kernel, size_t> kernelMaxWG_;

    cl_event timerEvent_ = nullptr;
    double totalMs_ = 0.0;
    int launchCount_ = 0;
    bool profiling_ = true;
    bool cacheEnabled_ = true;
};

// 便捷的 kernel 启动器：自动做 clSetKernelArg 类型分派 + 错误检查 + 计时
class Kernel {
public:
    Kernel() = default;
    Kernel(cl_kernel k, const char* name) : k_(k), name_(name) {}

    void set(int idx, cl_mem m);
    void set(int idx, int v);
    void set(int idx, float v);
    void set(int idx, cl_uint v);
    void setLocal(int idx, size_t bytes);
    // 传结构体参数（cl_int4 / cl_int2 / cl_float2 等）
    void setRaw(int idx, size_t size, const void* p) {
        const cl_int e = clSetKernelArg(k_, idx, size, p);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "[onnxcl] set arg %d (raw %zuB) on %s 失败: %d\n", idx, size, name_,
                         (int)e);
        }
    }

    // gws/lws 长度 1~3（不用的维度传 0）
    bool run(size_t g0, size_t g1 = 0, size_t g2 = 0,
             size_t l0 = 0, size_t l1 = 0, size_t l2 = 0, std::string* err = nullptr);

    cl_kernel raw() const { return k_; }

private:
    cl_kernel k_ = nullptr;
    const char* name_ = "";
};

// 取嵌入的内核源码；找不到返回 nullptr
const char* embeddedKernelSource(const std::string& name);

}  // namespace onnxcl

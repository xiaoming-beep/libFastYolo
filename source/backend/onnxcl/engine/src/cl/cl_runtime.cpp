#include "onnxcl/cl_runtime.h"

// 说明：clXxx(...) -> 函数指针表的重定向宏在 cl_api.h 末尾。
// 放在头文件里而不是本文件，是因为 cl_runtime.h 里有 inline 的 Kernel::setRaw()
// 会调 clSetKernelArg，而 inline 函数是在使用方（executor.cpp）展开的。

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

// 创建缓存目录用（不用 std::filesystem，保持 C++14 兼容）
#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace onnxcl {

// ---------------------------------------------------------------------------
// 错误字符串
// ---------------------------------------------------------------------------

static const char* clErrStr(cl_int e) {
    switch (e) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        default: return "CL_UNKNOWN_ERROR";
    }
}

static void setErr(std::string* err, const char* fmt, ...) {
    if (err == nullptr) return;
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    *err = buf;
}

static bool checkCl(cl_int e, const char* what, std::string* err) {
    if (e == CL_SUCCESS) return true;
    setErr(err, "%s 失败: %s (%d)", what, clErrStr(e), e);
    return false;
}

// ---------------------------------------------------------------------------
// 设备信息小工具
// ---------------------------------------------------------------------------

static std::string deviceStr(cl_device_id d, cl_device_info param) {
    size_t n = 0;
    if (clGetDeviceInfo(d, param, 0, nullptr, &n) != CL_SUCCESS || n == 0) return {};
    std::string s(n, '\0');
    if (clGetDeviceInfo(d, param, n, &s[0], nullptr) != CL_SUCCESS) return {};
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

template <typename T>
static T deviceScalar(cl_device_id d, cl_device_info param, T def = T(0)) {
    T v{};
    if (clGetDeviceInfo(d, param, sizeof(T), &v, nullptr) != CL_SUCCESS) return def;
    return v;
}

// ---------------------------------------------------------------------------
// 源码查找：环境变量目录优先，其次嵌入表
// ---------------------------------------------------------------------------

const char* ClRuntime::lookupSource(const std::string& name) const {
    static const char* envDir = std::getenv("ONNXCL_KERNELS");
    if (envDir != nullptr && envDir[0] != '\0') {
        std::string path = std::string(envDir) + "/" + name + ".cl";
        std::ifstream f(path, std::ios::binary);
        if (f) {
            // 故意 leak：kernel 源码在进程生命期内一直要用
            std::string* src = new std::string((std::istreambuf_iterator<char>(f)),
                                               std::istreambuf_iterator<char>());
            std::fprintf(stderr, "[onnxcl] 使用磁盘内核: %s (%zu 字节)\n", path.c_str(), src->size());
            return src->c_str();
        }
    }
    return embeddedKernelSource(name);
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

ClRuntime& ClRuntime::instance() {
    // 故意 leak，不在进程退出时析构。
    //
    // 原因：本单例持有 cl_context / cl_command_queue / cl_program / cl_kernel。
    // 如果让静态析构函数在进程退出时去 clRelease*，会有两个问题：
    //   1. 此时 OpenCL 运行时（OpenCL.dll / libOpenCL.so）可能已经被卸载，
    //      调用它的函数指针就是野指针；
    //   2. 析构发生在 DLL_PROCESS_DETACH 里，此时持有 loader lock，
    //      在锁内调外部库很容易死锁或直接崩掉。
    // 实测症状：主函数全部走完、"完成"也打印了，但进程退出码是 0xC0000409
    // （宿主是 JNI 的话就是整个 JVM 挂掉，而且栈上看不出原因）。
    //
    // 进程退出时操作系统会回收全部资源（含显存），所以 leak 是安全且推荐的做法。
    static ClRuntime* inst = new ClRuntime();
    return *inst;
}

bool ClRuntime::init(std::string* err, int deviceIndex) {
    if (context_ != nullptr) return true;

    // 先把 OpenCL 运行时加载起来（Windows: OpenCL.dll / Linux: libOpenCL.so.1）。
    // 不依赖 OpenCL.lib，所以没有装 OpenCL SDK 的机器也能编、也能跑
    if (!ensureClApiLoaded(err)) {
        return false;
    }

    // 内核编译缓存开关：ONNXCL_NO_CACHE=1 可以彻底关掉（排查编译问题时用）
    {
        const char* noCache = std::getenv("ONNXCL_NO_CACHE");
        cacheEnabled_ = !(noCache != nullptr && noCache[0] == '1');
    }

    cl_int e = CL_SUCCESS;

    // 1) 平台
    cl_uint nplat = 0;
    if (!checkCl(clGetPlatformIDs(0, nullptr, &nplat), "clGetPlatformIDs", err)) return false;
    if (nplat == 0) {
        setErr(err, "系统里没有 OpenCL 平台");
        return false;
    }
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);

    // 2) 挑设备：优先 GPU，找不到再退回任意设备
    std::vector<std::pair<cl_platform_id, cl_device_id>> candidates;
    for (cl_platform_id p : plats) {
        cl_uint ndev = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &ndev) == CL_SUCCESS && ndev > 0) {
            std::vector<cl_device_id> ds(ndev);
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, ndev, ds.data(), nullptr);
            for (cl_device_id d : ds) candidates.push_back({p, d});
        }
    }
    if (candidates.empty()) {
        for (cl_platform_id p : plats) {
            cl_uint ndev = 0;
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &ndev) == CL_SUCCESS && ndev > 0) {
                std::vector<cl_device_id> ds(ndev);
                clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, ndev, ds.data(), nullptr);
                for (cl_device_id d : ds) candidates.push_back({p, d});
            }
        }
    }
    if (candidates.empty()) {
        setErr(err, "没有可用的 OpenCL 设备");
        return false;
    }
    if (deviceIndex < 0 || deviceIndex >= static_cast<int>(candidates.size())) {
        setErr(err, "deviceIndex=%d 越界，只有 %zu 个设备", deviceIndex, candidates.size());
        return false;
    }
    platform_ = candidates[deviceIndex].first;
    device_ = candidates[deviceIndex].second;

    // 3) 设备属性
    {
        size_t n = 0;
        clGetPlatformInfo(platform_, CL_PLATFORM_NAME, 0, nullptr, &n);
        platformName_.resize(n);
        clGetPlatformInfo(platform_, CL_PLATFORM_NAME, n, &platformName_[0], nullptr);
        while (!platformName_.empty() && platformName_.back() == '\0') platformName_.pop_back();
    }
    deviceName_ = deviceStr(device_, CL_DEVICE_NAME);
    deviceVersion_ = deviceStr(device_, CL_DEVICE_VERSION);
    localMemBytes_ = deviceScalar<cl_ulong>(device_, CL_DEVICE_LOCAL_MEM_SIZE);
    maxWorkGroupSize_ = deviceScalar<size_t>(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    computeUnits_ = deviceScalar<cl_uint>(device_, CL_DEVICE_MAX_COMPUTE_UNITS);

    // 4) context
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &e);
    if (!checkCl(e, "clCreateContext", err)) {
        context_ = nullptr;
        return false;
    }

    // 5) queue（带 profiling，方便后面做 per-op 性能分析）
    {
        cl_queue_properties props[] = {CL_QUEUE_PROPERTIES,
                                       static_cast<cl_queue_properties>(CL_QUEUE_PROFILING_ENABLE), 0};
        // clCreateCommandQueueWithProperties 是 OpenCL 2.0 才有的符号；
        // 老运行时上取不到（指针为空），直接走下面的老接口回退
        if (clCreateCommandQueueWithProperties != nullptr) {
            queue_ = clCreateCommandQueueWithProperties(context_, device_, props, &e);
        } else {
            e = CL_INVALID_OPERATION;
            queue_ = nullptr;
        }
        if (e != CL_SUCCESS || queue_ == nullptr) {
            // 退回到老接口
            queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &e);
            if (!checkCl(e, "clCreateCommandQueue", err)) {
                queue_ = nullptr;
                return false;
            }
        }
    }

    // 缓存统计不在这里打：内核是 Executor::init 第 5 步才预编译的，
    // 此时编译计数还是 0。由 Executor::init 末尾调用 reportCacheStat()。
    return true;
}

ClRuntime::~ClRuntime() {
    for (auto& kv : kernels_) clReleaseKernel(kv.second);
    for (auto& kv : programs_) clReleaseProgram(kv.second);
    if (queue_ != nullptr) clReleaseCommandQueue(queue_);
    if (context_ != nullptr) clReleaseContext(context_);
}

// ---------------------------------------------------------------------------
// program / kernel
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 内核编译产物的磁盘缓存
//
// AMD 的 OpenCL 编译器很慢（本机 8 个内核约 3.7 秒，独占整个初始化的 99%），
// 所以把 clBuildProgram 的结果落盘，下次启动直接 clCreateProgramWithBinary。
// ---------------------------------------------------------------------------
namespace {

// 缓存目录：ONNXCL_CACHE_DIR > 系统标准位置
std::string onnxclCacheDir() {
    const char* env = std::getenv("ONNXCL_CACHE_DIR");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
#if defined(_WIN32)
    const char* base = std::getenv("LOCALAPPDATA");
    return (base != nullptr && base[0] != '\0') ? (std::string(base) + "/onnxcl_cache")
                                                 : std::string("onnxcl_cache");
#else
    const char* home = std::getenv("HOME");
    return (home != nullptr && home[0] != '\0') ? (std::string(home) + "/.cache/onnxcl")
                                                 : std::string("/tmp/onnxcl_cache");
#endif
}

// 逐级创建目录（不返回错误：建不出来就是缓存不生效，不影响功能）
void onnxclMakeDirs(const std::string& path) {
    for (size_t i = 1; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/' || path[i] == '\\') {
            const std::string sub = path.substr(0, i);
            if (sub.empty()) continue;
#if defined(_WIN32)
            _mkdir(sub.c_str());
#else
            mkdir(sub.c_str(), 0755);
#endif
        }
    }
}

uint64_t onnxclFnv1a(const void* data, size_t len, uint64_t h) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

}  // namespace

std::string ClRuntime::programCacheFile(const std::string& source, const std::string& options,
                                        const std::string& tag) const {
    // 缓存键必须同时包含源码、编译选项和设备身份：
    //   · 只哈希源码不够 —— 同一份源码在不同 -D 选项下产出的二进制不同
    //   · 必须带设备名/设备版本/驱动版本 —— 二进制是驱动相关的，驱动一升级就得重编
    uint64_t h = 1469598103934665603ULL;
    h = onnxclFnv1a(source.data(), source.size(), h);
    h = onnxclFnv1a(options.data(), options.size(), h);
    h = onnxclFnv1a(deviceName_.data(), deviceName_.size(), h);
    h = onnxclFnv1a(deviceVersion_.data(), deviceVersion_.size(), h);
    const std::string drv = deviceStr(device_, CL_DRIVER_VERSION);
    h = onnxclFnv1a(drv.data(), drv.size(), h);
    h = onnxclFnv1a(tag.data(), tag.size(), h);

    char name[64];
    std::snprintf(name, sizeof(name), "%016llx.bin", static_cast<unsigned long long>(h));
    return onnxclCacheDir() + "/" + name;
}

cl_program ClRuntime::loadCachedProgram(const std::string& path, const std::string& tag) const {
    std::ifstream f(path, std::ios::binary);
    if (!f) return nullptr;
    const std::string blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (blob.empty()) return nullptr;

    cl_int e = CL_SUCCESS;
    const unsigned char* data = reinterpret_cast<const unsigned char*>(blob.data());
    const size_t len = blob.size();
    cl_program prog = clCreateProgramWithBinary(context_, 1, &device_, &len, &data, nullptr, &e);
    if (e != CL_SUCCESS || prog == nullptr) return nullptr;
    e = clBuildProgram(prog, 1, &device_, "", nullptr, nullptr);
    if (e != CL_SUCCESS) {
        // 缓存坏了（版本不匹配、文件截断）就当没命中，走正常编译
        clReleaseProgram(prog);
        std::fprintf(stderr, "[onnxcl] 缓存二进制不可用(%s)，重新编译: %s\n", tag.c_str(), path.c_str());
        return nullptr;
    }
    return prog;
}

void ClRuntime::saveProgramBinary(cl_program prog, const std::string& path) const {
    size_t binSize = 0;
    if (clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(binSize), &binSize, nullptr) !=
        CL_SUCCESS) {
        return;
    }
    if (binSize == 0) return;

    std::vector<unsigned char> buf(binSize);
    unsigned char* p = buf.data();
    if (clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(p), &p, nullptr) != CL_SUCCESS) {
        return;
    }

    onnxclMakeDirs(onnxclCacheDir());
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return;
    f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

void ClRuntime::reportCacheStat() const {
    if (!cacheEnabled_) {
        std::fprintf(stderr, "[onnxcl] 内核编译缓存: 已关闭\n");
        return;
    }
    std::fprintf(stderr, "[onnxcl] 内核编译缓存: 命中 %d, 新编译 %d  (目录 %s)\n", cacheHits_,
                 cacheMisses_, onnxclCacheDir().c_str());
}

cl_program ClRuntime::buildFromSource(const std::string& source, const std::string& tag, std::string* err) {
    // -----------------------------------------------------------------------
    // 编译选项
    //
    // 【-cl-mad-enable】允许 a*b+c 融合成一条 MAD。这是安全的：MAD 只做一次舍入，
    //   比“先乘再单独加”精度还略好；AMD 上通常有 10~20% 的收益。
    //
    // 【-cl-fast-relaxed-math】没开。它允许重结合浮点运算、不处理 NaN/Inf，
    //   收益不确定但会损失精度 —— 我们的对拍容差是 rel 1e-4，而实测余量在 1e-6
    //   量级，理论上吃得下，但需要单独验证后再决定。
    //
    // 注意：选项字符串会参与缓存键（见 programCacheFile），改了会自动失效重编。
    // -----------------------------------------------------------------------
    const char* options = "-cl-mad-enable";

    // ---- 1. 先试磁盘缓存 ----
    std::string cacheFile;
    if (cacheEnabled_) {
        cacheFile = programCacheFile(source, options, tag);
        if (cl_program cached = loadCachedProgram(cacheFile, tag)) {
            cacheHits_++;
            return cached;
        }
    }
    cacheMisses_++;

    // ---- 2. 老老实实编译 ----
    cl_int e = CL_SUCCESS;
    const char* src = source.c_str();
    const size_t len = source.size();
    cl_program prog = clCreateProgramWithSource(context_, 1, &src, &len, &e);
    if (!checkCl(e, ("clCreateProgramWithSource(" + tag + ")").c_str(), err)) return nullptr;

    e = clBuildProgram(prog, 1, &device_, options, nullptr, nullptr);
    if (e != CL_SUCCESS) {
        size_t logLen = 0;
        clGetProgramBuildInfo(prog, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLen);
        std::string log(logLen + 1, '\0');
        clGetProgramBuildInfo(prog, device_, CL_PROGRAM_BUILD_LOG, logLen, &log[0], nullptr);
        setErr(err, "内核 %s 编译失败 (%s):\n%s", tag.c_str(), clErrStr(e), log.c_str());
        clReleaseProgram(prog);
        return nullptr;
    }

    // ---- 3. 存下来给下次用 ----
    if (cacheEnabled_) {
        saveProgramBinary(prog, cacheFile);
    }
    return prog;
}

cl_kernel ClRuntime::getKernel(const std::string& programName, const std::string& kernelName,
                               std::string* err) {
    const std::string key = programName + "::" + kernelName;
    auto kit = kernels_.find(key);
    if (kit != kernels_.end()) return kit->second;

    auto pit = programs_.find(programName);
    if (pit == programs_.end()) {
        const char* src = lookupSource(programName);
        if (src == nullptr) {
            setErr(err, "找不到内核源码 '%s'（嵌入表里没有，ONNXCL_KERNELS 目录里也没有）",
                   programName.c_str());
            return nullptr;
        }
        cl_program prog = buildFromSource(src, programName, err);
        if (prog == nullptr) return nullptr;
        pit = programs_.emplace(programName, prog).first;
    }

    cl_int e = CL_SUCCESS;
    cl_kernel k = clCreateKernel(pit->second, kernelName.c_str(), &e);
    if (e != CL_SUCCESS) {
        setErr(err, "clCreateKernel(%s in %s) 失败: %s", kernelName.c_str(), programName.c_str(),
               clErrStr(e));
        return nullptr;
    }
    kernels_[key] = k;
    return k;
}

// ---------------------------------------------------------------------------
// buffer
// ---------------------------------------------------------------------------

cl_mem ClRuntime::alloc(size_t bytes, std::string* err) {
    cl_int e = CL_SUCCESS;
    cl_mem m = clCreateBuffer(context_, CL_MEM_READ_WRITE, bytes, nullptr, &e);
    if (!checkCl(e, "clCreateBuffer", err)) return nullptr;
    return m;
}

cl_mem ClRuntime::allocWithData(const void* data, size_t bytes, std::string* err) {
    cl_int e = CL_SUCCESS;
    cl_mem m = clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes,
                              const_cast<void*>(data), &e);
    if (!checkCl(e, "clCreateBuffer(copy)", err)) return nullptr;
    return m;
}

void ClRuntime::release(cl_mem m) {
    if (m != nullptr) clReleaseMemObject(m);
}

bool ClRuntime::upload(cl_mem m, const void* data, size_t bytes, std::string* err) {
    if (m == nullptr) return checkCl(CL_INVALID_MEM_OBJECT, "upload", err);
    return checkCl(clEnqueueWriteBuffer(queue_, m, CL_TRUE, 0, bytes, data, 0, nullptr, nullptr),
              "clEnqueueWriteBuffer", err);
}

bool ClRuntime::download(cl_mem m, void* data, size_t bytes, std::string* err) {
    if (m == nullptr) return checkCl(CL_INVALID_MEM_OBJECT, "download", err);
    return checkCl(clEnqueueReadBuffer(queue_, m, CL_TRUE, 0, bytes, data, 0, nullptr, nullptr),
              "clEnqueueReadBuffer", err);
}

bool ClRuntime::copyBuffer(cl_mem src, cl_mem dst, size_t bytes, size_t srcOffset,
                           size_t dstOffset, std::string* err) {
    if (src == nullptr || dst == nullptr) {
        return checkCl(CL_INVALID_MEM_OBJECT, "copyBuffer", err);
    }
    return checkCl(clEnqueueCopyBuffer(queue_, src, dst, srcOffset, dstOffset, bytes, 0, nullptr,
                                       nullptr),
                   "clEnqueueCopyBuffer", err);
}

bool ClRuntime::fillZero(cl_mem m, size_t bytes, std::string* err) {
    const unsigned char zero = 0;
    return checkCl(clEnqueueFillBuffer(queue_, m, &zero, 1, 0, bytes, 0, nullptr, nullptr),
              "clEnqueueFillBuffer", err);
}

bool ClRuntime::finish(std::string* err) {
    return checkCl(clFinish(queue_), "clFinish", err);
}

bool ClRuntime::flush() {
    return clFlush(queue_) == CL_SUCCESS;
}

// ---------------------------------------------------------------------------
// Kernel 启动
// ---------------------------------------------------------------------------

size_t ClRuntime::kernelMaxWG(cl_kernel k) {
    auto it = kernelMaxWG_.find(k);
    if (it != kernelMaxWG_.end()) return it->second;
    size_t v = maxWorkGroupSize_;
    size_t got = 0;
    if (clGetKernelWorkGroupInfo(k, device_, CL_KERNEL_WORK_GROUP_SIZE, sizeof(v), &v, &got) !=
        CL_SUCCESS) {
        v = maxWorkGroupSize_;
    }
    // 再加一层保护：绝不超过设备上限
    if (v == 0 || v > maxWorkGroupSize_) v = maxWorkGroupSize_;
    if (v == 0) v = 1;
    kernelMaxWG_[k] = v;
    return v;
}

void Kernel::set(int idx, cl_mem m) {
    cl_int e = clSetKernelArg(k_, idx, sizeof(cl_mem), &m);
    if (e != CL_SUCCESS) {
        std::fprintf(stderr, "[onnxcl] set arg %d (mem) on %s 失败: %s\n", idx, name_, clErrStr(e));
    }
}

void Kernel::set(int idx, int v) {
    cl_int e = clSetKernelArg(k_, idx, sizeof(cl_int), &v);
    if (e != CL_SUCCESS) {
        std::fprintf(stderr, "[onnxcl] set arg %d (int) on %s 失败: %s\n", idx, name_, clErrStr(e));
    }
}

void Kernel::set(int idx, cl_uint v) {
    cl_int e = clSetKernelArg(k_, idx, sizeof(cl_uint), &v);
    if (e != CL_SUCCESS) {
        std::fprintf(stderr, "[onnxcl] set arg %d (uint) on %s 失败: %s\n", idx, name_, clErrStr(e));
    }
}

void Kernel::set(int idx, float v) {
    cl_int e = clSetKernelArg(k_, idx, sizeof(cl_float), &v);
    if (e != CL_SUCCESS) {
        std::fprintf(stderr, "[onnxcl] set arg %d (float) on %s 失败: %s\n", idx, name_, clErrStr(e));
    }
}

void Kernel::setLocal(int idx, size_t bytes) {
    cl_int e = clSetKernelArg(k_, idx, bytes, nullptr);
    if (e != CL_SUCCESS) {
        std::fprintf(stderr, "[onnxcl] set arg %d (local %zuB) on %s 失败: %s\n", idx, bytes, name_,
                     clErrStr(e));
    }
}

bool Kernel::run(size_t g0, size_t g1, size_t g2, size_t l0, size_t l1, size_t l2, std::string* err) {
    if (k_ == nullptr) {
        setErr(err, "kernel '%s' 是空的", name_);
        return false;
    }

    size_t gws[3] = {g0, g1 ? g1 : 1, g2 ? g2 : 1};
    size_t* lwsPtr = nullptr;
    size_t lws[3] = {l0, l1 ? l1 : 1, l2 ? l2 : 1};
    if (l0 != 0) {
        ClRuntime& rt = ClRuntime::instance();
        // 不能想当然用 256：驱动对每个内核给出的 CL_KERNEL_WORK_GROUP_SIZE 可能更小
        // （寄存器/共享内存压力大的内核尤其如此，比如 conv2d）
        const size_t kmax = rt.kernelMaxWG(k_);
        size_t prod = lws[0] * lws[1] * lws[2];
        if (prod > kmax) {
            size_t other = lws[1] * lws[2];
            size_t limit = kmax / (other ? other : 1);
            if (limit == 0) limit = 1;
            while (lws[0] > limit && lws[0] % 2 == 0) lws[0] /= 2;
            if (lws[0] > limit) lws[0] = limit;
        }
        // global 必须是 local 的整数倍。内核里都有 gid >= total 的边界检查，
        // 所以向上取整引入的多余线程是安全的。
        for (int i = 0; i < 3; ++i) {
            if (lws[i] == 0) lws[i] = 1;
            if (gws[i] % lws[i] != 0) {
                gws[i] = ((gws[i] + lws[i] - 1) / lws[i]) * lws[i];
            }
        }
        lwsPtr = lws;
    }

    int dim = 1;
    if (g2 != 0) dim = 3;
    else if (g1 != 0) dim = 2;

    ClRuntime& rt = ClRuntime::instance();
    cl_event ev = nullptr;
    cl_int e = clEnqueueNDRangeKernel(rt.queue(), k_, dim, nullptr, gws, lwsPtr, 0, nullptr,
                                      rt.profiling() ? &ev : nullptr);
    if (!checkCl(e, (std::string("clEnqueueNDRangeKernel(") + name_ + ")").c_str(), err)) return false;

    if (ev != nullptr) {
        clWaitForEvents(1, &ev);
        cl_ulong t0 = 0, t1 = 0;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        rt.addKernelTime(static_cast<double>(t1 - t0) / 1e6);
        clReleaseEvent(ev);
    }
    return true;
}

}  // namespace onnxcl

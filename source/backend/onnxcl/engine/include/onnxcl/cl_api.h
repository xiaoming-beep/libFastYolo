// OpenCL 运行时加载器
//
// 为什么不直接链接 OpenCL.lib：
//   Windows + MSVC 下没有安装 OpenCL SDK 就没有导入库（OpenCL.lib），系统只自带
//   OpenCL.dll。MinGW 可以直链 DLL，但 MSVC 不行（必须用导入库）。Linux 下虽然
//   find_library(OpenCL) 通常能找到，但也不是所有发行版默认带 -dev 包。
//
//   所以统一改成运行时加载（Windows: LoadLibrary，POSIX: dlopen），好处：
//     · 任何编译器/平台都不需要 SDK，只要系统有 OpenCL 运行时就能编译和运行
//     · OpenCL.dll / libOpenCL.so 缺失时能给出明确的中文错误，而不是"加载库就失败"
//     · 和 MNN 的 MNN_USE_LIB_WRAPPER 是同一个思路，属于成熟做法
//
// 用法：包含本头文件后，代码里照常写 clXxx(...) 即可 —— 下面的宏会把它重定向到
//       函数指针。函数指针表用 X-macro 维护，加新函数只要在 ONNXCL_CL_FUNCS 里加一行。
#ifndef LIBFASTYOLO_ONNXCL_CL_API_H
#define LIBFASTYOLO_ONNXCL_CL_API_H

// ↓ 必须在 #include <CL/cl.h> 之前：决定头文件里声明哪些版本的 API。
//   放到后面的话 cl.h 会按 1.0 配置，clCreateCommandQueue / clEnqueueFillBuffer
//   这些就不会被声明，编译直接报未定义。
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 200
#endif
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif

#include <CL/cl.h>

#include <string>

namespace onnxcl {

// 需要从运行时取的函数清单。加函数只要在这里加一行。
// 用 decltype(&::clXxx) 直接复用 cl.h 里的函数签名（含调用约定，如 Windows 的
// CL_API_CALL=__stdcall），不用手抄一遍参数表。
#define ONNXCL_CL_FUNCS(X)                    \
    X(clGetPlatformIDs)                       \
    X(clGetPlatformInfo)                      \
    X(clGetDeviceIDs)                         \
    X(clGetDeviceInfo)                        \
    X(clCreateContext)                        \
    X(clCreateCommandQueue)                   \
    X(clCreateProgramWithSource)              \
    X(clCreateProgramWithBinary)              \
    X(clBuildProgram)                         \
    X(clGetProgramBuildInfo)                  \
    X(clGetProgramInfo)                       \
    X(clCreateKernel)                         \
    X(clSetKernelArg)                         \
    X(clEnqueueNDRangeKernel)                 \
    X(clCreateBuffer)                         \
    X(clReleaseMemObject)                     \
    X(clEnqueueWriteBuffer)                   \
    X(clEnqueueReadBuffer)                    \
    X(clEnqueueCopyBuffer)                    \
    X(clEnqueueFillBuffer)                    \
    X(clFinish)                               \
    X(clFlush)                                \
    X(clReleaseKernel)                        \
    X(clReleaseProgram)                       \
    X(clReleaseCommandQueue)                  \
    X(clReleaseContext)                       \
    X(clWaitForEvents)                        \
    X(clGetEventProfilingInfo)                \
    X(clReleaseEvent)                         \
    X(clGetKernelWorkGroupInfo)

struct ClApi {
    // clCreateCommandQueue 在 OpenCL 2.0 里已弃用（推荐 WithProperties），
    // 但我们特意保留它作为 OpenCL 1.x 设备的回退路径，所以定向屏蔽这里的弃用告警。
    // MSVC 不加 -Wdeprecated-declarations，不需要处理。
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#define X(name) decltype(&::name) name = nullptr;
    ONNXCL_CL_FUNCS(X)
#undef X
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

    // OpenCL 2.0 才有的可选符号：老运行时上没有就退回 clCreateCommandQueue
    decltype(&::clCreateCommandQueueWithProperties) clCreateCommandQueueWithProperties = nullptr;

    // 加载系统 OpenCL 运行时。失败时 err 里是中文原因。
    bool load(std::string* err);

    // 动态库句柄是否拿到了
    bool available() const { return handle_ != nullptr; }

    // 运行时描述（平台名 + 设备名），日志用
    const std::string& runtimeDesc() const { return runtimeDesc_; }
    void setRuntimeDesc(const std::string& d) { runtimeDesc_ = d; }

private:
    void* handle_ = nullptr;
    std::string runtimeDesc_;
};

// 全局唯一实例
extern ClApi clApi;

// 便捷入口：加载（幂等），失败时返回中文错误
bool ensureClApiLoaded(std::string* err);

}  // namespace onnxcl

// ---------------------------------------------------------------------------
// 重定向到函数指针
//
// 必须放在结构体定义之后 —— 否则上面的 decltype(&::clXxx) 会被这些宏替换掉。
//
// 放在本头文件（而不是 cl_runtime.cpp）是因为 cl_runtime.h 里有 inline 的
// Kernel::setRaw() 会调 clSetKernelArg；inline 函数是在使用方（如 executor.cpp）
// 展开的，宏只写在 cl_runtime.cpp 里的话，executor.cpp 就会去链真的 clSetKernelArg，
// 而我们没有也不需要 OpenCL.lib（MinGW 能直链 dll，MSVC 不行）。
//
// 加载器自身（cl_api.cpp）先定义 ONNXCL_CL_API_NO_REDIRECT 跳过这一段。
// ---------------------------------------------------------------------------
#ifndef ONNXCL_CL_API_NO_REDIRECT
#define clGetPlatformIDs ::onnxcl::clApi.clGetPlatformIDs
#define clGetPlatformInfo ::onnxcl::clApi.clGetPlatformInfo
#define clGetDeviceIDs ::onnxcl::clApi.clGetDeviceIDs
#define clGetDeviceInfo ::onnxcl::clApi.clGetDeviceInfo
#define clCreateContext ::onnxcl::clApi.clCreateContext
#define clCreateCommandQueue ::onnxcl::clApi.clCreateCommandQueue
#define clCreateCommandQueueWithProperties ::onnxcl::clApi.clCreateCommandQueueWithProperties
#define clCreateProgramWithSource ::onnxcl::clApi.clCreateProgramWithSource
#define clCreateProgramWithBinary ::onnxcl::clApi.clCreateProgramWithBinary
#define clBuildProgram ::onnxcl::clApi.clBuildProgram
#define clGetProgramBuildInfo ::onnxcl::clApi.clGetProgramBuildInfo
#define clGetProgramInfo ::onnxcl::clApi.clGetProgramInfo
#define clCreateKernel ::onnxcl::clApi.clCreateKernel
#define clSetKernelArg ::onnxcl::clApi.clSetKernelArg
#define clEnqueueNDRangeKernel ::onnxcl::clApi.clEnqueueNDRangeKernel
#define clCreateBuffer ::onnxcl::clApi.clCreateBuffer
#define clReleaseMemObject ::onnxcl::clApi.clReleaseMemObject
#define clEnqueueWriteBuffer ::onnxcl::clApi.clEnqueueWriteBuffer
#define clEnqueueReadBuffer ::onnxcl::clApi.clEnqueueReadBuffer
#define clEnqueueCopyBuffer ::onnxcl::clApi.clEnqueueCopyBuffer
#define clEnqueueFillBuffer ::onnxcl::clApi.clEnqueueFillBuffer
#define clFinish ::onnxcl::clApi.clFinish
#define clFlush ::onnxcl::clApi.clFlush
#define clReleaseKernel ::onnxcl::clApi.clReleaseKernel
#define clReleaseProgram ::onnxcl::clApi.clReleaseProgram
#define clReleaseCommandQueue ::onnxcl::clApi.clReleaseCommandQueue
#define clReleaseContext ::onnxcl::clApi.clReleaseContext
#define clWaitForEvents ::onnxcl::clApi.clWaitForEvents
#define clGetEventProfilingInfo ::onnxcl::clApi.clGetEventProfilingInfo
#define clReleaseEvent ::onnxcl::clApi.clReleaseEvent
#define clGetKernelWorkGroupInfo ::onnxcl::clApi.clGetKernelWorkGroupInfo
#endif  // ONNXCL_CL_API_NO_REDIRECT

#endif  // LIBFASTYOLO_ONNXCL_CL_API_H
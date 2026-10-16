// OpenCL 运行时加载器的实现（Windows: LoadLibrary / POSIX: dlopen）
//
// 本文件是加载器自身，要直接操作 ClApi 的结构体成员，所以必须先关掉重定向宏，
// 否则下面的 this->clXxx 会被宏替换成 this->::onnxcl::clApi.clXxx，语法就错了。
#define ONNXCL_CL_API_NO_REDIRECT

#include "onnxcl/cl_api.h"

#include <cstring>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace onnxcl {

ClApi clApi;  // 全局唯一实例

namespace {

#if defined(_WIN32)
// Windows: LoadLibrary/GetProcAddress。注意句柄在公共头里只能存成 void*
// （公共头不能引 windows.h，否则会污染所有包含它的编译单元），
// 所以这里是 HMODULE 和 void* 互转的边界。
using LibHandle = HMODULE;
inline LibHandle openLib(const char* name) { return ::LoadLibraryA(name); }
inline void* lookupSym(void* h, const char* name) {
    return reinterpret_cast<void*>(::GetProcAddress(static_cast<HMODULE>(h), name));
}
#else
using LibHandle = void*;
inline LibHandle openLib(const char* name) { return ::dlopen(name, RTLD_NOW | RTLD_GLOBAL); }
inline void* lookupSym(void* h, const char* name) { return ::dlsym(h, name); }
#endif

}  // namespace

bool ClApi::load(std::string* err) {
    if (handle_ != nullptr) {
        return true;  // 幂等
    }

    // Windows 只有 OpenCL.dll（ICD loader，由显卡驱动安装）；
    // Linux 通常是 libOpenCL.so.1（不带 -dev 包也有）
    const char* candidates[] = {
#if defined(_WIN32)
        "OpenCL.dll",
#elif defined(__APPLE__)
        "/System/Library/Frameworks/OpenCL.framework/OpenCL",
#else
        "libOpenCL.so.1",
        "libOpenCL.so",
#endif
    };
    for (const char* name : candidates) {
        handle_ = openLib(name);
        if (handle_ != nullptr) {
            break;
        }
    }
    if (handle_ == nullptr) {
        if (err != nullptr) {
            *err = "加载 OpenCL 运行时失败：找不到 OpenCL.dll（Windows）或 libOpenCL.so.1（Linux）。"
                   "请确认显卡驱动已正确安装（NVIDIA/AMD/Intel 驱动都自带 OpenCL 运行时）";
        }
        return false;
    }

    // 逐个取必需符号。这里用 this-> 显式访问结构体成员（不加会出现成员/全局函数
    // 同名歧义）；本文件没有重定向宏，不会出现"自己引用自己"的问题
    std::vector<const char*> missing;

    auto require = [&](auto& fn, const char* sym) {
        fn = reinterpret_cast<typename std::remove_reference<decltype(fn)>::type>(
            lookupSym(handle_, sym));
        if (fn == nullptr) {
            missing.push_back(sym);
        }
    };

    require(this->clGetPlatformIDs, "clGetPlatformIDs");
    require(this->clGetPlatformInfo, "clGetPlatformInfo");
    require(this->clGetDeviceIDs, "clGetDeviceIDs");
    require(this->clGetDeviceInfo, "clGetDeviceInfo");
    require(this->clCreateContext, "clCreateContext");
    require(this->clCreateCommandQueue, "clCreateCommandQueue");
    require(this->clCreateProgramWithSource, "clCreateProgramWithSource");
    require(this->clCreateProgramWithBinary, "clCreateProgramWithBinary");
    require(this->clBuildProgram, "clBuildProgram");
    require(this->clGetProgramBuildInfo, "clGetProgramBuildInfo");
    require(this->clGetProgramInfo, "clGetProgramInfo");
    require(this->clCreateKernel, "clCreateKernel");
    require(this->clSetKernelArg, "clSetKernelArg");
    require(this->clEnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
    require(this->clCreateBuffer, "clCreateBuffer");
    require(this->clReleaseMemObject, "clReleaseMemObject");
    require(this->clEnqueueWriteBuffer, "clEnqueueWriteBuffer");
    require(this->clEnqueueReadBuffer, "clEnqueueReadBuffer");
    require(this->clEnqueueCopyBuffer, "clEnqueueCopyBuffer");
    require(this->clEnqueueFillBuffer, "clEnqueueFillBuffer");
    require(this->clFinish, "clFinish");
    require(this->clFlush, "clFlush");
    require(this->clReleaseKernel, "clReleaseKernel");
    require(this->clReleaseProgram, "clReleaseProgram");
    require(this->clReleaseCommandQueue, "clReleaseCommandQueue");
    require(this->clReleaseContext, "clReleaseContext");
    require(this->clWaitForEvents, "clWaitForEvents");
    require(this->clGetEventProfilingInfo, "clGetEventProfilingInfo");
    require(this->clReleaseEvent, "clReleaseEvent");
    require(this->clGetKernelWorkGroupInfo, "clGetKernelWorkGroupInfo");

    // 可选符号：缺了不影响主流程
    clCreateCommandQueueWithProperties = reinterpret_cast<decltype(clCreateCommandQueueWithProperties)>(
        lookupSym(handle_, "clCreateCommandQueueWithProperties"));

    if (!missing.empty()) {
        if (err != nullptr) {
            *err = "OpenCL 运行时缺少必需符号: ";
            for (size_t i = 0; i < missing.size(); ++i) {
                if (i != 0) {
                    *err += ", ";
                }
                *err += missing[i];
            }
        }
        return false;
    }
    return true;
}

bool ensureClApiLoaded(std::string* err) {
    return clApi.load(err);
}

}  // namespace onnxcl

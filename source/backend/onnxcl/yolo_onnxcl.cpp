// yolo_onnxcl.cpp —— onnxcl（OpenCL 直读 ONNX）推理后端
//
// 本文件职责：
//   把 onnxcl 引擎适配成 libFastYolo 的后端 ABI（yolo_infer_create/init/predict/destroy）。
//   引擎本体在 engine/ 目录，是自包含的（不依赖 MNN / TensorRT / ONNX Runtime），
//   这里只做三件事：
//     模型字节  -> 引擎（onnx 直接解析，没有格式转换步骤）
//     Letterbox -> 输入张量（planar RGB float，核心层已经归一化好，直接传）
//     引擎输出  -> YoloInferOutput（张量由本对象持有，下次 predict 前有效）
//
// 文件结构清单：
//   OnnxClYolo              引擎持有者（图 + 执行器 + 输出缓冲）
//   yolo_infer_create       读模型字节、构图、初始化执行器、回填输入形状
//   yolo_infer_init         回填 config 的输出形状 / 元素类型
//   yolo_infer_predict      跑一次推理，填充 outs[]
//   yolo_infer_destroy      释放
//
// 依赖说明：
//   · engine/ 自包含；OpenCL 走运行时加载（engine/include/onnxcl/cl_api.h），
//     不依赖 OpenCL.lib / OpenCL SDK，MinGW / MSVC / GCC 都能编
//   · 不跨 ABI 边界抛异常：引擎错误一律转成 STATUS_* 负数错误码（Java 侧再包成 YoloException）
//   · 输出张量为 float32（引擎内部就是 float32 计算）；非 float32 输出当前直接报不支持，
//     而不是给出一个类型对不上的指针让上层错读
#include <fast_yolo/yolo.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "onnxcl/cl_runtime.h"   // ClRuntime（设备信息日志用）
#include "onnxcl/executor.h"
#include "onnxcl/graph.h"

// 导出方式说明：
//   这四个 ABI 函数【不在定义上加 __declspec(dllexport)】，和 MNN/RKNN/openvino
//   后端保持一致。加了的后果（MSVC）：
//     error C2375: "yolo::yolo_infer_create": 重定义；不同的链接
//   因为 yolo.h 里的声明本身没有 dllexport，定义上凭空加一个就会被判成不同的链接。
//
//   导出靠构建系统：
//     MSVC  : 顶层 CMakeLists 里 WIN32 分支的 CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON
//     MinGW : 没有自动导出机制，在 CMakeLists 里显式加 -Wl,--export-all-symbols

namespace {

// 统一日志（带后端前缀，便于和 onnx/MNN 后端区分）
void logLine(const std::string &msg) {
    std::printf("[yolo_onnxcl] %s\n", msg.c_str());
}

// 引擎的 DType -> ONNX 元素类型码（YoloInferOutput.elem_type 用的是 ONNX 的编码）
int toOnnxElemType(const onnxcl::DType t) {
    switch (t) {
        case onnxcl::DType::F32: return 1;
        case onnxcl::DType::U8: return 2;
        case onnxcl::DType::I32: return 6;
        case onnxcl::DType::I64: return 7;
        case onnxcl::DType::F16: return 10;
        default: return 1;
    }
}

/**
 * 把引擎内部的 float32 缓冲按 ONNX 声明的类型导出。
 *
 * 引擎内部所有张量统一按 f32 存（理由见 engine/src/exec/cpu_ops.cpp 文件头），
 * 但模型的输出类型是 ONNX 里声明的 —— 比如语义分割的类别图就是 uint8。
 * 这里必须真做一次转换：否则上层按 uint8 去读一段 f32 内存，读出来全是垃圾。
 *
 * 返回写出的字节数；不支持的类型返回 0。
 */
size_t exportTensor(const std::vector<float> &src, const int elemType, std::vector<uint8_t> &dst) {
    const size_t n = src.size();
    switch (elemType) {
        case 1:  // float32
            dst.assign(reinterpret_cast<const uint8_t *>(src.data()),
                       reinterpret_cast<const uint8_t *>(src.data()) + n * sizeof(float));
            return n * sizeof(float);
        case 2: {  // uint8（语义分割类别图）；先截断到 [0,255]
            dst.resize(n);
            for (size_t i = 0; i < n; ++i) {
                const float v = src[i];
                dst[i] = static_cast<uint8_t>(v <= 0.0f ? 0
                                                        : (v >= 255.0f
                                                               ? 255
                                                               : static_cast<int>(v)));
            }
            return n;
        }
        case 3: {  // int8
            dst.resize(n);
            for (size_t i = 0; i < n; ++i) {
                const float v = src[i];
                const int iv = v <= -128.0f ? -128 : (v >= 127.0f ? 127 : static_cast<int>(v));
                dst[i] = static_cast<uint8_t>(static_cast<int8_t>(iv));
            }
            return n;
        }
        case 6: {  // int32
            dst.resize(n * sizeof(int32_t));
            int32_t *p = reinterpret_cast<int32_t *>(dst.data());
            for (size_t i = 0; i < n; ++i) p[i] = static_cast<int32_t>(src[i]);
            return n * sizeof(int32_t);
        }
        case 7: {  // int64
            dst.resize(n * sizeof(int64_t));
            int64_t *p = reinterpret_cast<int64_t *>(dst.data());
            for (size_t i = 0; i < n; ++i) p[i] = static_cast<int64_t>(src[i]);
            return n * sizeof(int64_t);
        }
        default:
            return 0;
    }
}

// 判断设备是否可用：本后端只有 OpenCL 一条路
bool deviceAcceptable(const yolo::InferDevice dev) {
    switch (dev) {
        case yolo::InferDevice::OPENCL:
        case yolo::InferDevice::AUTO:
        case yolo::InferDevice::ALL:
        case yolo::InferDevice::CPU:  // 允许：会选到 OpenCL 里第一个设备（含 CPU 型 OpenCL 设备）
            return true;
        default:
            return false;  // CUDA / VULKAN / NPU 不是本后端能提供的
    }
}

}  // namespace

/**
 * 引擎持有者
 *
 * 生命周期：create 里构造并完成初始化，此后 predict 复用；
 * 输出缓冲是成员变量，保证 YoloInferOutput.data 在下次 predict 前一直有效。
 */
class OnnxClYolo {
public:
    std::unique_ptr<onnxcl::Graph> graph;
    std::unique_ptr<onnxcl::Executor> exec;

    // 引擎图里的输入张量下标
    int input_tensor = -1;
    // 对外暴露的输出在 graph.outputs 里的下标（config->output_name 非空时只保留匹配的）
    std::vector<size_t> out_slots;
    // 引擎输出缓冲（成员持有，predict 之间复用，避免每帧分配）
    std::vector<std::vector<float>> out_buf;
    // 非 float32 输出的导出缓冲（每个输出一份，同样跨帧复用）
    std::vector<std::vector<uint8_t>> out_bytes;

    // 输入元素个数（校验 Letterbox 尺寸用）
    size_t input_elems = 0;
    // 构造期记录的错误信息
    std::string init_error;
    // 是否打开了逐算子/逐层性能统计（环境变量 ONNXCL_PROFILE=1 触发，
    // 用来定位「哪一层慢」；关闭时对性能零影响）
    bool profiling = false;

    bool ok() const { return init_error.empty() && exec != nullptr; }
};

// ---------------------------------------------------------------------------
// 后端 ABI
//
// 【为什么要包在 namespace yolo 里】
// yolo.h 是在 `namespace yolo { extern "C" { ... } }` 里声明这四个函数的。
// 如果我们在全局作用域的 extern "C" 里定义它们，MSVC 会报：
//     error C2375: "yolo::yolo_infer_create": 重定义；不同的链接
// （GCC/MinGW 宽容地放过了，所以这是 MSVC 帮我们挑出来的真问题）。
// 把定义放进同一个 namespace 就完全对齐了。
//
// 注意这里是 namespace 而不是 extern "C"：语言链接由 yolo.h 的声明决定，
// 定义沿用即可；而且定义上带 __declspec(dllexport) 是必要的 ——
// MSVC 靠 CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 自动导出，MinGW 没这个机制，
// 必须显式导出，否则 Java 侧 System.loadLibrary 找不到符号。
// ---------------------------------------------------------------------------
namespace yolo {

/**
 * 创建实例。失败返回 nullptr，原因打到日志。
 *
 * 【必须包 try/catch】：本函数经 extern "C" 边界被宿主（JNI / ctypes / 上层 C++）
 * 调用，C++ 异常一旦逸出就会直接 terminate 掉整个宿主进程（Windows 上表现为
 * 0xC0000409，而且栈上的日志因为 stdout 缓冲丢失，什么都看不到）。
 * 这里一律消化成 nullptr + 中文日志。
 */
void *yolo_infer_create(const yolo::YOLOConfig *config) {
    try {
        if (config == nullptr || config->model_data.empty()) {
            logLine("模型字节为空，创建失败");
            return nullptr;
        }
        if (!deviceAcceptable(config->device)) {
            logLine("本后端只支持 OpenCL（device 请设 OPENCL/AUTO/CPU），当前请求的设备不受支持");
            return nullptr;
        }

        auto self = std::make_unique<OnnxClYolo>();

        // 1) 直接解析 ONNX 字节（没有中间格式，也没有任何转换步骤）
        auto model = std::make_unique<onnxcl::ModelProto>();
        std::string err;
        if (!model->loadFromMemory(reinterpret_cast<const uint8_t *>(config->model_data.data()),
                                   config->model_data.size(), &err)) {
            logLine("ONNX 解析失败: " + err);
            return nullptr;
        }

        // 2) 构图（形状推导 + SiLU 融合都在这里面）
        self->graph = std::make_unique<onnxcl::Graph>();
        if (!onnxcl::buildGraph(*model, *self->graph, &err)) {
            logLine("构图失败: " + err);
            return nullptr;
        }
        model.reset();  // 权重已经物化进 Graph，字节可以放了

        // 3) 输入张量：优先匹配 config->input_name，匹配不上就用第一个输入
        const onnxcl::Graph &g = *self->graph;
        if (g.inputs.empty()) {
            logLine("模型没有输入");
            return nullptr;
        }
        self->input_tensor = g.inputs[0];
        if (!config->input_name.empty()) {
            for (const int idx : g.inputs) {
                if (g.at(idx).name == config->input_name) {
                    self->input_tensor = idx;
                    break;
                }
            }
        }
        const onnxcl::TensorInfo &in = g.at(self->input_tensor);
        if (in.dtype != onnxcl::DType::F32) {
            logLine("输入张量不是 float32，本后端暂不支持");
            return nullptr;
        }
        self->input_elems = static_cast<size_t>(in.numElements());

        // 4) 输出筛选：config->output_name 非空时只暴露该输出
        for (size_t i = 0; i < g.outputs.size(); ++i) {
            const onnxcl::TensorInfo &t = g.at(g.outputs[i]);
            if (!config->output_name.empty() && t.name != config->output_name) {
                continue;
            }
            if (t.dtype != onnxcl::DType::F32) {
                // 引擎内部统一 f32，但输出类型按 ONNX 声明导出（见 exportTensor）
                const int et = toOnnxElemType(t.dtype);
                if (et != 2 && et != 3 && et != 6 && et != 7) {
                    logLine("输出 " + t.name + " 的类型暂不支持导出（elem_type=" +
                            std::to_string(et) + "）");
                    return nullptr;
                }
            }
            self->out_slots.push_back(i);
        }
        if (self->out_slots.empty()) {
            logLine("没有可用的输出（output_name=" + config->output_name + " 没有匹配项？）");
            return nullptr;
        }

        // 5) 初始化执行器（编译内核 + 上传权重）。这一步最耗时（首次要编译 OpenCL 内核）
        self->exec = std::make_unique<onnxcl::Executor>(g);
        // 逐层性能统计：ONNXCL_PROFILE=1 打开（只在排查性能时用）
        {
            const char* prof = std::getenv("ONNXCL_PROFILE");
            self->profiling = (prof != nullptr && prof[0] == '1');
            self->exec->setProfiling(self->profiling);
        }
        // gpu_id 当作 OpenCL 设备序号用
        if (!self->exec->init(&err, config->gpu_id)) {
            logLine("执行器初始化失败: " + err);
            return nullptr;
        }
        self->out_buf.resize(g.outputs.size());

        const onnxcl::ClRuntime &rt = onnxcl::ClRuntime::instance();
        logLine("就绪：设备=" + rt.deviceName() + "，输入=" + in.name + " " +
                onnxcl::shapeToString(in.shape) + "，显存=" +
                std::to_string(static_cast<long>(self->exec->deviceBytes() / 1048576)) + " MB");

        return self.release();
    } catch (const std::exception &e) {
        logLine(std::string("create 抛出异常（已拦截）: ") + e.what());
        return nullptr;
    } catch (...) {
        logLine("create 抛出未知异常（已拦截）");
        return nullptr;
    }
}

/**
 * 回填输入/输出形状与类型到 config（核心层据此建 Letterbox、做后处理分发）。
 */
void yolo_infer_init(void *inference, yolo::YOLOConfig *config, int *code) {
    try {
        if (inference == nullptr || config == nullptr) {
            *code = STATUS_NOT_INITIALIZED;
            return;
        }
        auto *self = static_cast<OnnxClYolo *>(inference);
        if (!self->ok()) {
            *code = STATUS_MODEL_INIT_FAILED;
            return;
        }

        const onnxcl::Graph &g = *self->graph;
        config->input_shape = g.at(self->input_tensor).shape;

        config->output_shapes.clear();
        config->output_dtypes.clear();
        for (const size_t slot : self->out_slots) {
            const onnxcl::TensorInfo &t = g.at(g.outputs[slot]);
            config->output_shapes.push_back(t.shape);
            config->output_dtypes.push_back(toOnnxElemType(t.dtype));
        }
        if (config->output_shapes.empty()) {
            *code = STATUS_MODEL_INIT_FAILED;
            return;
        }
        config->output_shape = config->output_shapes.front();

        *code = STATUS_SUCCESS;
    } catch (const std::exception &e) {
        logLine(std::string("init 抛出异常（已拦截）: ") + e.what());
        *code = STATUS_MODEL_INIT_FAILED;
    } catch (...) {
        logLine("init 抛出未知异常（已拦截）");
        *code = STATUS_MODEL_INIT_FAILED;
    }
}

/**
 * 跑一次推理。outs 由核心层传入，本函数填充其中前 n 个（n <= max_outs）。
 * 返回实际输出数；失败返回 0 并写 *code。
 */
int yolo_infer_predict(void *inference, const yolo::Letterbox *letterbox,
                                      yolo::YoloInferOutput *outs, const int max_outs, int *code) {
    try {
        if (inference == nullptr || letterbox == nullptr || outs == nullptr) {
            *code = STATUS_NOT_INITIALIZED;
            return 0;
        }
        auto *self = static_cast<OnnxClYolo *>(inference);
        if (!self->ok()) {
            *code = STATUS_MODEL_INIT_FAILED;
            return 0;
        }

        // Letterbox 的像素数是 box_size*box_size*3，必须和模型输入元素数一致
        // （核心层在 init 之后按 config->input_shape 建 Letterbox，正常不会不一致）
        if (letterbox->pixels.size() != self->input_elems) {
            logLine("输入尺寸不匹配：Letterbox 有 " + std::to_string(letterbox->pixels.size()) +
                    " 个元素，模型要 " + std::to_string(self->input_elems));
            *code = STATUS_INFERENCE_FAILED;
            return 0;
        }

        std::string err;
        if (!self->exec->run(letterbox->pixels.data(), self->out_buf, &err)) {
            logLine("推理失败: " + err);
            *code = STATUS_INFERENCE_FAILED;
            return 0;
        }

        const onnxcl::Graph &g = *self->graph;
        const int n = std::min(static_cast<int>(self->out_slots.size()), max_outs);
        self->out_bytes.resize(n);
        for (int i = 0; i < n; ++i) {
            const size_t slot = self->out_slots[i];
            const onnxcl::TensorInfo &t = g.at(g.outputs[slot]);
            const std::vector<float> &buf = self->out_buf[slot];
            const int et = toOnnxElemType(t.dtype);

            yolo::YoloInferOutput &dst = outs[i];
            if (et == 1) {
                // float32：直接指向引擎缓冲，不做多余的拷贝
                dst.data = buf.data();
            } else {
                // 其它类型：转成 ONNX 声明的类型再给上层
                if (exportTensor(buf, et, self->out_bytes[i]) == 0) {
                    logLine("输出 " + t.name + " 的类型无法导出");
                    *code = STATUS_INFERENCE_FAILED;
                    return 0;
                }
                dst.data = self->out_bytes[i].data();
            }
            dst.size = static_cast<int64_t>(buf.size());
            dst.elem_type = et;
            dst.rank = static_cast<int>(std::min<size_t>(t.shape.size(), 8));
            for (int d = 0; d < 8; ++d) {
                dst.shape[d] = (d < dst.rank) ? t.shape[d] : 0;
            }
        }
        *code = STATUS_SUCCESS;
        return n;
    } catch (const std::exception &e) {
        // 推理是每帧都走的热路径，异常绝不允评到宿主
        logLine(std::string("predict 抛出异常（已拦截）: ") + e.what());
        *code = STATUS_INFERENCE_FAILED;
        return 0;
    } catch (...) {
        logLine("predict 抛出未知异常（已拦截）");
        *code = STATUS_INFERENCE_FAILED;
        return 0;
    }
}

/**
 * 释放。引擎的显存统一由 Executor 析构回收。
 */
void yolo_infer_destroy(void *inference) {
    if (inference == nullptr) {
        return;
    }
    try {
        auto* self = static_cast<OnnxClYolo*>(inference);
        // 打开了性能统计就在销毁前打一份报告（定位慢层用）
        if (self->profiling && self->exec != nullptr) {
            self->exec->printProfile(stderr);
            self->exec->printSlowNodes(stderr, 80);
        }
        delete self;
    } catch (const std::exception &e) {
        logLine(std::string("destroy 抛出异常（已拦截）: ") + e.what());
    } catch (...) {
        logLine("destroy 抛出未知异常（已拦截）");
    }
}

}  // namespace yolo

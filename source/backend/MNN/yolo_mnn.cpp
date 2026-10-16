#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <fast_yolo/yolo.h>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNForwardType.h>

#if defined(_WIN32) && defined(LIBFASTYOLO_EXPORTS)
#define YOLO_MNN_API __declspec(dllexport)
#elif defined(_WIN32)
#define YOLO_MNN_API __declspec(dllimport)
#else
#define YOLO_MNN_API
#endif

using namespace MNN::Express;

// 将 yolo::InferDevice 枚举映射为 MNN 后端类型
static MNNForwardType MapInferDevice(yolo::InferDevice device) {
    switch (device) {
        case yolo::InferDevice::CPU:   return MNN_FORWARD_CPU;
        case yolo::InferDevice::OPENCL:return MNN_FORWARD_OPENCL;
        case yolo::InferDevice::VULKAN:return MNN_FORWARD_VULKAN;
        case yolo::InferDevice::CUDA:  return MNN_FORWARD_CUDA;
        case yolo::InferDevice::AUTO:  return MNN_FORWARD_AUTO;
        default:                       return MNN_FORWARD_CPU;
    }
}

// halide 类型 → ONNX 元素类型码（1=float32, 2=uint8, 6=int32, 7=int64）
static int MapElemType(const halide_type_t &type) {
    if (type.code == halide_type_float) {
        return 1;
    }
    if (type.code == halide_type_uint) {
        return type.bits == 8 ? 2 : 6;
    }
    if (type.code == halide_type_int) {
        return type.bits == 64 ? 7 : 6;
    }
    return 1;
}

class MNNYolo {
public:
    std::shared_ptr<Executor::RuntimeManager> runtime_manager_; // MNN 运行时管理器
    std::shared_ptr<Module> net_;                                // 加载的 MNN 模型
    VARP input_var_;                                             // 模型输入张量
    std::vector<int64_t> input_shape_;                           // 输入维度

    // 单个输出描述
    struct OutputInfo {
        std::vector<int64_t> shape;
        int elem_type = 1;          // ONNX 元素类型码
        size_t elem_size = 4;       // 单个元素字节数（readMap 拷贝用）
        std::vector<uint8_t> buffer; // 输出缓冲（内部拷贝，稳定有效）
    };
    std::vector<OutputInfo> outputs_;

    MNNYolo(const std::vector<char> &model_data, yolo::InferDevice device, int num_threads) {
        // 配置推理后端（CPU / OpenCL / Vulkan / CUDA）
        MNN::ScheduleConfig config;
        config.type = MapInferDevice(device);
        config.numThread = std::max(num_threads, 1);

        // 创建运行时管理器
        runtime_manager_.reset(Executor::RuntimeManager::createRuntimeManager(config));
        if (!runtime_manager_) {
            throw std::runtime_error("Failed to create MNN RuntimeManager");
        }

        // 非 CPU 后端启用缓存加速
        if (config.type != MNN_FORWARD_CPU) {
            runtime_manager_->setCache("./cachefile");
        }

        // 从内存中加载 MNN 模型
        net_.reset(Module::load(
            {}, {},
            reinterpret_cast<const uint8_t *>(model_data.data()),
            model_data.size(),
            runtime_manager_
        ));
        if (!net_) {
            throw std::runtime_error("Failed to load MNN model");
        }

        // 从模型元数据动态读取输入 shape
        auto net_info = net_->getInfo();
        std::vector<int> mnn_input_shape;
        if (net_info && !net_info->inputs.empty()) {
            const auto &dims = net_info->inputs[0].dim;
            // MNN 中动态维度标记为 <= 0，替换为实际值
            for (auto d : dims) {
                if (d <= 0) {
                    mnn_input_shape.push_back(1); // batch 等动态维取 1
                } else {
                    mnn_input_shape.push_back(d);
                }
            }
        }
        if (mnn_input_shape.empty()) {
            mnn_input_shape = {1, 3, 640, 640}; // fallback
        }
        for (auto d : mnn_input_shape) {
            input_shape_.push_back(d);
        }
        std::cout << "Model input shape: [";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << input_shape_[i];
        }
        std::cout << "]" << std::endl;
        input_var_ = _Input(mnn_input_shape, NCHW, halide_type_of<float>());

        // 用 dummy 数据前向一次，获取全部输出 shape 与类型
        auto dummy_input = _Input(mnn_input_shape, NCHW, halide_type_of<float>());
        std::vector dummy_inputs = {dummy_input};
        auto dummy_outputs = net_->onForward(dummy_inputs);
        if (dummy_outputs.empty()) {
            throw std::runtime_error("MNN warmup forward failed");
        }

        for (size_t i = 0; i < dummy_outputs.size(); ++i) {
            auto output_var = dummy_outputs[i];
            // 确保输出为 NCHW 格式
            if (output_var->getInfo()->order != NCHW) {
                output_var = _Convert(output_var, NCHW);
            }
            auto info = output_var->getInfo();
            if (!info || info->dim.empty()) {
                continue;
            }
            OutputInfo out;
            for (const auto d : info->dim) {
                out.shape.push_back(d);
            }
            out.elem_type = MapElemType(info->type);
            out.elem_size = info->type.bytes();
            int64_t total = 1;
            for (const auto d : out.shape) {
                total *= d;
            }
            out.buffer.resize(static_cast<size_t>(total) * out.elem_size);
            std::cout << "Model out[" << i << "] shape: [";
            for (size_t j = 0; j < out.shape.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << out.shape[j];
            }
            std::cout << "] elem_type=" << out.elem_type << std::endl;
            outputs_.push_back(std::move(out));
        }
        if (outputs_.empty()) {
            throw std::runtime_error("MNN model has no usable output");
        }
    }

    ~MNNYolo() {
        // 析构时将运行时缓存写回磁盘
        if (runtime_manager_) {
            runtime_manager_->updateCache();
        }
    }

    int infer(const std::vector<float> &img_data, yolo::YoloInferOutput *outs, int max_outs, int *code) {
        // 将输入数据拷贝到 MNN 张量
        auto input_ptr = input_var_->writeMap<float>();
        ::memcpy(input_ptr, img_data.data(), img_data.size() * sizeof(float));

        // 执行前向推理
        std::vector<VARP> inputs = {input_var_};
        auto outputs = net_->onForward(inputs);
        if (outputs.empty()) {
            *code = STATUS_INFERENCE_FAILED;
            return 0;
        }

        const int n = std::min(static_cast<int>(outputs_.size()), std::min(max_outs, static_cast<int>(outputs.size())));
        for (int i = 0; i < n; ++i) {
            auto output_var = outputs[i];
            // 转为 NCHW 后读取结果
            if (output_var->getInfo()->order != NCHW) {
                output_var = _Convert(output_var, NCHW);
            }
            auto &out = outputs_[i];
            // 按类型读取并拷贝到内部缓冲（readMap 指针有效期有限）
            const void *src = nullptr;
            switch (out.elem_type) {
                case 7: src = output_var->readMap<int64_t>(); break;
                case 6: src = output_var->readMap<int32_t>(); break;
                case 2: src = output_var->readMap<uint8_t>(); break;
                default: src = output_var->readMap<float>(); break;
            }
            if (!src) {
                *code = STATUS_INFERENCE_FAILED;
                return 0;
            }
            const int64_t total = static_cast<int64_t>(out.buffer.size()) / out.elem_size;
            ::memcpy(out.buffer.data(), src, static_cast<size_t>(total) * out.elem_size);

            yolo::YoloInferOutput &dst = outs[i];
            dst.data = out.buffer.data();
            dst.size = total;
            dst.elem_type = out.elem_type;
            dst.rank = static_cast<int>(std::min(out.shape.size(), static_cast<size_t>(8)));
            for (int d = 0; d < dst.rank; ++d) {
                dst.shape[d] = out.shape[d];
            }
        }
        *code = STATUS_SUCCESS;
        return n;
    }
};

extern "C" {

// 创建推理引擎实例，返回不透明指针
void *yolo_infer_create(const yolo::YOLOConfig *config) {
    if (config->model_data.empty()) {
        return nullptr;
    }
    try {
        auto mnn = new MNNYolo(config->model_data, config->device, config->num_threads);
        if (mnn->input_shape_.empty() || mnn->outputs_.empty()) {
            delete mnn;
            return nullptr;
        }
        return mnn;
    } catch (...) {
        return nullptr;
    }
}

// 初始化：将获取到的输入/输出 shape 回填到 config
void yolo_infer_init(void *inference, yolo::YOLOConfig *config, int *code) {
    if (inference == nullptr) {
        *code = STATUS_NOT_INITIALIZED;
        return;
    }
    const auto mnn = static_cast<MNNYolo *>(inference);
    if (mnn->input_shape_.empty() || mnn->outputs_.empty()) {
        *code = STATUS_MODEL_INIT_FAILED;
        return;
    }
    config->input_shape = mnn->input_shape_;
    config->output_shapes.clear();
    config->output_dtypes.clear();
    for (const auto &out : mnn->outputs_) {
        config->output_shapes.push_back(out.shape);
        config->output_dtypes.push_back(out.elem_type);
    }
    config->output_shape = config->output_shapes.front();
    *code = STATUS_SUCCESS;
}

// 执行推理，结果指针写回 res_data
int yolo_infer_predict(void *inference,
                       const yolo::Letterbox *letterbox, yolo::YoloInferOutput *outs, int max_outs, int *code) {
    if (inference == nullptr) {
        *code = STATUS_NOT_INITIALIZED;
        return 0;
    }
    const auto mnn = static_cast<MNNYolo *>(inference);
    return mnn->infer(letterbox->pixels, outs, max_outs, code);
}

// 销毁推理引擎
void yolo_infer_destroy(void *inference) {
    if (inference == nullptr) {
        return;
    }
    const auto mnn = static_cast<MNNYolo *>(inference);
    delete mnn;
}

}

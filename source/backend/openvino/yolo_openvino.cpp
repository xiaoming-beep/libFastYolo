#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <fast_yolo/yolo.h>
#include <openvino/openvino.hpp>

#if defined(_WIN32) && defined(LIBFASTYOLO_EXPORTS)
#define YOLO_OPENVINO_API __declspec(dllexport)
#elif defined(_WIN32)
#define YOLO_OPENVINO_API __declspec(dllimport)
#else
#define YOLO_OPENVINO_API
#endif

// 将 yolo::InferDevice 枚举映射为 OpenVINO 设备串
// OpenVINO GPU 插件基于 OpenCL，故 OPENCL 映射为 "GPU"（仅 Intel 核显/Arc 可用）
// VULKAN / CUDA 无对应插件，回退 CPU
static std::string MapInferDevice(yolo::InferDevice device) {
    switch (device) {
        case yolo::InferDevice::CPU:    return "CPU";
        case yolo::InferDevice::NPU:    return "NPU";
        case yolo::InferDevice::OPENCL: return "GPU";
        case yolo::InferDevice::AUTO:   return "AUTO";
        case yolo::InferDevice::VULKAN:
        case yolo::InferDevice::CUDA:
        default:
            std::cerr << "[yolo_openvino] device not supported by OpenVINO, fallback to CPU" << std::endl;
            return "CPU";
    }
}

class OVYolo {
public:
    ov::Core core_;                                    // OpenVINO 运行时
    ov::CompiledModel compiled_;                       // 编译后的模型
    ov::InferRequest infer_request_;                   // 推理请求（持有输出张量）
    std::vector<int64_t> input_shape_;                 // 输入维度

    // 保留的模型输出（与 compiled_.outputs() 对应）
    struct OutputInfo {
        ov::Output<const ov::Node> node;   // 输出描述（取 shape / element type / tensor）
        std::vector<int64_t> shape;
        int elem_type = 1;                 // ONNX 元素类型码（1=float32, 7=int64）
    };
    std::vector<OutputInfo> outputs_;

    OVYolo(const std::vector<char> &model_data, const yolo::YOLOConfig *config) {
        // 从内存加载模型（支持 ONNX 等单文件格式；IR 的 xml+bin 需两个文件，不在本后端契约内）
        auto model = core_.read_model(
            std::string(model_data.begin(), model_data.end()),
            ov::Tensor{}
        );
        if (!model) {
            throw std::runtime_error("Failed to read model from memory");
        }

        // 编译模型到目标设备
        const std::string device = MapInferDevice(config->device);
        ov::AnyMap props;
        // INFERENCE_NUM_THREADS 仅 CPU 插件支持，GPU/NPU 设置会报错
        if (device == "CPU") {
            props[ov::inference_num_threads.name()] = std::max(config->num_threads, 1);
        }
        compiled_ = core_.compile_model(model, device, props);

        // 读取输入 shape（动态维度回退 640）
        ov::Shape in_shape = compiled_.input().get_shape();
        for (auto d : in_shape) {
            input_shape_.push_back(d > 0 ? static_cast<int64_t>(d) : 640);
        }
        if (input_shape_.empty()) {
            input_shape_ = {1, 3, 640, 640}; // fallback
        }

        // 读取全部输出（FLOAT32 / INT64 / UINT8；配置指定 output_name 时只保留该输出）
        for (auto &out : compiled_.outputs()) {
            const ov::element::Type type = out.get_element_type();
            if (type != ov::element::f32 && type != ov::element::i64 && type != ov::element::u8) {
                continue;
            }
            if (!config->output_name.empty() && out.get_any_name() != config->output_name) {
                continue;
            }
            OutputInfo info;
            info.node = out;
            ov::Shape out_shape = out.get_shape();
            for (auto d : out_shape) {
                info.shape.push_back(d > 0 ? static_cast<int64_t>(d) : 0);
            }
            if (type == ov::element::i64) {
                info.elem_type = 7;
            } else if (type == ov::element::u8) {
                info.elem_type = 2;
            } else {
                info.elem_type = 1;
            }
            outputs_.push_back(std::move(info));
        }
        if (outputs_.empty()) {
            throw std::runtime_error("no usable (f32/i64/u8) model output");
        }

        infer_request_ = compiled_.create_infer_request();
        std::cout << "Model input shape: [";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << input_shape_[i];
        }
        std::cout << "]" << std::endl;
        for (size_t i = 0; i < outputs_.size(); ++i) {
            std::cout << "Model out[" << i << "] (" << outputs_[i].node.get_any_name() << ") shape: [";
            for (size_t j = 0; j < outputs_[i].shape.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << outputs_[i].shape[j];
            }
            std::cout << "]" << std::endl;
        }
    }

    ~OVYolo() = default;

    int infer(const std::vector<float> &img_data, yolo::YoloInferOutput *outs, int max_outs, int *code) {
        const size_t expect = static_cast<size_t>(yolo::mul_size(input_shape_));
        if (img_data.size() != expect) {
            std::cerr << "[yolo_openvino] input size mismatch: " << img_data.size()
                      << " != " << expect << std::endl;
            *code = STATUS_INFERENCE_FAILED;
            return 0;
        }

        // 零拷贝：直接包装 letterbox 像素缓冲（调用期内有效）
        ov::Tensor input_tensor(ov::element::f32, ov::Shape(input_shape_.begin(), input_shape_.end()),
                                const_cast<float *>(img_data.data()));
        infer_request_.set_input_tensor(input_tensor);

        // 推理
        infer_request_.infer();

        // 零拷贝返回：输出张量由 infer_request_ 持有，在下一次 infer() 之前始终有效。
        // 契约：调用方（核心层后处理）必须在本次 predict 返回前消费完数据。
        const int n = std::min(static_cast<int>(outputs_.size()), max_outs);
        for (int i = 0; i < n; ++i) {
            auto out_tensor = infer_request_.get_tensor(outputs_[i].node);
            yolo::YoloInferOutput &dst = outs[i];
            dst.data = out_tensor.data();   // void*（OpenVINO >= 2024）
            dst.size = static_cast<int64_t>(out_tensor.get_byte_size()) /
                       static_cast<int64_t>(out_tensor.get_element_type().size());
            dst.elem_type = outputs_[i].elem_type;
            dst.rank = static_cast<int>(std::min(outputs_[i].shape.size(), static_cast<size_t>(8)));
            for (int d = 0; d < dst.rank; ++d) {
                dst.shape[d] = outputs_[i].shape[d];
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
        auto ov_yolo = new OVYolo(config->model_data, config);
        if (ov_yolo->input_shape_.empty() || ov_yolo->outputs_.empty()) {
            delete ov_yolo;
            return nullptr;
        }
        return ov_yolo;
    } catch (const std::exception &e) {
        std::cerr << "[yolo_openvino] init failed: " << e.what() << std::endl;
        return nullptr;
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
    const auto ov_yolo = static_cast<OVYolo *>(inference);
    if (ov_yolo->input_shape_.empty() || ov_yolo->outputs_.empty()) {
        *code = STATUS_MODEL_INIT_FAILED;
        return;
    }
    config->input_shape = ov_yolo->input_shape_;
    config->output_shapes.clear();
    config->output_dtypes.clear();
    for (const auto &out : ov_yolo->outputs_) {
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
    const auto ov_yolo = static_cast<OVYolo *>(inference);
    return ov_yolo->infer(letterbox->pixels, outs, max_outs, code);
}

// 销毁推理引擎
void yolo_infer_destroy(void *inference) {
    if (inference == nullptr) {
        return;
    }
    const auto ov_yolo = static_cast<OVYolo *>(inference);
    delete ov_yolo;
}

}

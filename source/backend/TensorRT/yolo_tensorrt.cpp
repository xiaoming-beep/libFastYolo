#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <fast_yolo/yolo.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#if defined(_WIN32) && defined(LIBFASTYOLO_EXPORTS)
#define YOLO_TRT_API __declspec(dllexport)
#elif defined(_WIN32)
#define YOLO_TRT_API __declspec(dllimport)
#else
#define YOLO_TRT_API
#endif

class SimpleLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
    }
};

class TensorRTYolo {
public:
    SimpleLogger logger_;
    nvinfer1::IRuntime *runtime_;
    nvinfer1::ICudaEngine *engine_;
    nvinfer1::IExecutionContext *context_;
    std::vector<int64_t> input_shape_;
    std::string input_name_;

    // 单个输出描述
    struct OutputInfo {
        std::string name;
        std::vector<int64_t> shape;
        int64_t size = 0;
        void *d_buf = nullptr;   // device 缓冲
        std::vector<float> host; // host 缓冲（float32 拷贝）
    };
    std::vector<OutputInfo> outputs_;

    explicit TensorRTYolo(const std::vector<char> &model_data, const yolo::YOLOConfig *config) {
        runtime_ = nvinfer1::createInferRuntime(logger_);
        engine_ = runtime_->deserializeCudaEngine(model_data.data(), model_data.size());
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize TensorRT engine");
        }
        context_ = engine_->createExecutionContext();

        // 枚举全部 I/O tensor：输入取第一个（或 config->input_name），输出取全部 FLOAT 绑定
        const int nb = engine_->getNbIOTensors();
        for (int i = 0; i < nb; ++i) {
            const char *name = engine_->getIOTensorName(i);
            const auto mode = engine_->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                if (input_name_.empty() ||
                    (config->input_name.empty() && name == std::string("images")) ||
                    name == config->input_name) {
                    input_name_ = name;
                    auto dims = engine_->getTensorShape(name);
                    for (int d = 0; d < dims.nbDims; ++d) {
                        input_shape_.push_back(dims.d[d]);
                    }
                }
                continue;
            }
            // 输出：仅支持 FLOAT32（半精度引擎输出需 FP16→FP32 转换，暂不支持）
            if (engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
                std::cerr << "[yolo_tensorrt] skip non-float output: " << name << std::endl;
                continue;
            }
            if (!config->output_name.empty() && name != config->output_name) {
                continue;
            }
            OutputInfo out;
            out.name = name;
            auto dims = engine_->getTensorShape(name);
            out.size = 1;
            for (int d = 0; d < dims.nbDims; ++d) {
                const int64_t dim = dims.d[d];
                out.shape.push_back(dim);
                out.size *= dim;
            }
            outputs_.push_back(std::move(out));
        }
        if (input_name_.empty() || outputs_.empty()) {
            throw std::runtime_error("engine has no usable input/output binding");
        }

        input_byte_size_ = static_cast<size_t>(yolo::mul_size(input_shape_)) * sizeof(float);
        cudaMalloc(&d_input_, input_byte_size_);
        // TensorRT 10 通过 setTensorAddress 为每个 I/O tensor 绑定设备地址
        context_->setTensorAddress(input_name_.c_str(), d_input_);
        for (auto &out : outputs_) {
            out.host.resize(static_cast<size_t>(out.size));
            cudaMalloc(&out.d_buf, static_cast<size_t>(out.size) * sizeof(float));
            context_->setTensorAddress(out.name.c_str(), out.d_buf);
        }
    }

    ~TensorRTYolo() {
        if (d_input_) cudaFree(d_input_);
        for (auto &out : outputs_) {
            if (out.d_buf) cudaFree(out.d_buf);
        }
        delete context_;
        delete engine_;
        delete runtime_;
    }

    int infer(const std::vector<float> &img_data, yolo::YoloInferOutput *outs, int max_outs, int *code) {
        cudaMemcpy(d_input_, img_data.data(), input_byte_size_, cudaMemcpyHostToDevice);

        context_->enqueueV3(cudaStreamLegacy);
        cudaDeviceSynchronize();

        for (size_t i = 0; i < outputs_.size() && static_cast<int>(i) < max_outs; ++i) {
            auto &out = outputs_[i];
            cudaMemcpy(out.host.data(), out.d_buf, static_cast<size_t>(out.size) * sizeof(float),
                       cudaMemcpyDeviceToHost);
            yolo::YoloInferOutput &dst = outs[i];
            dst.data = out.host.data();
            dst.size = out.size;
            dst.elem_type = 1;
            dst.rank = static_cast<int>(std::min(out.shape.size(), static_cast<size_t>(8)));
            for (int d = 0; d < dst.rank; ++d) {
                dst.shape[d] = out.shape[d];
            }
        }

        *code = STATUS_SUCCESS;
        return static_cast<int>(outputs_.size());
    }

private:
    void *d_input_ = nullptr;
    size_t input_byte_size_ = 0;
};

extern "C" {

void *yolo_infer_create(const yolo::YOLOConfig *config) {
    if (config->model_data.empty()) {
        return nullptr;
    }
    try {
        auto trt = new TensorRTYolo(config->model_data, config);
        if (trt->input_shape_.empty() || trt->outputs_.empty()) {
            delete trt;
            return nullptr;
        }
        return trt;
    } catch (...) {
        return nullptr;
    }
}

void yolo_infer_init(void *inference, yolo::YOLOConfig *config, int *code) {
    if (inference == nullptr) {
        *code = STATUS_NOT_INITIALIZED;
        return;
    }
    const auto trt = static_cast<TensorRTYolo *>(inference);
    if (trt->input_shape_.empty() || trt->outputs_.empty()) {
        *code = STATUS_MODEL_INIT_FAILED;
        return;
    }
    config->input_shape = trt->input_shape_;
    config->output_shapes.clear();
    config->output_dtypes.clear();
    for (const auto &out : trt->outputs_) {
        config->output_shapes.push_back(out.shape);
        config->output_dtypes.push_back(1); // TRT 后端仅支持 FLOAT32 输出
    }
    config->output_shape = config->output_shapes.front();
    *code = STATUS_SUCCESS;
}

int yolo_infer_predict(void *inference,
                       const yolo::Letterbox *letterbox, yolo::YoloInferOutput *outs, int max_outs, int *code) {
    if (inference == nullptr) {
        *code = STATUS_NOT_INITIALIZED;
        return 0;
    }
    const auto trt = static_cast<TensorRTYolo *>(inference);
    return trt->infer(letterbox->pixels, outs, max_outs, code);
}

void yolo_infer_destroy(void *inference) {
    if (inference == nullptr) {
        return;
    }
    const auto trt = static_cast<TensorRTYolo *>(inference);
    delete trt;
}

}

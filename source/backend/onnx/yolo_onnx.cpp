#include <iostream>
#include <memory>

#include <fast_yolo/img_util.h>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <utility>
#include <fast_yolo/yolo.h>
#if defined(_WIN32) && defined(LIBFASTYOLO_EXPORTS)
#define YOLO_ONNX_API __declspec(dllexport)
#elif defined(_WIN32)
#define YOLO_ONNX_API __declspec(dllimport)
#else
#define YOLO_ONNX_API
#endif


class ONNXYolo {
public:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::vector<int64_t> input_shape_;

    // 单个输出描述
    struct OutputInfo {
        std::string name;
        std::vector<int64_t> shape;
        int elem_type = 1;          // ONNX 元素类型码
        void *data_ptr = nullptr;   // 指向 output_tensors_ 中对应张量
    };
    // 保留的模型输出（顺序与 output_names_ 一致）
    std::vector<OutputInfo> outputs_;
    // Run 请求的输出名（const char* 数组，供 session_->Run 使用）
    std::vector<const char *> output_names_cstr_;
    // 输出张量生命周期（下一次 infer 前有效）
    std::vector<Ort::Value> output_tensors_;

    ONNXYolo(const std::vector<char> &model, const yolo::YOLOConfig *config, const std::string &input_name,
             const std::string &output_name_filter)
        : env_(ORT_LOGGING_LEVEL_WARNING, "ONNXYolo"),
          session_([&]() -> std::unique_ptr<Ort::Session> {
              // 会话创建的所有异常必须在这里消化掉：本构造函数经 yolo_infer_create
              // 的 extern "C" 边界调用，异常逃逸会让宿主进程（JNI/ctypes）直接 terminate。
              // CUDA EP 注册或会话创建失败（无设备/显存不足/cuDNN 缺失）时回退纯 CPU。
              auto make_options = [&](const bool cuda_ep) {
                  Ort::SessionOptions options;
                  options.SetIntraOpNumThreads(std::max(config->num_threads, 1));
                  options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
                  if (cuda_ep) {
                      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(
                          options, config->gpu_id));
                  }
                  return options;
              };
#ifdef HAVE_ORT_CUDA_EP
              // AUTO：优先 GPU，失败回退 CPU（容忍异构部署）。
              // CUDA：显式指定即硬性要求，任何 GPU 环节失败都直接初始化失败，绝不静默回退 CPU
              //（否则用户以为在用 GPU，实际在烧 CPU，只有性能能看出来——正是之前踩的坑）。
              if (config->device == yolo::InferDevice::CUDA ||
                  config->device == yolo::InferDevice::AUTO) {
                  const bool strict = config->device == yolo::InferDevice::CUDA;
                  try {
                      auto options = make_options(true);
                      auto session = std::make_unique<Ort::Session>(
                          env_, model.data(), model.size(), options);
                      std::cout << "[yolo_onnx] CUDA EP enabled, gpu_id=" << config->gpu_id << std::endl;
                      return session;
                  } catch (const Ort::Exception &e) {
                      if (strict) {
                          std::cout << "[yolo_onnx] CUDA EP required but init failed (no CPU fallback): "
                                    << e.what() << std::endl;
                          return nullptr; // 上层收到 STATUS_MODEL_INIT_FAILED
                      }
                      std::cout << "[yolo_onnx] CUDA EP unavailable, fallback to CPU: "
                                << e.what() << std::endl;
                  }
              }
#else
              // 本构建产物未含 CUDA EP（CPU 静态库或 -DONNX_GPU=OFF），
              // 显式要求 CUDA 同样视为失败，不回退 CPU
              if (config->device == yolo::InferDevice::CUDA) {
                  std::cout << "[yolo_onnx] CUDA EP required but this build has no CUDA support "
                               "(rebuild with -DONNX_GPU=ON + lib/linux-gpu): abort init" << std::endl;
                  return nullptr;
              }
#endif
              try {
                  auto options = make_options(false);
                  return std::make_unique<Ort::Session>(env_, model.data(), model.size(), options);
              } catch (const Ort::Exception &e) {
                  std::cout << "[yolo_onnx] session create failed: " << e.what() << std::endl;
                  return nullptr;
              }
          }()), input_name_(input_name) {
        if (!session_) {
            return; // 置空即失败，yolo_infer_create 据 input_shape_ 为空返回 nullptr
        }
        //读取模型输入信息（第一个 FLOAT 输入）
        const size_t num_input_nodes = session_->GetInputCount();
        for (size_t i = 0; i < num_input_nodes; ++i) {
            // 输入类型与形状
            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            if (const ONNXTensorElementDataType dtype = tensor_info.GetElementType();
                dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                input_shape_ = tensor_info.GetShape();
                break;
            }
        }
        //输出信息：保留 FLOAT32 / INT64 / UINT8 输出（seg 双输出、sem 类别图等任务需要）
        const size_t num_output_nodes = session_->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; ++i) {
            auto type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            const ONNXTensorElementDataType dtype = tensor_info.GetElementType();
            if (dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
                dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
                continue;
            }
            auto name = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            // 配置指定了 output_name 时只保留该输出
            if (!output_name_filter.empty() && name.get() != output_name_filter) {
                continue;
            }
            OutputInfo info;
            info.name = name.get();
            info.shape = tensor_info.GetShape();
            info.elem_type = static_cast<int>(dtype);
            outputs_.push_back(std::move(info));
        }
        if (!outputs_.empty()) {
            output_names_cstr_.reserve(outputs_.size());
            for (const auto &out : outputs_) {
                output_names_cstr_.push_back(out.name.c_str());
            }
        }
    }

    /**
     * 推理：填充 outs[max_outs]，返回输出数
     */
    int infer(const std::vector<float>& img_data, yolo::YoloInferOutput *outs, int max_outs, int *code) {
        const Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault
        );
        const Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(img_data.data()), yolo::mul_size(input_shape_), input_shape_.data(),
            input_shape_.size()
        );
        if (!input_tensor.IsTensor()) {
            *code = STATUS_INFERENCE_FAILED;
            return 0;
        }
        if (outputs_.empty()) {
            *code = STATUS_INFERENCE_FAILED;
            return 0;
        }
        const auto input_names = input_name_.data();
        //推理（请求全部保留的输出）
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            &input_names, &input_tensor, 1,
            output_names_cstr_.data(), output_names_cstr_.size()
        );
        if (output_tensors.size() != outputs_.size()) {
            *code = STATUS_INFERENCE_FAILED;
            return 0;
        }
        output_tensors_ = std::move(output_tensors);

        const int n = std::min(static_cast<int>(outputs_.size()), max_outs);
        for (int i = 0; i < n; ++i) {
            auto &out = outputs_[i];
            Ort::Value &tensor = output_tensors_[i];
            if (!tensor.IsTensor()) {
                *code = STATUS_INFERENCE_FAILED;
                return 0;
            }
            // 按元素类型取数据指针（张量由 output_tensors_ 持有，下一次 infer 前有效）
            if (out.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                out.data_ptr = tensor.GetTensorMutableData<int64_t>();
            } else if (out.elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
                out.data_ptr = tensor.GetTensorMutableData<uint8_t>();
            } else {
                out.data_ptr = tensor.GetTensorMutableData<float>();
            }
            yolo::YoloInferOutput &dst = outs[i];
            dst.data = out.data_ptr;
            dst.size = out.shape.empty() ? 0 : yolo::mul_size(out.shape);
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
//初始化
void *yolo_infer_create(const yolo::YOLOConfig *config) {
    if (config->model_data.empty()) {
        return nullptr;
    }
    const auto yolo = new ONNXYolo(config->model_data, config, config->input_name, config->output_name);
    if (yolo->input_shape_.empty() || yolo->outputs_.empty()) {
        delete yolo;
        return nullptr;
    }
    return yolo;
}

void yolo_infer_init(void *inference, yolo::YOLOConfig *config, int *code) {
    if (inference == nullptr) {
        *code = STATUS_NOT_INITIALIZED;
        return;
    }
    const auto onnx = static_cast<ONNXYolo *>(inference);
    if (onnx->input_shape_.empty() || onnx->outputs_.empty()) {
        *code = STATUS_MODEL_INIT_FAILED;
        return;
    }
    config->input_shape = onnx->input_shape_;
    config->output_shapes.clear();
    config->output_dtypes.clear();
    for (const auto &out : onnx->outputs_) {
        config->output_shapes.push_back(out.shape);
        config->output_dtypes.push_back(out.elem_type);
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
    const auto onnx = static_cast<ONNXYolo *>(inference);
    return onnx->infer(letterbox->pixels, outs, max_outs, code);
}

void yolo_infer_destroy(void *inference) {
    if (inference == nullptr) {
        return;
    }
    const auto onnx = static_cast<ONNXYolo *>(inference);
    delete onnx;
}
}

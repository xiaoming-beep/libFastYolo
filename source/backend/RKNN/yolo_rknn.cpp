// RKNN 推理后端（瑞芯微 NPU：RK3576 / RK3588 等）
// 职责：实现 yolo_infer_* 四个 C ABI 函数，把 libFastYolo 核心层接到 librknnrt
// 结构：RKNNYolo 类（加载/查询/推理） + 色彩转换工具 + extern "C" 契约函数
// 依赖：rknn_api.h + librknnrt.so（路径由 backend/RKNN/CMakeLists.txt 的 RKNN_SDK_DIR 指定）
// 输入约定：核心层 Letterbox::pixels 为 RGB 平面 CHW、0-1 归一化；
//          RKNN 量化模型要 uint8 NHWC 0-255（模型内含 mean=0/std=255），后端内做转换
// 输出约定：rknn_outputs_get(want_float=1) 自动反量化为 float32，核心解码器零改动

#include <rknn_api.h>

#include <fast_yolo/yolo.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32) && defined(LIBFASTYOLO_EXPORTS)
#define YOLO_RKNN_API __declspec(dllexport)
#elif defined(_WIN32)
#define YOLO_RKNN_API __declspec(dllimport)
#else
#define YOLO_RKNN_API
#endif

namespace {

// 量化输入中转：float RGB 平面 CHW（0-1）→ uint8 NHWC（0-255），多线程分行走
// pixels 布局 = [R 平面 H*W][G 平面][B 平面]（Letterbox::fill_bgr 写入顺序）
void chw_float_to_nhwc_u8(const float *pixels, const int width, const int height,
                          std::vector<uint8_t> &dst) {
    const int hw = width * height;
    dst.resize(static_cast<size_t>(hw) * 3);
    const float *r = pixels;
    const float *g = pixels + hw;
    const float *b = pixels + hw * 2;

    auto fill_rows = [&](const int y0, const int y1) {
        for (int y = y0; y < y1; ++y) {
            const float *pr = r + static_cast<size_t>(y) * width;
            const float *pg = g + static_cast<size_t>(y) * width;
            const float *pb = b + static_cast<size_t>(y) * width;
            uint8_t *p = &dst[static_cast<size_t>(y) * width * 3];
            for (int x = 0; x < width; ++x) {
                p[0] = static_cast<uint8_t>(std::clamp(pr[x] * 255.0f + 0.5f, 0.0f, 255.0f));
                p[1] = static_cast<uint8_t>(std::clamp(pg[x] * 255.0f + 0.5f, 0.0f, 255.0f));
                p[2] = static_cast<uint8_t>(std::clamp(pb[x] * 255.0f + 0.5f, 0.0f, 255.0f));
                p += 3;
            }
        }
    };

    // 8 核 A55 上单线程约 3-4ms，4 线程分行走可压到 1ms 内
    const unsigned hc = std::thread::hardware_concurrency();
    const unsigned nthread = std::min(hc == 0 ? 1u : hc, 4u);
    if (nthread <= 1) {
        fill_rows(0, height);
        return;
    }
    const int chunk = (height + static_cast<int>(nthread) - 1) / static_cast<int>(nthread);
    std::vector<std::thread> pool;
    pool.reserve(nthread);
    for (int y0 = 0; y0 < height; y0 += chunk) {
        pool.emplace_back(fill_rows, y0, std::min(height, y0 + chunk));
    }
    for (auto &t: pool) {
        t.join();
    }
}

// float 输入中转：float RGB 平面 CHW（0-1）→ float NHWC（0-1），多线程分行走
void chw_float_to_nhwc_f32(const float *pixels, const int width, const int height,
                           std::vector<float> &dst) {
    const int hw = width * height;
    dst.resize(static_cast<size_t>(hw) * 3);
    const float *r = pixels;
    const float *g = pixels + hw;
    const float *b = pixels + hw * 2;

    // FP 模型输入倍率：本工具链转换时烘焙 mean=0/std=255，故 FP 输入喂 0-255（默认 255）
    // FASTYOLO_RKNN_FLOAT_SCALE 可覆盖（如原生无归一化导出的模型喂 1=0-1）
    static const float fscale = [] {
        const char *e = getenv("FASTYOLO_RKNN_FLOAT_SCALE");
        return e && e[0] ? atof(e) : 255.0f;
    }();
    auto fill_rows = [&](const int y0, const int y1) {
        for (int y = y0; y < y1; ++y) {
            const float *pr = r + static_cast<size_t>(y) * width;
            const float *pg = g + static_cast<size_t>(y) * width;
            const float *pb = b + static_cast<size_t>(y) * width;
            float *p = &dst[static_cast<size_t>(y) * width * 3];
            for (int x = 0; x < width; ++x) {
                p[0] = pr[x] * fscale;
                p[1] = pg[x] * fscale;
                p[2] = pb[x] * fscale;
                p += 3;
            }
        }
    };

    const unsigned hc = std::thread::hardware_concurrency();
    const unsigned nthread = std::min(hc == 0 ? 1u : hc, 4u);
    if (nthread <= 1) {
        fill_rows(0, height);
        return;
    }
    const int chunk = (height + static_cast<int>(nthread) - 1) / static_cast<int>(nthread);
    std::vector<std::thread> pool;
    pool.reserve(nthread);
    for (int y0 = 0; y0 < height; y0 += chunk) {
        pool.emplace_back(fill_rows, y0, std::min(height, y0 + chunk));
    }
    for (auto &t: pool) {
        t.join();
    }
}

class RKNNYolo {
public:
    // 单个输出描述（buffer 常驻，保证核心层在下次 predict 前读取有效）
    struct OutputInfo {
        std::vector<int64_t> shape;  // 与 ONNX 元素类型码并列回填核心层
        std::vector<float> buffer;   // float 模式：反量化 float32 数据
        std::vector<int8_t> i8buf;   // int8 模式（优化图多输出）：原始量化数据
        int32_t zp = 0;              // int8 模式反量化参数
        float scale = 1.0f;
    };

    RKNNYolo(const std::vector<char> &model_data, yolo::InferDevice device, int num_threads) {
        // NPU 后端：device/num_threads 不参与调度，仅打印提示
        (void)num_threads;
        if (device == yolo::InferDevice::CPU) {
            std::cout << "[rknn] 提示: device=CPU 但 RKNN 后端固定跑 NPU" << std::endl;
        }

        const int ret = rknn_init(&ctx_, const_cast<char *>(model_data.data()),
                                  static_cast<uint32_t>(model_data.size()), 0, nullptr);
        if (ret < 0) {
            throw std::runtime_error("rknn_init 失败 ret=" + std::to_string(ret));
        }

        // 查询输入输出数量
        if (rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_)) != RKNN_SUCC) {
            throw std::runtime_error("rknn_query 输入输出数量失败");
        }

        // 输入属性 → 推导 H/W/C（dims 解析与 rknn_model_zoo demo 一致）
        rknn_tensor_attr in_attr;
        memset(&in_attr, 0, sizeof(in_attr));
        in_attr.index = 0;
        if (rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr)) != RKNN_SUCC) {
            throw std::runtime_error("rknn_query 输入属性失败");
        }
        if (in_attr.fmt == RKNN_TENSOR_NCHW) {
            width_ = in_attr.dims[3];
            height_ = in_attr.dims[2];
            channels_ = in_attr.dims[1];
        } else {
            width_ = in_attr.dims[2];
            height_ = in_attr.dims[1];
            channels_ = in_attr.dims[3];
        }
        if (width_ <= 0 || height_ <= 0 || channels_ <= 0) {
            throw std::runtime_error("输入形状非法");
        }
        // 核心层按 NCHW 约定理解 input_shape（Letterbox 是 CHW 平面）
        input_shape_ = {1, channels_, height_, width_};
        // 输入喂入方式按模型输入类型自适应：
        //   INT8/UINT8 → uint8 NHWC 0-255（转换工具链模型，mean=0/std=255 已烘焙）
        //   FP16/FLOAT32 → float NHWC 0-1（ultralytics 原生导出模型，无烘焙归一化）
        const bool fp_input = (in_attr.type == RKNN_TENSOR_FLOAT16 ||
                               in_attr.type == RKNN_TENSOR_FLOAT32);
        input_float_ = fp_input;
        input_uint8_.resize(fp_input ? 0 : static_cast<size_t>(width_) * height_ * channels_);
        input_f32_.resize(fp_input ? static_cast<size_t>(width_) * height_ * channels_ : 0);

        std::cout << "[rknn] 输入: " << width_ << "x" << height_ << "x" << channels_
                  << " fmt=" << get_format_string(in_attr.fmt)
                  << " type=" << get_type_string(in_attr.type)
                  << " 喂入=" << (fp_input ? "float 0-1" : "uint8 0-255") << std::endl;

        // 输出属性 → 预分配缓冲
        // 优化图（输出数 ≥9）走 int8 直读：want_float=1 会强制 runtime 把约 278 万
        // 元素全量转 float（11MB 拷贝+转换，约 5-8ms），int8 直读交给解码侧按需反量化
        int8_mode_ = io_num_.n_output >= 9;
        std::vector<rknn_tensor_attr> out_attrs(io_num_.n_output);
        for (uint32_t i = 0; i < io_num_.n_output; ++i) {
            rknn_tensor_attr attr;
            memset(&attr, 0, sizeof(attr));
            attr.index = i;
            if (rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)) != RKNN_SUCC) {
                throw std::runtime_error("rknn_query 输出属性失败 index=" + std::to_string(i));
            }
            out_attrs[i] = attr;
            OutputInfo out;
            for (uint32_t d = 0; d < attr.n_dims; ++d) {
                out.shape.push_back(attr.dims[d]);
            }
            if (int8_mode_) {
                out.i8buf.resize(attr.n_elems);
                out.zp = attr.zp;
                out.scale = attr.scale;
            } else {
                out.buffer.resize(attr.n_elems);
            }

            std::cout << "[rknn] 输出[" << i << "] shape=[";
            for (size_t j = 0; j < out.shape.size(); ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << out.shape[j];
            }
            std::cout << "] type=" << get_type_string(attr.type)
                      << " n_elems=" << attr.n_elems << std::endl;
            outputs_.push_back(std::move(out));
        }
        if (outputs_.empty()) {
            throw std::runtime_error("模型没有可用输出");
        }

        // sigmoid 通道范围判定（配套模型须先经 strip_sigmoid.py 去输出 Sigmoid，
        // int8 下 Sigmoid 输出分支会整体饱和为 0）：
        //   检测 [1, 4+nc, N]      → 分数行 = 4 .. C-1
        //   分割 [1, 4+nc+32, N]   → 分数行 = 4 .. C-33（末尾 32 行是掩码系数，线性值）
        // 是否真需要补 sigmoid 由首帧自动判定（分数区最大值 >1 视为 logits），
        // 可用 FASTYOLO_RKNN_NO_SIG=0 强制关闭
        const char *nosig = getenv("FASTYOLO_RKNN_NO_SIG");
        if (!(nosig && nosig[0] == '0') && !outputs_[0].shape.empty() && outputs_[0].shape[0] != 0) {
            const auto &s0 = outputs_[0].shape;
            const bool seg = (io_num_.n_output >= 2 && out_attrs[1].n_dims == 4);
            if (s0.size() == 3 && s0[1] > 4) {
                score_begin_ = 4;
                score_end_ = s0[1] - (seg ? 32 : 0);
                if (score_end_ <= score_begin_) {
                    score_begin_ = score_end_ = 0;
                }
            }
        }
    }

    ~RKNNYolo() {
        if (ctx_) {
            rknn_destroy(ctx_);
            ctx_ = 0;
        }
    }

    // 单帧推理：letterbox 像素 → NPU → 反量化输出
    int infer(const std::vector<float> &pixels, yolo::YoloInferOutput *outs,
              const int max_outs, int *code) {
        rknn_input in;
        memset(&in, 0, sizeof(in));
        in.index = 0;
        in.fmt = RKNN_TENSOR_NHWC;
        if (input_float_) {
            chw_float_to_nhwc_f32(pixels.data(), width_, height_, input_f32_);
            in.type = RKNN_TENSOR_FLOAT32;
            in.size = static_cast<uint32_t>(input_f32_.size() * sizeof(float));
            in.buf = input_f32_.data();
        } else {
            chw_float_to_nhwc_u8(pixels.data(), width_, height_, input_uint8_);
            in.type = RKNN_TENSOR_UINT8;
            in.size = static_cast<uint32_t>(input_uint8_.size());
            in.buf = input_uint8_.data();
        }
        if (rknn_inputs_set(ctx_, io_num_.n_input, &in) < 0) {
            *code = STATUS_INFERENCE_FAILED;
            return 0;
        }
        if (rknn_run(ctx_, nullptr) < 0) {
            *code = STATUS_INFERENCE_FAILED;
            return 0;
        }

        std::vector<rknn_output> rk_outs(io_num_.n_output);
        memset(rk_outs.data(), 0, sizeof(rknn_output) * io_num_.n_output);
        for (uint32_t i = 0; i < io_num_.n_output; ++i) {
            rk_outs[i].index = i;
            rk_outs[i].want_float = int8_mode_ ? 0 : 1; // int8 直读由解码侧按需反量化
        }
        if (rknn_outputs_get(ctx_, io_num_.n_output, rk_outs.data(), nullptr) < 0) {
            *code = STATUS_INFERENCE_FAILED;
            return 0;
        }

        const int n = std::min(static_cast<int>(io_num_.n_output), max_outs);
        for (int i = 0; i < n; ++i) {
            auto &out = outputs_[static_cast<size_t>(i)];
            yolo::YoloInferOutput &dst = outs[i];
            dst.rank = static_cast<int>(out.shape.size());
            for (int d = 0; d < dst.rank; ++d) {
                dst.shape[d] = out.shape[static_cast<size_t>(d)];
            }
            if (int8_mode_) {
                // int8 直读：原始量化数据 + zp/scale 交给解码侧
                memcpy(out.i8buf.data(), rk_outs[i].buf, out.i8buf.size());
                dst.data = out.i8buf.data();
                dst.size = static_cast<int64_t>(out.i8buf.size());
                dst.elem_type = 3;
                dst.zp = out.zp;
                dst.scale = out.scale;
                // 调试：FASTYOLO_RKNN_DEBUG=1 时打印每个 int8 输出的统计
                if (getenv("FASTYOLO_RKNN_DEBUG") != nullptr) {
                    int mn = 127, mx = -128;
                    double avg = 0;
                    for (const int8_t q: out.i8buf) {
                        mn = std::min<int>(mn, q);
                        mx = std::max<int>(mx, q);
                        avg += q;
                    }
                    avg /= static_cast<double>(out.i8buf.size());
                    printf("[rknn][i8] 输出[%d] q范围=[%d,%d] 均值=%.1f zp=%d scale=%f\n",
                           i, mn, mx, avg, out.zp, out.scale);
                }
                continue;
            }
            const size_t bytes = out.buffer.size() * sizeof(float);
            memcpy(out.buffer.data(), rk_outs[i].buf, bytes);

            // strip 模型约定：类别分数行为 logits 时原地补 sigmoid
            // （首帧自动判定：分数区最大值 >1 → logits；否则视为已带 sigmoid，跳过）
            if (i == 0 && !int8_mode_ && score_end_ > score_begin_) {
                if (!sig_decided_) {
                    float smx = 0;
                    const int64_t nn0 = out.shape[2];
                    for (int64_t c = score_begin_; c < score_end_; ++c) {
                        const float *row = out.buffer.data() + static_cast<size_t>(c) * nn0;
                        for (int64_t k = 0; k < nn0; ++k) {
                            smx = std::max(smx, row[k]);
                        }
                    }
                    apply_sigmoid_ = smx > 1.001f;
                    sig_decided_ = true;
                    std::cout << "[rknn] 首帧分数区最大=" << smx << " → "
                              << (apply_sigmoid_ ? "logits，补 sigmoid" : "已带 sigmoid，跳过")
                              << std::endl;
                }
                if (apply_sigmoid_) {
                    const int64_t nn = out.shape[2];
                    for (int64_t c = score_begin_; c < score_end_; ++c) {
                        float *row = out.buffer.data() + static_cast<size_t>(c) * nn;
                        for (int64_t k = 0; k < nn; ++k) {
                            const float v = row[k];
                            row[k] = v <= -8.0f ? 0.0f
                                   : v >= 8.0f ? 1.0f
                                               : 1.0f / (1.0f + std::exp(-v));
                        }
                    }
                }
            }

            dst.data = out.buffer.data();
            dst.size = static_cast<int64_t>(out.buffer.size());
            dst.elem_type = 1; // float32
        }
        rknn_outputs_release(ctx_, io_num_.n_output, rk_outs.data());

        // 调试：FASTYOLO_RKNN_DEBUG=1 时打印输出0统计（判断数据是否正常）
        if (getenv("FASTYOLO_RKNN_DEBUG") != nullptr && n > 0 && !int8_mode_) {
            const auto &b = outputs_[0].buffer;
            float mx = 0;
            for (const float v: b) {
                mx = std::max(mx, v);
            }
            std::cout << "[rknn][debug] out0 前6值:";
            for (size_t k = 0; k < 6 && k < b.size(); ++k) {
                std::cout << " " << b[k];
            }
            std::cout << "  全局最大=" << mx;
            // v8 头 [1, 4+nc, N]：第 4 行之后是类别分数区（sigmoid 后应 0-1）
            const auto &s = outputs_[0].shape;
            if (s.size() == 3 && s[1] > 4) {
                const int64_t nn = s[2];
                float smx = 0;
                for (size_t k = static_cast<size_t>(4 * nn); k < b.size(); ++k) {
                    smx = std::max(smx, b[k]);
                }
                std::cout << "  分数区最大=" << smx;
            }
            std::cout << std::endl;
        }
        *code = STATUS_SUCCESS;
        return n;
    }

    const std::vector<int64_t> &input_shape() const { return input_shape_; }
    const std::vector<OutputInfo> &outputs() const { return outputs_; }

private:
    rknn_context ctx_ = 0;
    rknn_input_output_num io_num_{};
    int width_ = 0;
    int height_ = 0;
    int channels_ = 3;
    int64_t score_begin_ = 0;                // 类别分数行起始（0=无）
    int64_t score_end_ = 0;                  // 类别分数行结束
    bool int8_mode_ = false;                 // 优化图多输出：int8 直读 + 解码侧反量化
    bool sig_decided_ = false;               // 首帧是否已判定分数形态
    bool apply_sigmoid_ = false;             // 判定结果：true=logits 需补 sigmoid
    std::vector<int64_t> input_shape_;       // NCHW 约定 {1,C,H,W}
    std::vector<uint8_t> input_uint8_;       // NHWC 量化输入中转（int8 模型）
    std::vector<float> input_f32_;           // NHWC float 输入中转（FP 模型，0-1）
    bool input_float_ = false;               // true=float 喂入（FP 模型）
    std::vector<OutputInfo> outputs_;        // 反量化输出（常驻缓冲）
};

} // namespace

extern "C" {

// 创建推理引擎实例，返回不透明指针
void *yolo_infer_create(const yolo::YOLOConfig *config) {
    if (config->model_data.empty()) {
        return nullptr;
    }
    try {
        auto rknn = new RKNNYolo(config->model_data, config->device, config->num_threads);
        if (rknn->input_shape().empty() || rknn->outputs().empty()) {
            delete rknn;
            return nullptr;
        }
        return rknn;
    } catch (const std::exception &e) {
        std::cout << "[rknn] 初始化异常: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

// 初始化：将输入/输出 shape 回填到 config（核心层据此做任务识别与解码）
void yolo_infer_init(void *inference, yolo::YOLOConfig *config, int *code) {
    if (inference == nullptr) {
        *code = STATUS_NOT_INITIALIZED;
        return;
    }
    const auto rknn = static_cast<RKNNYolo *>(inference);
    config->input_shape = rknn->input_shape();
    config->output_shapes.clear();
    config->output_dtypes.clear();
    for (const auto &out : rknn->outputs()) {
        config->output_shapes.push_back(out.shape);
        config->output_dtypes.push_back(1); // float32
    }
    config->output_shape = config->output_shapes.front();
    *code = STATUS_SUCCESS;
}

// 执行推理：letterbox 像素进 NPU，反量化输出写回 outs（后端持有，下次 predict 前有效）
int yolo_infer_predict(void *inference,
                       const yolo::Letterbox *letterbox, yolo::YoloInferOutput *outs,
                       const int max_outs, int *code) {
    if (inference == nullptr) {
        *code = STATUS_NOT_INITIALIZED;
        return 0;
    }
    const auto rknn = static_cast<RKNNYolo *>(inference);
    return rknn->infer(letterbox->pixels, outs, max_outs, code);
}

// 销毁推理引擎
void yolo_infer_destroy(void *inference) {
    if (inference == nullptr) {
        return;
    }
    const auto rknn = static_cast<RKNNYolo *>(inference);
    delete rknn;
}

}

/*
 * libFastYolo Python SDK 纯 C 绑定层实现
 *
 * 与 source/jni/yolo_jni.cpp 同模式：标量参数展开、不跨语言传 C++ 对象。
 * 供 python/fastyolo/ 经 ctypes 调用；不依赖任何 Python 头文件/库。
 *
 * 注意：yolo.h 必须先于可能间接定义 NTSTATUS 宏的头文件引入
 * （STATUS_SUCCESS 等宏冲突，JNI 层同款处理）。
 */
#include <fast_yolo/fastyolo_c.h>
#include <fast_yolo/yolo.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace {

/** 句柄内容：配置 + 推理引擎（YOLOInference 析构时自动 destroy 后端） */
struct Ctx {
    std::shared_ptr<yolo::YOLOConfig> config;
    yolo::YOLOInference *inference = nullptr;

    ~Ctx() { delete inference; }
};

// 分配并拷贝（失败返回 nullptr）
template <typename T>
T *alloc_copy(const std::vector<T> &src) {
    if (src.empty()) {
        return nullptr;
    }
    auto *dst = new (std::nothrow) T[src.size()];
    if (dst != nullptr) {
        std::copy(src.begin(), src.end(), dst);
    }
    return dst;
}

}  // namespace

extern "C" {

void *fastyolo_create(const uint8_t *model, size_t model_len,
                      float min_score, float iou_threshold,
                      int device, int num_threads, int gpu_id, int precision,
                      const char *input_name, const char *output_name,
                      int track_frame_rate, int track_buffer,
                      float track_thresh, float high_thresh, float match_thresh,
                      int task, int num_classes, int num_keypoints, int top_k,
                      int *code) {
    *code = STATUS_MODEL_NOT_FOUND;
    if (model == nullptr || model_len == 0) {
        return nullptr;
    }
    try {
        std::vector<char> model_data(model, model + model_len);
        auto config = std::make_shared<yolo::YOLOConfig>(model_data, min_score, iou_threshold);
        config->device = static_cast<yolo::InferDevice>(device);
        config->num_threads = num_threads;
        config->gpu_id = gpu_id;
        config->precision = precision;
        config->input_name = (input_name != nullptr) ? input_name : "images";
        config->output_name = (output_name != nullptr) ? output_name : "";
        config->task = static_cast<yolo::TaskType>(task);
        config->num_classes = num_classes;
        config->num_keypoints = num_keypoints;
        config->top_k = top_k;
        config->track_frame_rate = track_frame_rate;
        config->track_buffer = track_buffer;
        config->track_thresh = track_thresh;
        config->high_thresh = high_thresh;
        config->match_thresh = match_thresh;

        auto *inference = new (std::nothrow) yolo::YOLOInference(config);
        if (inference == nullptr) {
            *code = STATUS_MODEL_INIT_FAILED;
            return nullptr;
        }
        auto *ctx = new (std::nothrow) Ctx{std::move(config), inference};
        if (ctx == nullptr) {
            delete inference;
            *code = STATUS_MODEL_INIT_FAILED;
            return nullptr;
        }
        *code = STATUS_SUCCESS;
        return ctx;
    } catch (...) {
        *code = STATUS_MODEL_INIT_FAILED;
        return nullptr;
    }
}

int fastyolo_init(void *handle,
                  int64_t *in_shape, int in_cap, int *in_rank,
                  int64_t *out_shape, int out_cap, int *out_rank) {
    if (handle == nullptr) {
        return STATUS_NOT_INITIALIZED;
    }
    auto *ctx = static_cast<Ctx *>(handle);
    const int code = ctx->inference->init();
    if (code != STATUS_SUCCESS) {
        return code;
    }

    const auto &is = ctx->config->input_shape;
    const auto &os = ctx->config->output_shape;
    *in_rank = static_cast<int>(is.size());
    *out_rank = static_cast<int>(os.size());
    const int in_n = std::min(static_cast<int>(is.size()), in_cap);
    const int out_n = std::min(static_cast<int>(os.size()), out_cap);
    std::copy_n(is.begin(), in_n, in_shape);
    std::copy_n(os.begin(), out_n, out_shape);
    return STATUS_SUCCESS;
}

int fastyolo_output_info(void *handle,
                         int64_t *out_shapes, int out_cap,
                         int *out_ranks, int *out_dtypes, int out_count_cap,
                         int *out_count) {
    if (handle == nullptr) {
        return STATUS_NOT_INITIALIZED;
    }
    auto *ctx = static_cast<Ctx *>(handle);
    const auto &shapes = ctx->config->output_shapes;
    const auto &dtypes = ctx->config->output_dtypes;
    *out_count = static_cast<int>(shapes.size());

    int written = 0;
    const int n = std::min(static_cast<int>(shapes.size()), out_count_cap);
    for (int i = 0; i < n; ++i) {
        out_ranks[i] = static_cast<int>(shapes[i].size());
        out_dtypes[i] = (i < static_cast<int>(dtypes.size())) ? dtypes[i] : 1;
        for (const int64_t d : shapes[i]) {
            if (written < out_cap) {
                out_shapes[written++] = d;
            }
        }
    }
    return STATUS_SUCCESS;
}

int fastyolo_task(void *handle) {
    if (handle == nullptr) {
        return STATUS_NOT_INITIALIZED;
    }
    auto *ctx = static_cast<Ctx *>(handle);
    return static_cast<int>(ctx->inference->resolve_task());
}

void *fastyolo_letterbox_create(int box_size, float color, int center_crop) {
    return new (std::nothrow) yolo::Letterbox(box_size, color, center_crop != 0);
}

void fastyolo_letterbox_update(void *lb, int width, int height,
                               int *dst_width, int *dst_height) {
    auto *letterbox = static_cast<yolo::Letterbox *>(lb);
    if (letterbox == nullptr) {
        return;
    }
    letterbox->update(width, height);
    *dst_width = letterbox->dst_width;
    *dst_height = letterbox->dst_height;
}

void fastyolo_letterbox_fill_bgr(void *lb, const uint8_t *bgr) {
    auto *letterbox = static_cast<yolo::Letterbox *>(lb);
    if (letterbox == nullptr || bgr == nullptr) {
        return;
    }
    letterbox->fill_bgr(bgr);
}

const float *fastyolo_letterbox_pixels(void *lb, int *len) {
    auto *letterbox = static_cast<yolo::Letterbox *>(lb);
    if (letterbox == nullptr) {
        *len = 0;
        return nullptr;
    }
    *len = static_cast<int>(letterbox->pixels.size());
    return letterbox->pixels.data();
}

static FastyoloResult *predict_or_track(void *handle, void *lb,
                                        float score_threshold, float iou_threshold,
                                        bool need_masks,
                                        int *code, bool tracking) {
    *code = STATUS_SUCCESS;
    if (handle == nullptr || lb == nullptr) {
        *code = STATUS_NOT_INITIALIZED;
        return nullptr;
    }
    auto *ctx = static_cast<Ctx *>(handle);
    auto *letterbox = static_cast<yolo::Letterbox *>(lb);
    auto *result = new (std::nothrow) FastyoloResult{};
    if (result == nullptr) {
        *code = STATUS_INFERENCE_FAILED;
        return nullptr;
    }
    try {
        const yolo::TaskType task = ctx->inference->resolve_task();
        result->task = static_cast<int>(task);
        switch (task) {
            case yolo::TaskType::DETECT: {
                std::vector<yolo::YOLOResult> results;
                if (tracking) {
                    results = ctx->inference->track(*letterbox, code, score_threshold, iou_threshold);
                } else {
                    results = ctx->inference->predict(*letterbox, code, score_threshold, iou_threshold);
                }
                if (*code != STATUS_SUCCESS) {
                    break;
                }
                result->box_count = static_cast<int>(results.size());
                if (result->box_count > 0) {
                    auto *boxes = new (std::nothrow) FastyoloBox[results.size()];
                    result->boxes = boxes;
                    if (boxes == nullptr) {
                        *code = STATUS_INFERENCE_FAILED;
                        break;
                    }
                    for (size_t i = 0; i < results.size(); ++i) {
                        const auto &r = results[i];
                        boxes[i] = {r.x, r.y, r.width, r.height, r.clsId, r.score, r.track_id};
                    }
                }
                break;
            }
            case yolo::TaskType::SEGMENT:
            case yolo::TaskType::POSE:
            case yolo::TaskType::OBB: {
                // 框类任务支持跟踪（SEGMENT 跟踪只出框；掩码在线按需走 JNI track(budgets)）
                std::vector<yolo::YOLOResult> results;
                if (tracking) {
                    results = ctx->inference->track(*letterbox, code, score_threshold, iou_threshold);
                } else {
                    results = ctx->inference->predict(*letterbox, code, score_threshold, iou_threshold,
                                                      need_masks);
                }
                if (*code != STATUS_SUCCESS) {
                    break;
                }
                result->box_count = static_cast<int>(results.size());
                if (result->box_count > 0) {
                    auto *boxes = new (std::nothrow) FastyoloBox[results.size()];
                    result->boxes = boxes;
                    if (boxes == nullptr) {
                        *code = STATUS_INFERENCE_FAILED;
                        break;
                    }
                    for (size_t i = 0; i < results.size(); ++i) {
                        const auto &r = results[i];
                        boxes[i] = {r.x, r.y, r.width, r.height, r.clsId, r.score, r.track_id};
                    }
                }
                // POSE：关键点
                if (task == yolo::TaskType::POSE && result->box_count > 0) {
                    result->keypoint_count = static_cast<int>(results[0].keypoints.size());
                    if (result->keypoint_count > 0) {
                        auto *kpts = new (std::nothrow) FastyoloKeypoint[
                            static_cast<size_t>(result->box_count) * result->keypoint_count];
                        result->keypoints = kpts;
                        if (kpts == nullptr) {
                            *code = STATUS_INFERENCE_FAILED;
                            break;
                        }
                        for (size_t i = 0; i < results.size(); ++i) {
                            const auto &r = results[i];
                            for (size_t k = 0; k < r.keypoints.size(); ++k) {
                                const size_t idx = i * r.keypoints.size() + k;
                                kpts[idx] = {r.keypoints[k].x, r.keypoints[k].y, r.keypoints[k].score};
                            }
                        }
                    }
                }
                // SEGMENT：ROI 掩码按框顺序拼接（每框长度 = 对应框 w*h，0/1）
                if (task == yolo::TaskType::SEGMENT && result->box_count > 0) {
                    size_t total = 0;
                    for (const auto &r : results) {
                        total += r.mask.size();
                    }
                    if (total > 0) {
                        auto *masks = new (std::nothrow) uint8_t[total];
                        result->masks = masks;
                        if (masks == nullptr) {
                            *code = STATUS_INFERENCE_FAILED;
                            break;
                        }
                        size_t off = 0;
                        for (const auto &r : results) {
                            if (r.mask.empty()) {
                                continue;
                            }
                            std::copy(r.mask.begin(), r.mask.end(), masks + off);
                            off += r.mask.size();
                        }
                    }
                }
                // OBB：旋转角（弧度，与 boxes 平行）
                if (task == yolo::TaskType::OBB && result->box_count > 0) {
                    auto *angles = new (std::nothrow) float[results.size()];
                    result->angles = angles;
                    if (angles == nullptr) {
                        *code = STATUS_INFERENCE_FAILED;
                        break;
                    }
                    for (size_t i = 0; i < results.size(); ++i) {
                        angles[i] = results[i].angle;
                    }
                }
                break;
            }
            case yolo::TaskType::CLASSIFY: {
                // 稠密任务不支持跟踪（无检测框）
                if (tracking) {
                    *code = STATUS_TASK_UNSUPPORTED;
                    break;
                }
                const yolo::ClassifyResult r = ctx->inference->classify(*letterbox, code);
                if (*code != STATUS_SUCCESS) {
                    break;
                }
                result->top_k = r.top_k;
                result->class_count = static_cast<int>(r.scores.size());
                result->class_ids = alloc_copy(r.top_ids);
                result->class_scores = alloc_copy(r.top_scores);
                result->all_class_scores = alloc_copy(r.scores);
                break;
            }
            case yolo::TaskType::SEMSEG:
            case yolo::TaskType::DEPTH: {
                // 稠密任务不支持跟踪（无检测框）
                if (tracking) {
                    *code = STATUS_TASK_UNSUPPORTED;
                    break;
                }
                const yolo::DenseResult r = (task == yolo::TaskType::SEMSEG)
                                                ? ctx->inference->semseg(*letterbox, code)
                                                : ctx->inference->depth(*letterbox, code);
                if (*code != STATUS_SUCCESS) {
                    break;
                }
                result->map_w = r.width;
                result->map_h = r.height;
                if (task == yolo::TaskType::SEMSEG) {
                    result->class_map = alloc_copy(r.class_map);
                } else {
                    result->depth_map = alloc_copy(r.depth_map);
                    result->depth_min = r.depth_min;
                    result->depth_max = r.depth_max;
                }
                break;
            }
            default:
                *code = STATUS_TASK_UNSUPPORTED;
                break;
        }
    } catch (...) {
        *code = STATUS_INFERENCE_FAILED;
    }
    if (*code != STATUS_SUCCESS) {
        fastyolo_free_result(result);
        return nullptr;
    }
    return result;
}

FastyoloResult *fastyolo_predict(void *handle, void *lb,
                                 float score_threshold, float iou_threshold,
                                 int need_masks,
                                 int *code) {
    return predict_or_track(handle, lb, score_threshold, iou_threshold,
                            need_masks != 0, code, false);
}

FastyoloResult *fastyolo_track(void *handle, void *lb,
                               float score_threshold, float iou_threshold,
                               int *code) {
    return predict_or_track(handle, lb, score_threshold, iou_threshold,
                            false, code, true);
}

void fastyolo_free_result(FastyoloResult *result) {
    if (result == nullptr) {
        return;
    }
    delete[] result->boxes;
    delete[] result->keypoints;
    delete[] result->masks;
    delete[] result->angles;
    delete[] result->class_ids;
    delete[] result->class_scores;
    delete[] result->all_class_scores;
    delete[] result->class_map;
    delete[] result->depth_map;
    delete result;
}

void fastyolo_letterbox_destroy(void *lb) {
    delete static_cast<yolo::Letterbox *>(lb);
}

void fastyolo_destroy(void *handle) {
    delete static_cast<Ctx *>(handle);
}

}  // extern "C"

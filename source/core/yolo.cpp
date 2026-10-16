#include <fast_yolo/yolo.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fast_yolo/img_util.h>
#include <iostream>
#include <ByteTrack/BYTETracker.h>
#include "postprocess.h"


void yolo::Letterbox::update(const int width, const int height) {
    std::fill(pixels.begin(), pixels.end(), this->def_color);
    this->img_width = width;
    this->img_height = height;
    if (center_crop_) {
        // 分类模式：按最短边缩放（取较大比例），一边 == box_size，另一边 ≥ box_size（随后中心裁剪）
        this->scale = std::max(static_cast<float>(box_size) / static_cast<float>(width),
                               static_cast<float>(box_size) / static_cast<float>(height));
        this->dst_width = static_cast<int>(std::round(static_cast<float>(width) * scale));
        this->dst_height = static_cast<int>(std::round(static_cast<float>(height) * scale));
        this->scaled_x = 0;
        this->scaled_y = 0;
        return;
    }
    // 计算缩放
    this->scale = std::min(static_cast<float>(box_size) / static_cast<float>(width),
                           static_cast<float>(box_size) / static_cast<float>(height));
    this->dst_width = static_cast<int>(std::round(static_cast<float>(width) * scale));
    this->dst_height = static_cast<int>(std::round(static_cast<float>(height) * scale));
    this->scaled_x = static_cast<float>(box_size - dst_width) / 2;
    this->scaled_y = static_cast<float>(box_size - dst_height) / 2;
}

/**
 * 构建结果
 */
yolo::YOLOResult yolo::Letterbox::complete_result(const nms::Detection &detection) const {
    const float x1 = (detection.x1 - scaled_x) / scale;
    const float y1 = (detection.y1 - scaled_y) / scale;
    const float x2 = (detection.x2 - scaled_x) / scale;
    const float y2 = (detection.y2 - scaled_y) / scale;
    int x = static_cast<int>(std::round(x1));
    int y = static_cast<int>(std::round(y1));
    int w = static_cast<int>(std::round(x2 - x1));
    int h = static_cast<int>(std::round(y2 - y1));
    x = std::max(0, x);
    y = std::max(0, y);
    w = std::max(1, std::min(w, img_width - x));
    h = std::max(1, std::min(h, img_height - y));
    return {x, y, w, h, detection.class_id, detection.score};
}


void yolo::Letterbox::fill_bgr(const uint8_t *data) {
    if (center_crop_) {
        center_crop_fill_bgr(box_size, dst_width, dst_height, data, pixels.data());
        return;
    }
    if (dst_width < box_size) {
        horizontal_fill_bgr(box_size, dst_width, dst_height, data, pixels.data());
        return;
    }
    vertical_fill_bgr(box_size, dst_width, dst_height, data, pixels.data());
}

//初始化
int yolo::YOLOInference::init() {
    if (this->context_) {
        return STATUS_SUCCESS;
    }
    //内容创建
    void *content = yolo_infer_create(config_.get());
    if (!content) {
        return STATUS_MODEL_INIT_FAILED;
    }
    int code;
    //初始化
    yolo_infer_init(content, config_.get(), &code);
    this->context_ = content;
    if (code != STATUS_SUCCESS) {
        destroy();
    }
    return code;
}

/**
 * 处理结果
 */
std::vector<yolo::YOLOResult> detect_to_result(const yolo::Letterbox &letterbox,
                                               const std::vector<nms::Detection> &results) {
    std::vector<yolo::YOLOResult> arr;
    arr.reserve(results.size());
    for (const auto& detection: results) {
        arr.push_back(letterbox.complete_result(detection));
    }
    return arr;
}

/**
 * 执行后端推理，填充输出数组（后端持有，下一次 predict 前有效）
 */
int yolo::YOLOInference::infer_outputs(const Letterbox &letterbox, YoloInferOutput *outs,
                                       const int max_outs, int *code) const {
    if (!context_) {
        *code = STATUS_NOT_INITIALIZED;
        return 0;
    }
    return yolo_infer_predict(this->context_, &letterbox, outs, max_outs, code);
}

/**
 * 推断任务类型（config->task == AUTO 时按输出形状推断）
 *
 * 规则（歧义时用户可显式指定 task）：
 *   rank2            → CLASSIFY       [1, N]
 *   rank3 + 非float → SEMSEG         [1, H, W] 类别图（uint8/int64）
 *   rank4 + 通道1    → DEPTH          [1, 1, H, W]
 *   双输出 + 第2个4D → SEGMENT        检测头 + [1, 32, H, W] 原型掩码
 *   rank3 + 列≤64    → 端到端（免 NMS）：6→DETECT, 7→OBB, 38→SEGMENT
 *   rank3 + 行>300   → POSE（C=4+nc+3k 且 k≥10）否则 DETECT（v8 OBB 需显式指定）
 */
yolo::TaskType yolo::YOLOInference::resolve_task() const {
    if (config_->task != TaskType::AUTO) {
        return config_->task;
    }
    const auto &shapes = config_->output_shapes;
    if (shapes.empty()) {
        return TaskType::DETECT;
    }
    const auto &s0 = shapes[0];
    const int64_t rank = static_cast<int64_t>(s0.size());
    if (rank == 2) {
        return TaskType::CLASSIFY;
    }
    if (rank == 3 && !config_->output_dtypes.empty() && config_->output_dtypes[0] != 1) {
        return TaskType::SEMSEG;   // [1, H, W] uint8/int64 类别图
    }
    if (rank == 4 && s0[1] == 1) {
        return TaskType::DEPTH;
    }
    if (shapes.size() == 9) {
        return TaskType::DETECT;  // RKNN 优化图: 3 尺度 × (box/score/sum)，须先于 SEGMENT 判定
    }
    if (shapes.size() >= 2 && shapes[1].size() == 4) {
        return TaskType::SEGMENT;
    }
    if (rank == 3 && s0[2] <= 64 && s0[1] <= 300) {
        switch (s0[2]) {
            case 7:  return TaskType::OBB;
            case 38: return TaskType::SEGMENT;
            default: return TaskType::DETECT; // 6 等
        }
    }
    if (rank == 3 && s0[1] > 300) {
        // v8 风格头 [1, C, N]：POSE 判定（C = 4 + nc + 3k，默认 nc=1）
        const int64_t c = s0[1];
        if ((c - 5) % 3 == 0 && (c - 5) / 3 >= 10) {
            return TaskType::POSE;
        }
    }
    return TaskType::DETECT;
}

//推理
std::vector<yolo::YOLOResult> yolo::YOLOInference::predict(const Letterbox &letterbox, int *code,
                                                           const float score_threshold,
                                                           const float iou_threshold,
                                                           const bool need_masks) const {
    const bool t_on = [] {
        const char *e = getenv("FASTYOLO_TIMING");
        return e && e[0] != '\0';
    }();
    const auto t_p0 = std::chrono::steady_clock::now();
    YoloInferOutput outs[12];
    const int out_count = infer_outputs(letterbox, outs, 12, code);
    const auto t_p1 = std::chrono::steady_clock::now();
    if (*code != STATUS_SUCCESS || out_count <= 0) {
        return {};
    }
    const float min_s = score_threshold >= 0.0f ? score_threshold : config_->min_score;
    const float iou_t = iou_threshold >= 0.0f ? iou_threshold : config_->iou_threshold;
    const TaskType task = resolve_task();
    std::vector<yolo::YOLOResult> out;
    switch (task) {
        case TaskType::DETECT: {
            const std::vector<nms::Detection> results = decode_detect(
                outs, out_count, *config_, letterbox, min_s, iou_t, nms_processor_);
            out = detect_to_result(letterbox, results);
            break;
        }
        case TaskType::SEGMENT: {
            // 两阶段：先解码+NMS 出框（便宜，每帧必跑）；掩码仅在 need_masks 时全量生成
            auto segBoxes = decode_segment_boxes(outs, out_count, *config_, letterbox,
                                                 min_s, iou_t, code);
            if (*code != STATUS_SUCCESS) {
                return {};
            }
            if (need_masks && !segBoxes.empty()) {
                std::vector<int> picks(segBoxes.size());
                for (size_t i = 0; i < picks.size(); ++i) {
                    picks[i] = static_cast<int>(i);
                }
                build_segment_masks(outs, *config_, letterbox, segBoxes, picks);
            }
            out.reserve(segBoxes.size());
            for (auto &b : segBoxes) {
                out.push_back(std::move(b.r));
            }
            break;
        }
        case TaskType::POSE:
            out = decode_pose(outs, out_count, *config_, letterbox, min_s, iou_t, code);
            break;
        case TaskType::OBB:
            out = decode_obb(outs, out_count, *config_, letterbox, min_s, iou_t, code);
            break;
        case TaskType::CLASSIFY:
        case TaskType::SEMSEG:
        case TaskType::DEPTH:
        default:
            // 稠密/分类任务请使用 classify() / semseg() / depth() 入口
            *code = STATUS_TASK_UNSUPPORTED;
            return {};
    }
    if (t_on) {
        const double infer_ms =
            std::chrono::duration<double, std::milli>(t_p1 - t_p0).count();
        const double post_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_p1)
                .count();
        fprintf(stderr, "[predict] infer=%.2fms post=%.2fms dets=%zu\n", infer_ms, post_ms,
                out.size());
    }
    return out;
}

//图像分类
yolo::ClassifyResult yolo::YOLOInference::classify(const Letterbox &letterbox, int *code) const {
    YoloInferOutput outs[12];
    const int out_count = infer_outputs(letterbox, outs, 12, code);
    if (*code != STATUS_SUCCESS || out_count <= 0) {
        return {};
    }
    if (resolve_task() != TaskType::CLASSIFY) {
        *code = STATUS_TASK_UNSUPPORTED;
        return {};
    }
    return decode_classify(outs, out_count, *config_, code);
}

//语义分割
yolo::DenseResult yolo::YOLOInference::semseg(const Letterbox &letterbox, int *code) const {
    YoloInferOutput outs[12];
    const int out_count = infer_outputs(letterbox, outs, 12, code);
    if (*code != STATUS_SUCCESS || out_count <= 0) {
        return {};
    }
    if (resolve_task() != TaskType::SEMSEG) {
        *code = STATUS_TASK_UNSUPPORTED;
        return {};
    }
    return decode_dense(outs, out_count, *config_, letterbox, TaskType::SEMSEG, code);
}

//深度估计
yolo::DenseResult yolo::YOLOInference::depth(const Letterbox &letterbox, int *code) const {
    YoloInferOutput outs[12];
    const int out_count = infer_outputs(letterbox, outs, 12, code);
    if (*code != STATUS_SUCCESS || out_count <= 0) {
        return {};
    }
    if (resolve_task() != TaskType::DEPTH) {
        *code = STATUS_TASK_UNSUPPORTED;
        return {};
    }
    return decode_dense(outs, out_count, *config_, letterbox, TaskType::DEPTH, code);
}

//跟踪推理（SEGMENT 掩码懒加载：budgets 驱动，见 MaskBudget）
std::vector<yolo::YOLOResult> yolo::YOLOInference::track(const Letterbox &letterbox, int *code,
                                                          const float score_threshold,
                                                          const float iou_threshold,
                                                          const MaskBudget *budgets,
                                                          const int budget_count) {
    // ByteTrack 仅支持有检测框的任务（DETECT/SEGMENT/POSE/OBB）
    const TaskType task = resolve_task();
    if (task != TaskType::DETECT && task != TaskType::SEGMENT &&
        task != TaskType::POSE && task != TaskType::OBB) {
        *code = STATUS_TASK_UNSUPPORTED;
        return {};
    }
    const float min_s = score_threshold >= 0.0f ? score_threshold : config_->min_score;
    const float iou_t = iou_threshold >= 0.0f ? iou_threshold : config_->iou_threshold;

    const bool is_seg = (task == TaskType::SEGMENT);

    // ---- 1. 本帧检测框（SEGMENT 用两阶段：先只出框；其余任务走 predict 带载荷） ----
    YoloInferOutput outs[12];
    std::vector<SegBox> segBoxes;          // SEGMENT：框 + anchor/letterbox 载荷
    std::vector<YOLOResult> results;       // 非 SEGMENT：完整结果（POSE 关键点/OBB 角度）
    size_t src_n = 0;
    if (is_seg) {
        const int out_count = infer_outputs(letterbox, outs, 12, code);
        if (*code != STATUS_SUCCESS || out_count <= 0) {
            return {};
        }
        segBoxes = decode_segment_boxes(outs, out_count, *config_, letterbox,
                                        min_s, iou_t, code);
        if (*code != STATUS_SUCCESS) {
            return {};
        }
        src_n = segBoxes.size();
    } else {
        results = predict(letterbox, code, score_threshold, iou_threshold, false);
        if (*code != STATUS_SUCCESS) {
            return {};
        }
        src_n = results.size();
    }
    if (src_n == 0) {
        return {};
    }

    // ---- 2. Lazy init tracker on first call ----
    if (!tracker_) {
        tracker_ = new byte_track::BYTETracker(
            config_->track_frame_rate,
            config_->track_buffer,
            config_->track_thresh,
            config_->high_thresh,
            config_->match_thresh
        );
    }

    // ---- 3. Convert boxes → ByteTrack Object（带原始下标，跟踪后回填载荷） ----
    std::vector<byte_track::Object> objects;
    objects.reserve(src_n);
    for (size_t i = 0; i < src_n; ++i) {
        const YOLOResult &r = is_seg ? segBoxes[i].r : results[i];
        objects.emplace_back(
            byte_track::Rect<float>(
                static_cast<float>(r.x), static_cast<float>(r.y),
                static_cast<float>(r.width), static_cast<float>(r.height)
            ),
            r.clsId, r.score, static_cast<int>(i)
        );
    }

    // ---- 4. ByteTrack update ----
    auto *tracker = static_cast<byte_track::BYTETracker *>(tracker_);
    auto tracked = tracker->update(objects);

    // ---- 5. SEGMENT：预算表决定哪些框需要掩码（含重叠仲裁补偿） ----
    //     budgets == null / 空表 = 本帧无人消费掩码 → 全部懒（只出框）
    std::vector<bool> want_mask(src_n, false);
    if (is_seg && budgets != nullptr && budget_count > 0) {
        auto budget_area = [&](int trackId) -> float {
            for (int i = 0; i < budget_count; ++i) {
                if (budgets[i].trackId == trackId) {
                    return budgets[i].bestArea;
                }
            }
            return -1.0f;  // 无记录 = 新船/尚无 best → 需要掩码
        };
        auto overlap01 = [](const YOLOResult &a, const YOLOResult &b) -> bool {
            const int ix1 = std::max(a.x, b.x);
            const int iy1 = std::max(a.y, b.y);
            const int ix2 = std::min(a.x + a.width, b.x + b.width);
            const int iy2 = std::min(a.y + a.height, b.y + b.height);
            return ix2 > ix1 && iy2 > iy1;  // IoU > 0
        };
        // 5.1 预算判定：无 best 或本帧框面积超过 best → 候选
        for (const auto &tobj : tracked) {
            const int det = tobj->getDetIndex();
            if (det < 0 || det >= static_cast<int>(src_n)) {
                continue;
            }
            const YOLOResult &r = segBoxes[static_cast<size_t>(det)].r;
            const float ba = budget_area(tobj->getTrackId());
            want_mask[static_cast<size_t>(det)] = (ba < 0.0f)
                || (static_cast<float>(r.width) * r.height > ba);
        }
        // 5.2 重叠补偿：候选框与未候选跟踪框相交时，对方也生成掩码（供 contested 仲裁）
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t i = 0; i < tracked.size() && !changed; ++i) {
                const int di = tracked[i]->getDetIndex();
                if (di < 0 || di >= static_cast<int>(src_n)) {
                    continue;
                }
                if (!want_mask[static_cast<size_t>(di)]) {
                    continue;
                }
                for (size_t j = 0; j < tracked.size(); ++j) {
                    const int dj = tracked[j]->getDetIndex();
                    if (dj < 0 || dj >= static_cast<int>(src_n) || di == dj) {
                        continue;
                    }
                    if (!want_mask[static_cast<size_t>(dj)]
                        && overlap01(segBoxes[static_cast<size_t>(di)].r,
                                     segBoxes[static_cast<size_t>(dj)].r)) {
                        want_mask[static_cast<size_t>(dj)] = true;
                        changed = true;
                        break;
                    }
                }
            }
        }
        std::vector<int> picks;
        for (size_t i = 0; i < src_n; ++i) {
            if (want_mask[i]) {
                picks.push_back(static_cast<int>(i));
            }
        }
        build_segment_masks(outs, *config_, letterbox, segBoxes, picks);
    }

    // ---- 6. Build results with track_id ----
    std::vector<YOLOResult> tracked_results;
    tracked_results.reserve(tracked.size());
    for (const auto &tobj : tracked) {
        const int det_idx = tobj->getDetIndex();
        if (det_idx < 0 || det_idx >= static_cast<int>(src_n)) {
            continue;
        }
        if (!is_seg && task == TaskType::DETECT) {
            // 原行为：返回跟踪器的预测框（Kalman 平滑）
            const auto &rect = tobj->getRect();
            YOLOResult result(
                static_cast<int>(rect.x()),
                static_cast<int>(rect.y()),
                static_cast<int>(rect.width()),
                static_cast<int>(rect.height()),
                tobj->getLabel(),
                tobj->getScore()
            );
            result.track_id = static_cast<int>(tobj->getTrackId());
            tracked_results.push_back(result);
        } else {
            // SEGMENT/POSE/OBB：保留原始检测框与载荷（掩码/关键点/角度必须与框对应），
            // 仅回填 track_id
            YOLOResult result = is_seg ? std::move(segBoxes[static_cast<size_t>(det_idx)].r)
                                       : results[static_cast<size_t>(det_idx)];
            result.track_id = static_cast<int>(tobj->getTrackId());
            tracked_results.push_back(std::move(result));
        }
    }
    return tracked_results;
}

void yolo::YOLOInference::destroy() {
    if (tracker_) {
        delete static_cast<byte_track::BYTETracker *>(tracker_);
        tracker_ = nullptr;
    }
    if (!this->context_) {
        return;
    }
    yolo_infer_destroy(this->context_);
    this->context_ = nullptr;
}

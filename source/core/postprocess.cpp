#include "postprocess.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>

namespace yolo {
namespace {

constexpr int ONNX_FLOAT = 1;
constexpr int ONNX_UINT8 = 2;
constexpr int ONNX_INT64 = 7;

inline float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// 阶段计时（仅环境变量 FASTYOLO_TIMING=1 时输出，默认零开销）
struct StageTimer {
    using clk = std::chrono::steady_clock;
    static bool on() {
        static const bool v = [] {
            const char *e = getenv("FASTYOLO_TIMING");
            return e && e[0] != '\0';
        }();
        return v;
    }
    clk::time_point t0;
    void start() { t0 = clk::now(); }
    double ms() const { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }
};

// 通道优先 [1, C, N] 布局访问
struct ChannelHead {
    const float *data;
    int64_t C;
    int64_t N;
    float at(int64_t c, int64_t n) const {
        return data[c * N + n];
    }
};

// 行优先 [1, N, W] 布局访问
struct RowHead {
    const float *data;
    int64_t N;
    int64_t W;
    const float *row(int64_t n) const {
        return data + n * W;
    }
};

// 候选（anchor 索引作载荷，用于取掩码系数/关键点/角度）
struct Candidate {
    float x1, y1, x2, y2;  // letterbox 坐标 xyxy
    float score;
    int class_id;
    int anchor;
};

// 最近邻缩放 uint8
void resize_nearest_u8(const uint8_t *src, int sw, int sh, uint8_t *dst, int dw, int dh) {
    for (int y = 0; y < dh; ++y) {
        const int sy = std::min((y * sh) / dh, sh - 1);
        const uint8_t *srow = src + sy * sw;
        uint8_t *drow = dst + y * dw;
        for (int x = 0; x < dw; ++x) {
            drow[x] = srow[std::min((x * sw) / dw, sw - 1)];
        }
    }
}

// 双线性缩放 float（OpenCV 同公式：src = (dst + 0.5) * scale - 0.5，坐标 clamp）
void resize_bilinear_f32(const float *src, int sw, int sh, float *dst, int dw, int dh) {
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) {
        return;
    }
    const float scale_x = static_cast<float>(sw) / static_cast<float>(dw);
    const float scale_y = static_cast<float>(sh) / static_cast<float>(dh);
    std::vector<int> x0s(dw);
    std::vector<float> fxs(dw);
    for (int x = 0; x < dw; ++x) {
        const float sx = (x + 0.5f) * scale_x - 0.5f;
        const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, sw - 1);
        x0s[x] = x0;
        fxs[x] = sx - std::floor(sx);
    }
    for (int y = 0; y < dh; ++y) {
        const float sy = (y + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, sh - 1);
        const int y1 = std::clamp(y0 + 1, 0, sh - 1);
        const float fy = sy - std::floor(sy);
        const float *s0 = src + y0 * sw;
        const float *s1 = src + y1 * sw;
        float *drow = dst + y * dw;
        for (int x = 0; x < dw; ++x) {
            const int x0 = x0s[x];
            const int x1 = std::clamp(x0 + 1, 0, sw - 1);
            const float fx = fxs[x];
            drow[x] = (s0[x0] * (1.0f - fx) + s0[x1] * fx) * (1.0f - fy) +
                      (s1[x0] * (1.0f - fx) + s1[x1] * fx) * fy;
        }
    }
}

// v8 头候选解码：通道 0-3 = xywh（中心格式），4..4+nc-1 = 类别分数
void decode_v8_candidates(const float *data, int64_t C, int64_t N, int64_t nc,
                          float min_score, std::vector<Candidate> &out) {
    std::vector<float> best(static_cast<size_t>(N));
    std::vector<int32_t> best_cls(static_cast<size_t>(N));
    const float *score_ptr = data + 4 * N;
    for (int64_t c = 4; c < 4 + nc; ++c, score_ptr += N) {
        for (int64_t n = 0; n < N; ++n) {
            const float s = score_ptr[n];
            if (s > best[n]) {
                best[n] = s;
                best_cls[n] = static_cast<int32_t>(c - 4);
            }
        }
    }
    for (int64_t n = 0; n < N; ++n) {
        if (best[n] < min_score) {
            continue;
        }
        const float x = data[n];
        const float y = data[N + n];
        const float w = data[2 * N + n];
        const float h = data[3 * N + n];
        out.push_back({x - w * 0.5f, y - h * 0.5f, x + w * 0.5f, y + h * 0.5f,
                       best[n], best_cls[n], static_cast<int>(n)});
    }
}

// e2e 行候选解码：行 = [x1, y1, x2, y2, score, cls, ...]
void decode_e2e_candidates(const float *data, int64_t N, int64_t W, float min_score,
                           std::vector<Candidate> &out) {
    for (int64_t n = 0; n < N; ++n) {
        const float *row = data + n * W;
        if (row[4] < min_score) {
            continue;
        }
        out.push_back({row[0], row[1], row[2], row[3], row[4],
                       static_cast<int>(row[5]), static_cast<int>(n)});
    }
}

// 原图框（letterbox 逆变换 + 截断，与 complete_result 一致）
struct OrigBox {
    int x, y, w, h;
};

OrigBox to_orig_box(const Letterbox &lb, float x1, float y1, float x2, float y2) {
    const float ox1 = lb.map_back_x(x1);
    const float oy1 = lb.map_back_y(y1);
    const float ox2 = lb.map_back_x(x2);
    const float oy2 = lb.map_back_y(y2);
    int x = static_cast<int>(std::round(ox1));
    int y = static_cast<int>(std::round(oy1));
    int w = static_cast<int>(std::round(ox2 - ox1));
    int h = static_cast<int>(std::round(oy2 - oy1));
    x = std::max(0, x);
    y = std::max(0, y);
    w = std::max(1, std::min(w, lb.img_width - x));
    h = std::max(1, std::min(h, lb.img_height - y));
    return {x, y, w, h};
}

YOLOResult make_result(const Letterbox &lb, const Candidate &c) {
    const OrigBox ob = to_orig_box(lb, c.x1, c.y1, c.x2, c.y2);
    return YOLOResult(ob.x, ob.y, ob.w, ob.h, c.class_id, c.score);
}

// 候选 → Detection 列表 + NMS
std::vector<int32_t> nms_candidates(std::vector<Candidate> &cands, float iou_threshold) {
    std::vector<nms::Detection> dets;
    dets.reserve(cands.size());
    for (const auto &c : cands) {
        dets.emplace_back(c.x1, c.y1, c.x2, c.y2, c.score, c.class_id);
    }
    return nms::nms_keep(dets, iou_threshold);
}

// 输入分辨率（正方形 letterbox 边长）
int letterbox_size(const YOLOConfig &config, const Letterbox &lb) {
    if (config.input_shape.size() >= 4 && config.input_shape[3] > 0) {
        return static_cast<int>(config.input_shape[3]);
    }
    return lb.dst_width > 0 ? lb.dst_width : 640;
}

}  // namespace（内部 helper 结束；以下解码函数对 core/yolo.cpp 导出）

// ------------------------------------------------------------------ //
// SEGMENT：两阶段懒加载 —— ① 解码+NMS 只出框；② 按需生成 ROI 掩码
// ------------------------------------------------------------------ //
std::vector<SegBox> decode_segment_boxes(const YoloInferOutput *outs, int out_count,
                                         const YOLOConfig &config, const Letterbox &lb,
                                         float min_score, float iou_threshold, int *code) {
    std::vector<SegBox> boxes;
    if (out_count < 2 || !outs[0].data || !outs[1].data || outs[0].elem_type != ONNX_FLOAT) {
        *code = STATUS_INFERENCE_FAILED;
        return boxes;
    }
    const auto &s0 = config.output_shapes[0];
    const auto &s1 = config.output_shapes[1];
    if (s0.size() < 3 || s1.size() < 4 || s1[1] != 32) {
        *code = STATUS_INFERENCE_FAILED;
        return boxes;
    }
    const int64_t C = s0[1];
    const int64_t N = s0[2];
    constexpr int64_t kCoeffs = 32;
    const float *data0 = static_cast<const float *>(outs[0].data);

    // e2e 行布局 [1, N, 6+32]（N≤300）；否则 v8 头 [1, C, N]（C = 4 + nc + 32）
    const bool e2e = (s0[2] <= 64 && s0[1] <= 300);
    StageTimer tt;
    double t_cand = 0, t_nms = 0;
    std::vector<Candidate> cands;
    tt.start();
    if (e2e) {
        if (s0[2] != 6 + kCoeffs) {
            *code = STATUS_INFERENCE_FAILED;
            return boxes;
        }
        decode_e2e_candidates(data0, s0[1], s0[2], min_score, cands);
    } else {
        if (C <= 4 + kCoeffs) {
            *code = STATUS_INFERENCE_FAILED;
            return boxes;
        }
        decode_v8_candidates(data0, C, N, C - 4 - kCoeffs, min_score, cands);
    }
    t_cand = tt.ms();
    if (cands.empty()) {
        if (StageTimer::on()) {
            fprintf(stderr, "[seg] no-cand cand=%.2fms\n", t_cand);
        }
        *code = STATUS_SUCCESS;
        return boxes;
    }
    tt.start();
    const auto keep = nms_candidates(cands, iou_threshold);
    t_nms = tt.ms();
    boxes.reserve(keep.size());
    for (const int32_t ki : keep) {
        const Candidate &c = cands[ki];
        boxes.push_back({make_result(lb, c), c.anchor, c.x1, c.y1, c.x2, c.y2});
    }
    if (StageTimer::on()) {
        fprintf(stderr, "[seg] boxes=%zu cand=%.2fms nms=%.2fms\n", boxes.size(), t_cand, t_nms);
    }
    *code = STATUS_SUCCESS;
    return boxes;
}

void build_segment_masks(const YoloInferOutput *outs, const YOLOConfig &config,
                         const Letterbox &lb, std::vector<SegBox> &boxes,
                         const std::vector<int> &picks) {
    if (picks.empty()) {
        return;
    }
    if (!outs[0].data || !outs[1].data) {
        return;
    }
    const auto &s0 = config.output_shapes[0];
    const auto &s1 = config.output_shapes[1];
    const int64_t C = s0[1];
    const int64_t N = s0[2];
    const int64_t ph = s1[2];
    const int64_t pw = s1[3];
    constexpr int64_t kCoeffs = 32;
    const bool e2e = (s0[2] <= 64 && s0[1] <= 300);
    const float *data0 = static_cast<const float *>(outs[0].data);
    const float *proto = static_cast<const float *>(outs[1].data);  // [32, ph, pw]
    const int box = letterbox_size(config, lb);
    const float sx = static_cast<float>(pw) / static_cast<float>(box);
    const float sy = static_cast<float>(ph) / static_cast<float>(box);

    StageTimer tt;
    tt.start();
    std::vector<float> crop;   // 框内 proto 区域线性组合（未 sigmoid）
    std::vector<int> x0s;      // resize 每列查表
    std::vector<float> fxs;
    int done = 0;
    for (const int pi : picks) {
        if (pi < 0 || pi >= static_cast<int>(boxes.size())) {
            continue;
        }
        SegBox &b = boxes[pi];
        // 掩码系数（v8：最后 32 通道；e2e：行尾 32 个）
        std::array<float, kCoeffs> coeffs{};
        if (!e2e) {
            for (int64_t k = 0; k < kCoeffs; ++k) {
                coeffs[k] = data0[(C - kCoeffs + k) * N + b.anchor];
            }
        } else {
            const float *row = data0 + b.anchor * s0[2];
            for (int64_t k = 0; k < kCoeffs; ++k) {
                coeffs[k] = row[6 + k];
            }
        }
        // 裁到框（letterbox 坐标 → proto 坐标，与 ultralytics crop_mask 一致）
        const int x1p = std::clamp(static_cast<int>(std::floor(b.lx1 * sx)), 0, static_cast<int>(pw) - 1);
        const int y1p = std::clamp(static_cast<int>(std::floor(b.ly1 * sy)), 0, static_cast<int>(ph) - 1);
        const int x2p = std::clamp(static_cast<int>(std::ceil(b.lx2 * sx)), x1p + 1, static_cast<int>(pw));
        const int y2p = std::clamp(static_cast<int>(std::ceil(b.ly2 * sy)), y1p + 1, static_cast<int>(ph));
        const int cw = x2p - x1p;
        const int ch = y2p - y1p;
        // mask = Σ coeff_k * proto_k（未 sigmoid），只在框内区域组合
        crop.resize(static_cast<size_t>(cw) * ch);
        for (int y = 0; y < ch; ++y) {
            const float *prow = proto + (static_cast<size_t>(y1p + y) * pw + x1p);
            float *drow = crop.data() + static_cast<size_t>(y) * cw;
            for (int x = 0; x < cw; ++x) {
                float v = 0.0f;
                for (int64_t k = 0; k < kCoeffs; ++k) {
                    v += coeffs[k] * prow[k * ph * pw + x];
                }
                drow[x] = v;
            }
        }
        // 缩放到框尺寸（= 旧实现的框内区域），阈值恒等变换 sigmoid(v)>0.5 ⟺ v>0
        const int dw = b.r.width;
        const int dh = b.r.height;
        const float scale_x = static_cast<float>(cw) / static_cast<float>(dw);
        const float scale_y = static_cast<float>(ch) / static_cast<float>(dh);
        x0s.resize(dw);
        fxs.resize(dw);
        for (int x = 0; x < dw; ++x) {
            const float sp = (x + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(sp)), 0, cw - 1);
            x0s[x] = x0;
            fxs[x] = sp - std::floor(sp);
        }
        b.r.mask.resize(static_cast<size_t>(dw) * dh);
        uint8_t *mbase = b.r.mask.data();
        for (int y = 0; y < dh; ++y) {
            const float sp = (y + 0.5f) * scale_y - 0.5f;
            const int y0 = std::clamp(static_cast<int>(std::floor(sp)), 0, ch - 1);
            const int y1 = std::clamp(y0 + 1, 0, ch - 1);
            const float fy = sp - std::floor(sp);
            const float *s0r = crop.data() + static_cast<size_t>(y0) * cw;
            const float *s1r = crop.data() + static_cast<size_t>(y1) * cw;
            uint8_t *mrow = mbase + static_cast<size_t>(y) * dw;
            for (int x = 0; x < dw; ++x) {
                const int x0 = x0s[x];
                const int x1 = std::clamp(x0 + 1, 0, cw - 1);
                const float fx = fxs[x];
                const float v = (s0r[x0] * (1.0f - fx) + s0r[x1] * fx) * (1.0f - fy) +
                                (s1r[x0] * (1.0f - fx) + s1r[x1] * fx) * fy;
                mrow[x] = v > 0.0f ? 1 : 0;
            }
        }
        done++;
    }
    if (StageTimer::on()) {
        fprintf(stderr, "[segmask] picks=%zu done=%d %.2fms\n", picks.size(), done, tt.ms());
    }
}

// ------------------------------------------------------------------ //
// POSE：检测 + 关键点
// ------------------------------------------------------------------ //
std::vector<YOLOResult> decode_pose(const YoloInferOutput *outs, int out_count,
                                    const YOLOConfig &config, const Letterbox &lb,
                                    float min_score, float iou_threshold, int *code) {
    std::vector<YOLOResult> results;
    if (out_count < 1 || !outs[0].data || outs[0].elem_type != ONNX_FLOAT) {
        *code = STATUS_INFERENCE_FAILED;
        return results;
    }
    const auto &s0 = config.output_shapes[0];
    if (s0.size() < 3) {
        *code = STATUS_INFERENCE_FAILED;
        return results;
    }
    const int64_t C = s0[1];
    const int64_t N = s0[2];
    const float *data = static_cast<const float *>(outs[0].data);
    const int64_t nc = config.num_classes > 0 ? config.num_classes : 1;

    int64_t k = 0;         // 关键点数
    int64_t kpt_base = 0;  // 关键点通道起始
    // e2e 行布局 [1, N, 6+3k]（N≤300）；否则 v8 头 [1, C, N]（C = 4 + nc + 3k）
    const bool e2e = (s0[2] <= 64 && s0[1] <= 300);
    std::vector<Candidate> cands;
    if (e2e) {
        const int64_t W = s0[2];
        if (W <= 6 || (W - 6) % 3 != 0) {
            *code = STATUS_INFERENCE_FAILED;
            return results;
        }
        k = (W - 6) / 3;
        kpt_base = 6;
        decode_e2e_candidates(data, N, W, min_score, cands);
    } else {
        // v8 头 [1, C, N]：C = 4 + nc + 3k
        if (C <= 4 + nc || (C - 4 - nc) % 3 != 0) {
            *code = STATUS_INFERENCE_FAILED;
            return results;
        }
        k = (C - 4 - nc) / 3;
        kpt_base = 4 + nc;
        decode_v8_candidates(data, C, N, nc, min_score, cands);
    }
    if (k <= 0) {
        *code = STATUS_INFERENCE_FAILED;
        return results;
    }
    if (cands.empty()) {
        *code = STATUS_SUCCESS;
        return results;
    }
    const auto keep = nms_candidates(cands, iou_threshold);

    for (const int32_t ki : keep) {
        const Candidate &c = cands[ki];
        YOLOResult r = make_result(lb, c);
        r.keypoints.resize(static_cast<size_t>(k));
        for (int64_t i = 0; i < k; ++i) {
            float kx, ky, ks;
            if (!e2e) {
                kx = data[(kpt_base + 3 * i) * N + c.anchor];
                ky = data[(kpt_base + 3 * i + 1) * N + c.anchor];
                ks = sigmoidf(data[(kpt_base + 3 * i + 2) * N + c.anchor]);
            } else {
                const float *row = data + c.anchor * s0[2];
                kx = row[kpt_base + 3 * i];
                ky = row[kpt_base + 3 * i + 1];
                ks = sigmoidf(row[kpt_base + 3 * i + 2]);
            }
            r.keypoints[i] = {lb.map_back_x(kx), lb.map_back_y(ky), ks};
        }
        results.push_back(std::move(r));
    }
    *code = STATUS_SUCCESS;
    return results;
}

// ------------------------------------------------------------------ //
// OBB：旋转框（AABB + 角度）
// ------------------------------------------------------------------ //
std::vector<YOLOResult> decode_obb(const YoloInferOutput *outs, int out_count,
                                   const YOLOConfig &config, const Letterbox &lb,
                                   float min_score, float iou_threshold, int *code) {
    std::vector<YOLOResult> results;
    if (out_count < 1 || !outs[0].data || outs[0].elem_type != ONNX_FLOAT) {
        *code = STATUS_INFERENCE_FAILED;
        return results;
    }
    const auto &s0 = config.output_shapes[0];
    if (s0.size() < 3) {
        *code = STATUS_INFERENCE_FAILED;
        return results;
    }
    const int64_t C = s0[1];
    const int64_t N = s0[2];
    const float *data = static_cast<const float *>(outs[0].data);

    std::vector<Candidate> cands;
    std::vector<float> rcx, rcy, rw, rh, rang;
    const bool e2e = (s0[2] <= 64 && s0[1] <= 300);
    if (e2e) {
        // e2e 行 [1, N, 7]：行 = [cx, cy, w, h, conf, cls, angle]（angle 为弧度）
        if (s0[2] != 7) {
            *code = STATUS_INFERENCE_FAILED;
            return results;
        }
        for (int64_t n = 0; n < N; ++n) {
            const float *row = data + n * 7;
            if (row[4] < min_score) {
                continue;
            }
            const float x = row[0];
            const float y = row[1];
            const float w = row[2];
            const float h = row[3];
            cands.push_back({x - w * 0.5f, y - h * 0.5f, x + w * 0.5f, y + h * 0.5f,
                             row[4], static_cast<int>(row[5]), static_cast<int>(n)});
            rcx.push_back(x);
            rcy.push_back(y);
            rw.push_back(w);
            rh.push_back(h);
            rang.push_back(row[6]);
        }
    } else {
        // v8 头 [1, C, N]：C = 4 + nc + 1，布局 [x, y, w, h, class_probs..., angle]
        // 角度在最后一通道，且 ONNX 导出时已解码为弧度（(sigmoid-0.25)*π，见 ultralytics OBB head）
        const int64_t nc = C - 5;
        if (nc < 1) {
            *code = STATUS_INFERENCE_FAILED;
            return results;
        }
        std::vector<float> best(static_cast<size_t>(N));
        std::vector<int32_t> best_cls(static_cast<size_t>(N));
        const float *score_ptr = data + 4 * N;
        for (int64_t c = 4; c < 4 + nc; ++c, score_ptr += N) {
            for (int64_t n = 0; n < N; ++n) {
                const float s = score_ptr[n];
                if (s > best[n]) {
                    best[n] = s;
                    best_cls[n] = static_cast<int32_t>(c - 4);
                }
            }
        }
        for (int64_t n = 0; n < N; ++n) {
            if (best[n] < min_score) {
                continue;
            }
            const float x = data[n];
            const float y = data[N + n];
            const float w = data[2 * N + n];
            const float h = data[3 * N + n];
            const float angle = data[(C - 1) * N + n];  // 已解码弧度
            cands.push_back({x - w * 0.5f, y - h * 0.5f, x + w * 0.5f, y + h * 0.5f,
                             best[n], best_cls[n], static_cast<int>(n)});
            rcx.push_back(x);
            rcy.push_back(y);
            rw.push_back(w);
            rh.push_back(h);
            rang.push_back(angle);
        }
    }
    if (cands.empty()) {
        *code = STATUS_SUCCESS;
        return results;
    }

    // 旋转 IoU NMS
    std::vector<nms::Detection> dets;
    dets.reserve(cands.size());
    for (const auto &c : cands) {
        dets.emplace_back(c.x1, c.y1, c.x2, c.y2, c.score, c.class_id);
    }
    const auto keep = nms::nms_keep_rotated(dets, rcx, rcy, rw, rh, rang, iou_threshold);

    // 结果：旋转框角点 → AABB（原图整数坐标）+ 角度
    for (const int32_t ki : keep) {
        const Candidate &c = cands[ki];
        const float cx = rcx[ki];
        const float cy = rcy[ki];
        const float w = rw[ki];
        const float h = rh[ki];
        const float ang = rang[ki];
        const float ct = std::cos(ang);
        const float st = std::sin(ang);
        const float hw = w * 0.5f;
        const float hh = h * 0.5f;
        float min_x = 1e9f, min_y = 1e9f, max_x = -1e9f, max_y = -1e9f;
        const float local[4][2] = {{hw, hh}, {-hw, hh}, {-hw, -hh}, {hw, -hh}};
        for (const auto &p : local) {
            const float px = cx + p[0] * ct - p[1] * st;
            const float py = cy + p[0] * st + p[1] * ct;
            min_x = std::min(min_x, px);
            min_y = std::min(min_y, py);
            max_x = std::max(max_x, px);
            max_y = std::max(max_y, py);
        }
        const OrigBox ob = to_orig_box(lb, min_x, min_y, max_x, max_y);
        YOLOResult r(ob.x, ob.y, ob.w, ob.h, c.class_id, c.score);
        r.angle = ang;
        results.push_back(std::move(r));
    }
    *code = STATUS_SUCCESS;
    return results;
}

// ------------------------------------------------------------------ //
// 公开入口
// ------------------------------------------------------------------ //
// RKNN 优化图（9 输出 = 3 尺度 × box/score/sum）：DFL 解码在 CPU
// 每尺度: box [1,64,h,w] + score [1,nc,h,w]（已 sigmoid）+ sum [1,1,h,w]（快速过滤）
// ------------------------------------------------------------------ //
constexpr int64_t kDflLen = 16;  // v8 DFL 每边 bin 数

// DFL：每条边 dfl_len 个 bin 做 softmax 后取数学期望 → 边距离
void compute_dfl(const float *tensor, const int64_t dfl_len, float *box) {
    for (int b = 0; b < 4; ++b) {
        float exp_sum = 0;
        float acc = 0;
        const float *base = tensor + b * dfl_len;
        for (int64_t i = 0; i < dfl_len; ++i) {
            exp_sum += std::exp(base[i]);
        }
        for (int64_t i = 0; i < dfl_len; ++i) {
            acc += std::exp(base[i]) / exp_sum * static_cast<float>(i);
        }
        box[b] = acc;
    }
}

// 单尺度候选解码（移植自 rknn_model_zoo process_fp32；分数已 sigmoid，sum 做快速过滤）
void decode_rknnopt_scale(const float *box_t, const float *score_t, const float *sum_t,
                          const int64_t grid_h, const int64_t grid_w, const int64_t stride,
                          const int64_t dfl_len, const int64_t nc, const float min_score,
                          std::vector<Candidate> &out) {
    const int64_t grid_len = grid_h * grid_w;
    std::vector<float> dfl_in(static_cast<size_t>(dfl_len) * 4);
    for (int64_t i = 0; i < grid_h; ++i) {
        for (int64_t j = 0; j < grid_w; ++j) {
            const int64_t offset = i * grid_w + j;
            // sum ≥ 单类最大分数，先用它快速过滤，省掉 nc 通道扫描
            if (sum_t != nullptr && sum_t[offset] < min_score) {
                continue;
            }
            float max_score = 0;
            int64_t max_cls = -1;
            for (int64_t c = 0; c < nc; ++c) {
                const float sc = score_t[offset + c * grid_len];
                if (sc > max_score) {
                    max_score = sc;
                    max_cls = c;
                }
            }
            if (max_score < min_score) {
                continue;
            }
            float box[4];
            if (dfl_len == 1) {
                // yolo11 fork 优化图：DFL 已在图内完成，4 通道即 l/t/r/b 距离
                box[0] = box_t[offset];
                box[1] = box_t[grid_len + offset];
                box[2] = box_t[2 * grid_len + offset];
                box[3] = box_t[3 * grid_len + offset];
            } else {
                for (int64_t k = 0; k < dfl_len * 4; ++k) {
                    dfl_in[k] = box_t[offset + k * grid_len];
                }
                compute_dfl(dfl_in.data(), dfl_len, box);
            }
            const float x1 = (-box[0] + j + 0.5f) * static_cast<float>(stride);
            const float y1 = (-box[1] + i + 0.5f) * static_cast<float>(stride);
            const float x2 = (box[2] + j + 0.5f) * static_cast<float>(stride);
            const float y2 = (box[3] + i + 0.5f) * static_cast<float>(stride);
            out.push_back({x1, y1, x2, y2, max_score, static_cast<int>(max_cls), -1});
        }
    }
}

// int8 量化域阈值 → 量化值（四舍五入截断到 int8）
inline int8_t to_q8(const float v, const int32_t zp, const float scale) {
    const int q = static_cast<int>(std::lround(v / scale)) + zp;
    return static_cast<int8_t>(std::clamp(q, -128, 127));
}

// 单尺度候选解码 int8 版：通道优先布局 [C][h][w]（与 want_float 输出一致），
// sum/分数在量化域比较，仅对过筛候选反量化 4*dfl_len 个框值
void decode_rknnopt_scale_i8(const int8_t *box_t, const int32_t box_zp, const float box_scale,
                             const int8_t *score_t, const int32_t score_zp, const float score_scale,
                             const int8_t *sum_t, const int32_t sum_zp, const float sum_scale,
                             const int64_t grid_h, const int64_t grid_w, const int64_t stride,
                             const int64_t dfl_len, const int64_t nc, const float min_score,
                             std::vector<Candidate> &out) {
    const int8_t score_thres_q = to_q8(min_score, score_zp, score_scale);
    const int8_t sum_thres_q = to_q8(min_score, sum_zp, sum_scale);
    const int64_t grid_len = grid_h * grid_w;
    std::vector<float> dfl_in(static_cast<size_t>(dfl_len) * 4);
    for (int64_t i = 0; i < grid_h; ++i) {
        for (int64_t j = 0; j < grid_w; ++j) {
            const int64_t offset = i * grid_w + j;
            // sum ≥ 单类最大分数，快速过滤
            if (sum_t[offset] < sum_thres_q) {
                continue;
            }
            // 类别分数通道优先 [nc][h][w]
            const int8_t *sbase = score_t + offset;
            int max_q = score_thres_q;
            int64_t max_cls = -1;
            for (int64_t c = 0; c < nc; ++c) {
                const int8_t q = sbase[c * grid_len];
                if (q > max_q) {
                    max_q = q;
                    max_cls = c;
                }
            }
            if (max_cls < 0) {
                continue;
            }
            float box[4];
            if (dfl_len == 1) {
                // yolo11 fork 优化图：DFL 已在图内完成，4 通道即 l/t/r/b 距离（反量化后）
                box[0] = static_cast<float>(box_t[offset] - box_zp) * box_scale;
                box[1] = static_cast<float>(box_t[grid_len + offset] - box_zp) * box_scale;
                box[2] = static_cast<float>(box_t[2 * grid_len + offset] - box_zp) * box_scale;
                box[3] = static_cast<float>(box_t[3 * grid_len + offset] - box_zp) * box_scale;
            } else {
                for (int64_t k = 0; k < dfl_len * 4; ++k) {
                    dfl_in[k] = static_cast<float>(box_t[k * grid_len + offset] - box_zp) * box_scale;
                }
                compute_dfl(dfl_in.data(), dfl_len, box);
            }
            const float x1 = (-box[0] + j + 0.5f) * static_cast<float>(stride);
            const float y1 = (-box[1] + i + 0.5f) * static_cast<float>(stride);
            const float x2 = (box[2] + j + 0.5f) * static_cast<float>(stride);
            const float y2 = (box[3] + i + 0.5f) * static_cast<float>(stride);
            out.push_back({x1, y1, x2, y2, static_cast<float>(max_q - score_zp) * score_scale,
                           static_cast<int>(max_cls), -1});
        }
    }
}

// RKNN 优化图检测解码：9 输出按尺度循环 → 候选 → NMS（letterbox 坐标）
std::vector<nms::Detection> decode_rknnopt_detect(const YoloInferOutput *outs, const int out_count,
                                                  const YOLOConfig &config, const Letterbox &lb,
                                                  const float min_score, const float iou_threshold,
                                                  nms::NmsProcessor &nms_processor) {
    (void)lb;
    std::vector<nms::Detection> out;
    if (out_count < 9) {
        return out;
    }
    const int64_t model_in = config.input_shape.size() >= 4 ? config.input_shape[2]
                                                            : static_cast<int64_t>(lb.dst_height);
    const bool i8_mode = (outs[0].elem_type == 3);  // int8 直读（后端量化数据 + zp/scale）
    std::vector<Candidate> cands;
    for (int64_t b = 0; b < 3; ++b) {
        const auto &bs = config.output_shapes[static_cast<size_t>(b) * 3];  // box [1, 64, h, w]
        if (bs.size() < 4 || bs[1] % 4 != 0) {
            return out;
        }
        const int64_t grid_h = bs[2];
        const int64_t grid_w = bs[3];
        const int64_t dfl_len = bs[1] / 4;
        const int64_t nc = config.output_shapes[static_cast<size_t>(b) * 3 + 1][1];
        const int64_t stride = model_in / grid_h;
        if (i8_mode) {
            if (!outs[b * 3].data || !outs[b * 3 + 1].data || !outs[b * 3 + 2].data) {
                return out;
            }
            decode_rknnopt_scale_i8(
                static_cast<const int8_t *>(outs[b * 3].data), outs[b * 3].zp, outs[b * 3].scale,
                static_cast<const int8_t *>(outs[b * 3 + 1].data), outs[b * 3 + 1].zp,
                outs[b * 3 + 1].scale,
                static_cast<const int8_t *>(outs[b * 3 + 2].data), outs[b * 3 + 2].zp,
                outs[b * 3 + 2].scale,
                grid_h, grid_w, stride, dfl_len, nc, min_score, cands);
        } else {
            const float *box_t = static_cast<const float *>(outs[b * 3].data);
            const float *score_t = static_cast<const float *>(outs[b * 3 + 1].data);
            const float *sum_t = static_cast<const float *>(outs[b * 3 + 2].data);
            if (!box_t || !score_t || !sum_t) {
                return out;
            }
            decode_rknnopt_scale(box_t, score_t, sum_t, grid_h, grid_w, stride, dfl_len, nc,
                                 min_score, cands);
        }
    }
    const auto keep = nms_candidates(cands, iou_threshold);
    out.reserve(keep.size());
    for (const int32_t ki : keep) {
        const Candidate &c = cands[ki];
        out.emplace_back(c.x1, c.y1, c.x2, c.y2, c.score, c.class_id);
    }
    return out;
}

// ------------------------------------------------------------------ //

std::vector<nms::Detection> decode_detect(const YoloInferOutput *outs, int out_count,
                                          const YOLOConfig &config, const Letterbox &lb,
                                          float min_score, float iou_threshold,
                                          nms::NmsProcessor &nms_processor) {
    StageTimer tt;
    tt.start();
    if (out_count < 1 || outs[0].data == nullptr || config.output_shapes.empty()) {
        return {};
    }
    const auto &output_shapes = config.output_shapes;
    // RKNN 优化图 9 输出（3 尺度 × box/score/sum）：DFL 解码在 CPU
    if (output_shapes.size() == 9) {
        auto out = decode_rknnopt_detect(outs, out_count, config, lb, min_score, iou_threshold,
                                         nms_processor);
        if (StageTimer::on()) {
            fprintf(stderr, "[det-rknnopt] %.2fms dets=%zu\n", tt.ms(), out.size());
        }
        return out;
    }
    const auto &shape = output_shapes[0];
    std::vector<nms::Detection> out;
    if (shape.size() >= 3 && shape[2] == 6) {
        // 端到端输出 [1, N, 6]（模型内已做 NMS，免后处理）
        out = nms::no_nms(static_cast<const float *>(outs[0].data), shape, min_score);
    } else {
        // v8 风格头 [1, C, N]，需要 NMS
        out = nms_processor.process(static_cast<const float *>(outs[0].data), shape, min_score,
                                    iou_threshold);
    }
    if (StageTimer::on()) {
        fprintf(stderr, "[det] %.2fms dets=%zu\n", tt.ms(), out.size());
    }
    return out;
}

ClassifyResult decode_classify(const YoloInferOutput *outs, int out_count,
                               const YOLOConfig &config, int *code) {
    ClassifyResult r;
    if (out_count < 1 || !outs[0].data || outs[0].elem_type != ONNX_FLOAT || outs[0].rank < 2) {
        *code = STATUS_INFERENCE_FAILED;
        return r;
    }
    const int64_t n = outs[0].shape[1];
    if (n <= 0) {
        *code = STATUS_INFERENCE_FAILED;
        return r;
    }
    const float *logits = static_cast<const float *>(outs[0].data);
    std::vector<float> scores(static_cast<size_t>(n));
    // 部分导出的分类模型在图中已做 Softmax（输出为概率分布，和≈1）；
    // 检测到已归一化则直接用，否则在此做 softmax
    float sum0 = 0.0f;
    bool in_unit = true;
    for (int64_t i = 0; i < n; ++i) {
        const float v = logits[i];
        sum0 += v;
        if (v < 0.0f || v > 1.0f) {
            in_unit = false;
        }
    }
    if (in_unit && std::fabs(sum0 - 1.0f) < 1e-3f) {
        for (int64_t i = 0; i < n; ++i) {
            scores[i] = logits[i];
        }
    } else {
        // softmax（数值稳定）
        float maxv = logits[0];
        for (int64_t i = 1; i < n; ++i) {
            maxv = std::max(maxv, logits[i]);
        }
        double sum = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            scores[i] = std::exp(static_cast<double>(logits[i]) - maxv);
            sum += scores[i];
        }
        for (auto &s : scores) {
            s = static_cast<float>(s / sum);
        }
    }
    r.scores = std::move(scores);
    // 前 top_k（分数降序）
    const int k = std::max(1, std::min(config.top_k, static_cast<int>(n)));
    r.top_k = k;
    std::vector<int> idx(static_cast<size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
        return r.scores[a] > r.scores[b];
    });
    r.top_ids.resize(static_cast<size_t>(k));
    r.top_scores.resize(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
        r.top_ids[i] = idx[i];
        r.top_scores[i] = r.scores[idx[i]];
    }
    r.top_class = r.top_ids[0];
    r.top_score = r.top_scores[0];
    *code = STATUS_SUCCESS;
    return r;
}

DenseResult decode_dense(const YoloInferOutput *outs, int out_count,
                         const YOLOConfig &config, const Letterbox &letterbox,
                         TaskType task, int *code) {
    DenseResult r;
    if (out_count < 1 || !outs[0].data) {
        *code = STATUS_INFERENCE_FAILED;
        return r;
    }
    const auto &s0 = config.output_shapes[0];
    if (s0.size() < 3) {
        *code = STATUS_INFERENCE_FAILED;
        return r;
    }
    const int64_t oh = s0[s0.size() - 2];
    const int64_t ow = s0[s0.size() - 1];
    if (oh <= 0 || ow <= 0) {
        *code = STATUS_INFERENCE_FAILED;
        return r;
    }
    // 有效区（去掉 letterbox 填充，输入分辨率坐标）
    const int box = letterbox_size(config, letterbox);
    const int vx0 = std::clamp(static_cast<int>(std::floor(letterbox.get_scaled_x())), 0, box - 1);
    const int vy0 = std::clamp(static_cast<int>(std::floor(letterbox.get_scaled_y())), 0, box - 1);
    const int vw = std::min(letterbox.dst_width, box - vx0);
    const int vh = std::min(letterbox.dst_height, box - vy0);
    if (vw <= 0 || vh <= 0) {
        *code = STATUS_SUCCESS;
        return r;
    }
    r.width = letterbox.img_width;
    r.height = letterbox.img_height;
    if (task == TaskType::SEMSEG) {
        // 输出可为 int64（类别 id）或 uint8（常见语义分割导出）
        std::vector<uint8_t> raw_ids(static_cast<size_t>(oh) * ow);
        if (outs[0].elem_type == ONNX_INT64) {
            const int64_t *cls = static_cast<const int64_t *>(outs[0].data);
            for (int64_t i = 0; i < oh * ow; ++i) {
                raw_ids[i] = static_cast<uint8_t>(cls[i]);
            }
        } else if (outs[0].elem_type == ONNX_UINT8) {
            const uint8_t *cls = static_cast<const uint8_t *>(outs[0].data);
            std::copy(cls, cls + oh * ow, raw_ids.begin());
        } else {
            *code = STATUS_INFERENCE_FAILED;
            return r;
        }
        // 裁有效区（模型分辨率 → 输入分辨率按比例映射）
        std::vector<uint8_t> cropped(static_cast<size_t>(vw) * vh);
        for (int y = 0; y < vh; ++y) {
            const int my = std::min(static_cast<int>((vy0 + y) * oh / box), static_cast<int>(oh) - 1);
            for (int x = 0; x < vw; ++x) {
                const int mx = std::min(static_cast<int>((vx0 + x) * ow / box), static_cast<int>(ow) - 1);
                cropped[static_cast<size_t>(y) * vw + x] = raw_ids[static_cast<size_t>(my) * ow + mx];
            }
        }
        // 最近邻缩放到原图
        r.class_map.resize(static_cast<size_t>(r.width) * r.height);
        resize_nearest_u8(cropped.data(), vw, vh, r.class_map.data(), r.width, r.height);
    } else {  // DEPTH
        if (outs[0].elem_type != ONNX_FLOAT) {
            *code = STATUS_INFERENCE_FAILED;
            return r;
        }
        const float *depth = static_cast<const float *>(outs[0].data);
        std::vector<float> cropped(static_cast<size_t>(vw) * vh);
        float dmin = std::numeric_limits<float>::max();
        float dmax = -std::numeric_limits<float>::max();
        for (int y = 0; y < vh; ++y) {
            const int my = std::min(static_cast<int>((vy0 + y) * oh / box), static_cast<int>(oh) - 1);
            for (int x = 0; x < vw; ++x) {
                const int mx = std::min(static_cast<int>((vx0 + x) * ow / box), static_cast<int>(ow) - 1);
                const float v = depth[static_cast<size_t>(my) * ow + mx];
                cropped[static_cast<size_t>(y) * vw + x] = v;
                dmin = std::min(dmin, v);
                dmax = std::max(dmax, v);
            }
        }
        r.depth_min = dmin;
        r.depth_max = dmax;
        // 双线性缩放到原图
        r.depth_map.resize(static_cast<size_t>(r.width) * r.height);
        resize_bilinear_f32(cropped.data(), vw, vh, r.depth_map.data(), r.width, r.height);
    }
    *code = STATUS_SUCCESS;
    return r;
}

}  // namespace yolo

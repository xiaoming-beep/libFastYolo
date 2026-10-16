#include <fast_yolo/nms.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <limits>
#include <cstdint>

//计算值
void nms::DetectionBox::reset_value() const {
    this->classId = -1;
    this->score = -0xFFFF;
    if (buff.size() < 5) {
        return;
    }
    int offset = 4;
    for (int i = 0, size = static_cast<int>(buff.size()) - offset; i < size; i++) {
        if (buff[offset] > this->score) {
            this->classId = i;
            this->score = buff[offset];
        }
        offset++;
    }
}

//转化结果
nms::Detection nms::DetectionBox::to_detect() const {
    const float x = buff[0];
    const float y = buff[1];
    const float w = buff[2];
    const float h = buff[3];
    return Detection{
        x - w * 0.5f,
        y - h * 0.5f,
        x + w * 0.5f,
        y + h * 0.5f,
        score,
        classId
    };
}

nms::Detection nms::DetectionBox::no_nms() const {
    return Detection{
        buff[0],
        buff[1],
        buff[2],
        buff[3],
        buff[4],
        static_cast<int>(buff[5])
    };
}

namespace {
    /**
     * 计算iou相似度（与旧实现逐位一致）
     */
    float compute_iou(const float x1a, const float y1a, const float x2a, const float y2a,
                      const float x1b, const float y1b, const float x2b, const float y2b) {
        const float area1 = (x2a - x1a) * (y2a - y1a);
        const float area2 = (x2b - x1b) * (y2b - y1b);
        const float left = std::max(x1a, x1b);
        const float top = std::max(y1a, y1b);
        const float right = std::min(x2a, x2b);
        const float bottom = std::min(y2a, y2b);
        const float interArea = std::max(right - left, static_cast<float>(0)) * std::max(
                                    bottom - top, static_cast<float>(0));
        const float unionArea = area1 + area2 - interArea;
        if (unionArea < 1e-8f) {
            return 0.0f;
        }
        return interArea / unionArea;
    }
}

/**
 * NmsProcessor 实现：工作缓冲为实例成员，随对象生命周期存在，
 * 多线程/多句柄/嵌套调用均安全，且跨帧复用避免每帧堆分配。
 */
std::vector<nms::Detection> nms::NmsProcessor::process(const float *output_data,
                                                       const std::vector<int64_t> &output_shape,
                                                       float min_score,
                                                       float iou_threshold) {
    if (output_shape.size() < 3) {
        //数据验证失败
        return {};
    }
    const int64_t C = output_shape[1];
    const int64_t N = output_shape[2];
    if (C < 5 || N <= 0) {
        return {};
    }

    best_score_.resize(static_cast<size_t>(N));
    best_cls_.resize(static_cast<size_t>(N));
    std::fill(best_score_.begin(), best_score_.end(), -std::numeric_limits<float>::max());

    // Pass A：顺序扫描分类通道（ch4..C-1），每 anchor 记录最高分类分与类别
    const float *score_ptr = output_data + 4 * N;
    for (int64_t c = 4; c < C; ++c, score_ptr += N) {
        for (int64_t h = 0; h < N; ++h) {
            const float s = score_ptr[h];
            if (s > best_score_[h]) {
                best_score_[h] = s;
                best_cls_[h] = static_cast<int32_t>(c - 4);
            }
        }
    }

    // Pass B：收集过阈值候选（坐标通道 ch0..3 顺序流式读取，与旧实现 to_detect 一致）
    cand_.clear();
    for (int64_t h = 0; h < N; ++h) {
        if (best_score_[h] < min_score) {
            continue;
        }
        const float x = output_data[h];
        const float y = output_data[N + h];
        const float w = output_data[2 * N + h];
        const float hh = output_data[3 * N + h];
        cand_.push_back(Candidate{
            x - w * 0.5f,
            y - hh * 0.5f,
            x + w * 0.5f,
            y + hh * 0.5f,
            best_score_[h],
            best_cls_[h]
        });
    }
    if (cand_.empty()) {
        return {};
    }

    // 按 (class_id 升序, score 降序) 排序：类内降序保证 NMS 从最高分开始处理
    std::sort(cand_.begin(), cand_.end(),
              [](const Candidate &a, const Candidate &b) {
                  if (a.class_id != b.class_id) {
                      return a.class_id < b.class_id;
                  }
                  return a.score > b.score;
              });

    // 分段流式 NMS：与旧 iou_filter 语义等价（贪心保留最高分框，抑制与其 IoU 超阈值的框）
    std::vector<nms::Detection> results;
    results.reserve(cand_.size());
    keep_.reserve(32);
    const size_t total = cand_.size();
    size_t seg_start = 0;
    for (size_t i = 0; i <= total; ++i) {
        const bool seg_end = (i == total) || (cand_[i].class_id != cand_[seg_start].class_id);
        if (!seg_end) {
            continue;
        }
        keep_.clear();
        for (size_t j = seg_start; j < i; ++j) {
            const Candidate &cur = cand_[j];
            bool suppressed = false;
            for (const size_t k: keep_) {
                const Candidate &kept = cand_[k];
                if (compute_iou(cur.x1, cur.y1, cur.x2, cur.y2,
                                kept.x1, kept.y1, kept.x2, kept.y2) > iou_threshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                keep_.push_back(j);
                results.emplace_back(cur.x1, cur.y1, cur.x2, cur.y2, cur.score, cur.class_id);
            }
        }
        seg_start = i;
    }
    return results;
}

/**
 * 执行nms（兼容入口：每次自建处理器，无全局状态、任何调用场景安全；
 * 高频路径请复用 NmsProcessor 实例以消除每帧缓冲分配）
 */
std::vector<nms::Detection> nms::non_max_suppression(const float *output_data, const std::vector<int64_t> &output_shape,
                                                     float min_score,
                                                     float iou_threshold) {
    NmsProcessor processor;
    return processor.process(output_data, output_shape, min_score, iou_threshold);
}

std::vector<nms::Detection> nms::no_nms(const float *output_data, const std::vector<int64_t> &output_shape,
                                        float min_score) {
    if (output_shape.size() < 3) {
        //数据验证失败
        return {};
    }
    const int64_t height = output_shape[2];
    if (height != 6) {
        return {};
    }
    const int64_t width = output_shape[1];
    //结果存储
    std::vector<Detection> results;
    results.reserve(static_cast<size_t>(width));
    for (int64_t w = 0; w < width; w++) {
        const float *col = output_data + w * 6;
        if (col[4] < min_score) {
            continue;
        }
        results.emplace_back(col[0], col[1], col[2], col[3], col[4], static_cast<int>(col[5]));
    }
    return results;
}

namespace {
    /**
     * 按 (类别升序, 分数降序) 排序的索引
     */
    std::vector<size_t> sort_by_class_score(const std::vector<nms::Detection> &detections) {
        std::vector<size_t> idx(detections.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            idx[i] = i;
        }
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
            if (detections[a].class_id != detections[b].class_id) {
                return detections[a].class_id < detections[b].class_id;
            }
            return detections[a].score > detections[b].score;
        });
        return idx;
    }
}

std::vector<int32_t> nms::nms_keep(const std::vector<Detection> &detections, float iou_threshold) {
    std::vector<int32_t> keep;
    const size_t total = detections.size();
    if (total == 0) {
        return keep;
    }
    const auto idx = sort_by_class_score(detections);
    // 分段流式 NMS（与 NmsProcessor::process 语义一致）
    std::vector<int32_t> kept_in_seg;
    kept_in_seg.reserve(32);
    size_t seg_start = 0;
    for (size_t i = 0; i <= total; ++i) {
        const bool seg_end = (i == total) || (detections[idx[i]].class_id != detections[idx[seg_start]].class_id);
        if (!seg_end) {
            continue;
        }
        kept_in_seg.clear();
        for (size_t j = seg_start; j < i; ++j) {
            const Detection &cur = detections[idx[j]];
            bool suppressed = false;
            for (const int32_t k: kept_in_seg) {
                const Detection &kept = detections[k];
                if (compute_iou(cur.x1, cur.y1, cur.x2, cur.y2,
                                kept.x1, kept.y1, kept.x2, kept.y2) > iou_threshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                kept_in_seg.push_back(static_cast<int32_t>(idx[j]));
                keep.push_back(static_cast<int32_t>(idx[j]));
            }
        }
        seg_start = i;
    }
    return keep;
}

namespace {
    struct Point2 {
        float x;
        float y;
    };

    /**
     * 旋转框 4 角点（cx, cy 中心；w, h 尺寸；angle 弧度）
     */
    void rotated_corners(float cx, float cy, float w, float h, float angle, Point2 out[4]) {
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        const float hw = w * 0.5f;
        const float hh = h * 0.5f;
        // 局部坐标（顺时针旋转：x' = x*cos - y*sin, y' = x*sin + y*cos）
        const Point2 local[4] = {{hw, hh}, {-hw, hh}, {-hw, -hh}, {hw, -hh}};
        for (int i = 0; i < 4; ++i) {
            out[i] = {cx + local[i].x * c - local[i].y * s,
                      cy + local[i].x * s + local[i].y * c};
        }
    }

    float polygon_area(const std::vector<Point2> &poly) {
        if (poly.size() < 3) {
            return 0.0f;
        }
        float area = 0.0f;
        for (size_t i = 0; i < poly.size(); ++i) {
            const size_t j = (i + 1) % poly.size();
            area += poly[i].x * poly[j].y - poly[j].x * poly[i].y;
        }
        return std::fabs(area) * 0.5f;
    }

    // 点相对有向边 (a->b) 的位置：>0 在左侧（保留侧）
    float cross2(const Point2 &a, const Point2 &b, const Point2 &p) {
        return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
    }

    // 线段 ab 与边 clip1->clip2 的交点
    Point2 line_intersect(const Point2 &a, const Point2 &b, const Point2 &c1, const Point2 &c2) {
        const float dx = b.x - a.x;
        const float dy = b.y - a.y;
        const float cx = c2.x - c1.x;
        const float cy = c2.y - c1.y;
        const float denom = dx * cy - dy * cx;
        if (std::fabs(denom) < 1e-10f) {
            return b;
        }
        const float t = ((c1.x - a.x) * cy - (c1.y - a.y) * cx) / denom;
        return {a.x + t * dx, a.y + t * dy};
    }

    /**
     * Sutherland–Hodgman：用凸多边形 clip（逆时针顶点序）裁剪 subject
     */
    std::vector<Point2> clip_polygon(const std::vector<Point2> &subject, const Point2 clip[4]) {
        std::vector<Point2> output = subject;
        for (int e = 0; e < 4; ++e) {
            if (output.empty()) {
                return output;
            }
            const Point2 c1 = clip[e];
            const Point2 c2 = clip[(e + 1) % 4];
            std::vector<Point2> input;
            input.swap(output);
            for (size_t i = 0; i < input.size(); ++i) {
                const Point2 &cur = input[i];
                const Point2 &nxt = input[(i + 1) % input.size()];
                const bool cur_in = cross2(c1, c2, cur) >= 0;
                const bool nxt_in = cross2(c1, c2, nxt) >= 0;
                if (cur_in && nxt_in) {
                    output.push_back(nxt);
                } else if (cur_in && !nxt_in) {
                    output.push_back(line_intersect(cur, nxt, c1, c2));
                } else if (!cur_in && nxt_in) {
                    output.push_back(line_intersect(cur, nxt, c1, c2));
                    output.push_back(nxt);
                }
            }
        }
        return output;
    }

    float rotated_iou(const Point2 a[4], const Point2 b[4]) {
        const float area_a = polygon_area(std::vector<Point2>(a, a + 4));
        const float area_b = polygon_area(std::vector<Point2>(b, b + 4));
        if (area_a < 1e-8f || area_b < 1e-8f) {
            return 0.0f;
        }
        const std::vector<Point2> inter = clip_polygon(std::vector<Point2>(a, a + 4), b);
        const float area_inter = polygon_area(inter);
        const float area_union = area_a + area_b - area_inter;
        if (area_union < 1e-8f) {
            return 0.0f;
        }
        return area_inter / area_union;
    }
}

std::vector<int32_t> nms::nms_keep_rotated(const std::vector<Detection> &detections,
                                           const std::vector<float> &cx, const std::vector<float> &cy,
                                           const std::vector<float> &w, const std::vector<float> &h,
                                           const std::vector<float> &angle,
                                           float iou_threshold) {
    std::vector<int32_t> keep;
    const size_t total = detections.size();
    if (total == 0) {
        return keep;
    }
    // 预计算角点
    std::vector<std::array<Point2, 4>> corners(total);
    for (size_t i = 0; i < total; ++i) {
        rotated_corners(cx[i], cy[i], w[i], h[i], angle[i], corners[i].data());
    }
    const auto idx = sort_by_class_score(detections);
    std::vector<int32_t> kept_in_seg;
    kept_in_seg.reserve(32);
    size_t seg_start = 0;
    for (size_t i = 0; i <= total; ++i) {
        const bool seg_end = (i == total) || (detections[idx[i]].class_id != detections[idx[seg_start]].class_id);
        if (!seg_end) {
            continue;
        }
        kept_in_seg.clear();
        for (size_t j = seg_start; j < i; ++j) {
            const size_t cur = idx[j];
            bool suppressed = false;
            for (const int32_t k: kept_in_seg) {
                if (rotated_iou(corners[cur].data(), corners[k].data()) > iou_threshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                kept_in_seg.push_back(static_cast<int32_t>(cur));
                keep.push_back(static_cast<int32_t>(cur));
            }
        }
        seg_start = i;
    }
    return keep;
}

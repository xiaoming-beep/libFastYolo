/*
 * 后处理解码器（内部头文件，不对外安装）
 *
 * 按任务类型将后端原始输出解码为业务结果：
 *   - DETECT:   v8 头 [1, C, N]（NMS）或端到端 [1, N, 6]（免 NMS）
 *   - SEGMENT:  两阶段懒加载——① decode_segment_boxes 只解码+NMS 出框（每帧必跑，
 *               廉价）；② build_segment_masks 只对被选中的框生成 ROI 掩码（框内
 *               w*h 字节 0/1，按需触发，避免每帧全幅掩码的清零+写回）
 *   - POSE:     检测 + 关键点（原图坐标）
 *   - OBB:      旋转框（AABB + 角度）
 *   - CLASSIFY: softmax + top-k
 *   - SEMSEG:   [1, H, W] int64 类别图 → 原图分辨率 uint8
 *   - DEPTH:    [1, 1, H, W] float → 原图分辨率 float
 */
#ifndef LIBFASTYOLO_POSTPROCESS_H
#define LIBFASTYOLO_POSTPROCESS_H
#include <fast_yolo/yolo.h>
#include <fast_yolo/nms.h>

namespace yolo {

// DETECT 解码：v8 头 [1,C,N]（NMS）/ e2e [1,N,6]（免 NMS）/ RKNN 优化图 9 输出（DFL 解码+NMS）
std::vector<nms::Detection> decode_detect(const YoloInferOutput *outs, int out_count,
                                          const YOLOConfig &config, const Letterbox &lb,
                                          float min_score, float iou_threshold,
                                          nms::NmsProcessor &nms_processor);

// CLASSIFY 解码（输出 [1, N] logits → softmax → top_k）
ClassifyResult decode_classify(const YoloInferOutput *outs, int out_count,
                               const YOLOConfig &config, int *code);

// SEMSEG / DEPTH 解码（稠密输出 → 原图分辨率）
DenseResult decode_dense(const YoloInferOutput *outs, int out_count,
                         const YOLOConfig &config, const Letterbox &letterbox,
                         TaskType task, int *code);

// ------------------------------------------------------------------ //
// SEGMENT 两阶段（懒加载掩码）：
//   每帧只调 decode_segment_boxes 拿框（≈0.6ms，无掩码成本）；
//   当且仅当本帧确实有人要消费某框掩码时，把索引放进 picks 调
//   build_segment_masks，为这些框生成 ROI 掩码。
// ------------------------------------------------------------------ //

/** 一个 SEGMENT 保留框：结果框 + 取掩码系数所需载荷（anchor/letterbox 坐标） */
struct SegBox {
    YOLOResult r;               // 原图坐标框（mask 为空，由第二阶段填充）
    int anchor = -1;            // 候选下标（v8/e2e 取 32 通道系数用）
    float lx1 = 0, ly1 = 0;     // letterbox 坐标框（裁剪/缩放基准，与解码一致）
    float lx2 = 0, ly2 = 0;
};

/** 阶段①：候选解码 + NMS，只返回框（含 anchor/letterbox 坐标载荷） */
std::vector<SegBox> decode_segment_boxes(const YoloInferOutput *outs, int out_count,
                                         const YOLOConfig &config, const Letterbox &letterbox,
                                         float min_score, float iou_threshold, int *code);

/**
 * 阶段②：为 boxes[picks[i]] 生成 ROI 掩码（r.mask 长度 = r.width*r.height，
 * 0/1 二值，行跨距 r.width，局部坐标原点 = 框左上角）。与旧"全幅掩码"在框内
 * 区域逐位一致（框外旧值恒 0）。
 */
void build_segment_masks(const YoloInferOutput *outs, const YOLOConfig &config,
                         const Letterbox &letterbox, std::vector<SegBox> &boxes,
                         const std::vector<int> &picks);

// POSE 解码：检测 + 关键点（原图坐标）
std::vector<YOLOResult> decode_pose(const YoloInferOutput *outs, int out_count,
                                    const YOLOConfig &config, const Letterbox &letterbox,
                                    float min_score, float iou_threshold, int *code);

// OBB 解码：旋转框（AABB + 角度）
std::vector<YOLOResult> decode_obb(const YoloInferOutput *outs, int out_count,
                                   const YOLOConfig &config, const Letterbox &letterbox,
                                   float min_score, float iou_threshold, int *code);

}
#endif

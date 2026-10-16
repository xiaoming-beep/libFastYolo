#ifndef LIBFASTYOLO_FASTYOLO_C_H
#define LIBFASTYOLO_FASTYOLO_C_H

/*
 * libFastYolo Python SDK 纯 C 绑定层（供 ctypes 调用）
 *
 * 与 source/jni/yolo_jni.cpp 同模式：所有参数展开为标量/POD，
 * 不跨语言传递 C++ 对象（YOLOConfig/Letterbox 含 std::vector 等非 POD 成员）。
 * 不依赖任何 Python 头文件，同一份 DLL 兼容任意 Python 3.x。
 *
 * 错误码复用 yolo.h 的 STATUS_* 宏：
 *   0 成功；-1 模型未找到；-2 模型初始化失败；-3 未初始化；-4 推理失败；-5 任务不支持
 */

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(LIBFASTYOLO_EXPORTS)
#define FASTYOLO_C_API __declspec(dllexport)
#elif defined(_WIN32)
#define FASTYOLO_C_API __declspec(dllimport)
#else
#define FASTYOLO_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* 检测结果（坐标已反算回原图，整数像素） */
typedef struct FastyoloBox {
    int x;
    int y;
    int width;
    int height;
    int cls_id;
    float score;
    int track_id; /* -1 表示未跟踪 */
} FastyoloBox;

/* 任务类型（与 yolo::TaskType 值一致） */
typedef enum FastyoloTask {
    TASK_AUTO = 0,
    TASK_DETECT = 1,
    TASK_SEGMENT = 2,
    TASK_POSE = 3,
    TASK_OBB = 4,
    TASK_CLASSIFY = 5,
    TASK_SEMSEG = 6,
    TASK_DEPTH = 7
} FastyoloTask;

/* 关键点（原图坐标） */
typedef struct FastyoloKeypoint {
    float x;
    float y;
    float score;
} FastyoloKeypoint;

/*
 * 统一推理结果（fastyolo_predict / fastyolo_track 返回，fastyolo_free_result 释放）。
 * 按任务类型填充对应字段，其余为 NULL/0：
 *   DETECT:   boxes
 *   SEGMENT:  boxes + masks（ROI：每框掩码长度 = 对应框 width*height，0/1 二值，
 *            行跨距 = 框 width，局部原点 = 框左上角 (x,y)；按 boxes 顺序拼接，总长
 *            = Σ boxes[i].width*boxes[i].height。仅 fastyolo_predict(need_masks=1)
 *            生成；fastyolo_track 只出框——在线按需掩码请走 JNI track(budgets)）
 *   POSE:     boxes + keypoints（每框 keypoint_count 个）
 *   OBB:      boxes（AABB）+ angles（弧度，与 boxes 平行）
 *   CLASSIFY: class_ids/class_scores/top_k + all_class_scores/class_count
 *   SEMSEG:   class_map（map_w * map_h uint8 类别 id）
 *   DEPTH:    depth_map（map_w * map_h float32 原始深度）+ depth_min/depth_max
 * 各数组分别分配，fastyolo_free_result 统一释放。
 */
typedef struct FastyoloResult {
    int task;                 /* 实际任务（FastyoloTask） */
    FastyoloBox *boxes;       /* [box_count] */
    int box_count;
    FastyoloKeypoint *keypoints; /* POSE: [box_count * keypoint_count] */
    int keypoint_count;          /* 每个框的关键点数（非 POSE 为 0） */
    uint8_t *masks;           /* SEGMENT: ROI 掩码按框拼接（长度 = Σ 框 w*h） */
    float *angles;            /* OBB: [box_count] 弧度 */
    int *class_ids;           /* CLASSIFY: 前 top_k 类别 id（分数降序） */
    float *class_scores;      /* CLASSIFY: 前 top_k 分数 */
    int top_k;
    float *all_class_scores;  /* CLASSIFY: 全部分类分数（softmax 后，长度 class_count） */
    int class_count;
    uint8_t *class_map;       /* SEMSEG: [map_w * map_h] 类别 id */
    float *depth_map;         /* DEPTH: [map_w * map_h] 原始深度 */
    float depth_min;
    float depth_max;
    int map_w;
    int map_h;
} FastyoloResult;

/*
 * 创建推理引擎（YOLOConfig + YOLOInference）。
 * model: 模型字节流（ONNX 单文件等，OpenVINO 后端不支持 IR .xml+.bin）。
 * 其余参数对应 yolo::YOLOConfig 字段（device 见 yolo::InferDevice 枚举值，
 * task 见 FastyoloTask，TASK_AUTO 时按输出形状自动推断）。
 * 成功返回不透明句柄（fastyolo_destroy 释放），失败返回 NULL 并写 *code。
 */
FASTYOLO_C_API void *fastyolo_create(const uint8_t *model, size_t model_len,
                                     float min_score, float iou_threshold,
                                     int device, int num_threads, int gpu_id, int precision,
                                     const char *input_name, const char *output_name,
                                     int track_frame_rate, int track_buffer,
                                     float track_thresh, float high_thresh, float match_thresh,
                                     int task, int num_classes, int num_keypoints, int top_k,
                                     int *code);

/*
 * 初始化后端引擎，并把输入/首个输出 shape 回填到调用方数组。
 * in_shape/out_shape 容量至少 8；*in_rank/*out_rank 返回实际维度数。
 * 返回 STATUS_* 错误码（0 成功）。
 */
FASTYOLO_C_API int fastyolo_init(void *handle,
                                 int64_t *in_shape, int in_cap, int *in_rank,
                                 int64_t *out_shape, int out_cap, int *out_rank);

/*
 * 查询全部输出的形状与元素类型（多输出模型用）。
 * out_shapes 容量 = 各输出 rank 之和（不足截断）；out_ranks/out_dtypes 容量 = out_count_cap。
 * out_dtypes 为 ONNX 元素类型码（1=float32, 7=int64）。返回 STATUS_* 错误码。
 */
FASTYOLO_C_API int fastyolo_output_info(void *handle,
                                        int64_t *out_shapes, int out_cap,
                                        int *out_ranks, int *out_dtypes, int out_count_cap,
                                        int *out_count);

/*
 * 查询解析后的任务类型（FastyoloTask；AUTO 时 init 后返回按输出形状推断的结果）。
 */
FASTYOLO_C_API int fastyolo_task(void *handle);

/* 创建 Letterbox（box_size 目标边长，color 填充色 0~255，默认 114 灰；
 * center_crop 非 0 时为中心裁剪模式：按最短边缩放后取中心 box_size×box_size，
 * 分类任务用，不填充） */
FASTYOLO_C_API void *fastyolo_letterbox_create(int box_size, float color, int center_crop);

/* 按原图尺寸计算缩放与目标尺寸（与 C++ Letterbox::update 完全一致），回填 dst_* */
FASTYOLO_C_API void fastyolo_letterbox_update(void *lb, int width, int height,
                                              int *dst_width, int *dst_height);

/*
 * 填充像素（与 C++ Letterbox::fill_bgr 完全一致，零拷贝，读取后即可释放 bgr）。
 * bgr: dst_width*dst_height*3 字节 BGR uint8（调用方需先按 update 结果缩放好图像）。
 */
FASTYOLO_C_API void fastyolo_letterbox_fill_bgr(void *lb, const uint8_t *bgr);

/*
 * 读取 letterbox 像素缓冲（float planar RRR..GGG..BBB，长度 box_size*box_size*3）。
 * 仅在下一次 update/fill 前有效；主要用于测试与调试。
 */
FASTYOLO_C_API const float *fastyolo_letterbox_pixels(void *lb, int *len);

/* 推理（score_threshold/iou_threshold < 0 时用配置默认值；仅 DETECT 生效）。
 * need_masks：1 时 SEGMENT 对全部保留框生成 ROI 掩码（离线一次性消费），0 只出框。
 * 成功返回 FastyoloResult（fastyolo_free_result 释放）；失败返回 NULL 并写 *code。 */
FASTYOLO_C_API FastyoloResult *fastyolo_predict(void *handle, void *lb,
                                                float score_threshold, float iou_threshold,
                                                int need_masks,
                                                int *code);

/* 推理 + ByteTrack 跟踪（结果带 track_id；SEGMENT 只出框不生成掩码——
 * 在线按需掩码请走 JNI track(MaskBudget[])，见 fastyolo_jni） */
FASTYOLO_C_API FastyoloResult *fastyolo_track(void *handle, void *lb,
                                              float score_threshold, float iou_threshold,
                                              int *code);

FASTYOLO_C_API void fastyolo_free_result(FastyoloResult *result);
FASTYOLO_C_API void fastyolo_letterbox_destroy(void *lb);
FASTYOLO_C_API void fastyolo_destroy(void *handle);

#ifdef __cplusplus
}
#endif

#endif /* LIBFASTYOLO_FASTYOLO_C_H */

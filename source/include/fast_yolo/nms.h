#ifndef LIBFASTYOLO_NMS_H
#define LIBFASTYOLO_NMS_H
#include <cstdint>
#include <cstring>
#include <vector>

namespace nms {
    // 位置信息
    class Detection {
    public:
        //位置信息
        float x1;
        float y1;
        float x2;
        float y2;
        //图像坐标
        float score;
        int class_id;

        Detection(float x1, float y1, float x2, float y2, float score, int class_id)
            : x1(x1),
              y1(y1),
              x2(x2),
              y2(y2),
              score(score),
              class_id(class_id) {
        }
    };

    /**
    * 基础二维数据
    */
    class DetectionBox {
        mutable float score = -1;
        std::vector<float> buff;
        mutable int classId = -1;

    public:
        explicit DetectionBox(const size_t size) : buff(size) {
        }

        float &operator[](const size_t index) {
            return buff[index];
        }

        // const 版本：用于 const 对象的只读访问
        const float &operator[](const size_t index) const {
            return buff[index];
        }

        /**
         * 获取相似度
         */
        float get_score() const {
            if (classId == -1) {
                reset_value();
            }
            return score;
        }

        /**
         *
         * @return 获取分类id
         */
        int get_class_id() const {
            if (classId == -1) {
                reset_value();
            }
            return classId;
        }

        /**
         * 重置值
         */
        void reset_value() const;

        [[nodiscard]] Detection to_detect() const;

        [[nodiscard]] Detection no_nms() const;

        /**
         * 清理数据
         */
        void clear() {
            std::fill(buff.begin(), buff.end(), 0.0f);
        }

        void copy(const void *addr, int size) {
            memcpy(buff.data(), addr, size);
        }
    };

    /**
     * NMS 处理器：工作缓冲绑定在实例上，随对象生命周期存在。
     * 多线程/多句柄/嵌套调用均安全（每个句柄持有独立缓冲），
     * 且跨帧复用避免每帧堆分配。性能路径请使用本类（如 YOLOInference 内部）。
     */
    class NmsProcessor {
    public:
        std::vector<Detection> process(const float *output_data, const std::vector<int64_t> &output_shape,
                                       float min_score,
                                       float iou_threshold);

    private:
        struct Candidate {
            float x1;
            float y1;
            float x2;
            float y2;
            float score;
            int class_id;
        };
        std::vector<float> best_score_;   // 每 anchor 最高分类分
        std::vector<int32_t> best_cls_;   // 每 anchor 最高分类类别
        std::vector<Candidate> cand_;     // 过阈值候选（排序 + 流式 NMS）
        std::vector<size_t> keep_;        // 段内已保留框索引
    };

    /**
     * 执行nms
     */
    std::vector<Detection> non_max_suppression(const float *output_data, const std::vector<int64_t> &output_shape,
                                               float min_score,
                                               float iou_threshold);

    /**
     * 处理免nms
     */
    std::vector<Detection> no_nms(const float *output_data, const std::vector<int64_t> &output_shape, float min_score);

    /**
     * 通用 NMS：按 (类别, 分数降序) 贪心抑制，返回保留的下标（输入顺序）。
     * 用于需要保留检测附带数据（掩码系数/关键点/角度等）的解码路径。
     */
    std::vector<int32_t> nms_keep(const std::vector<Detection> &detections, float iou_threshold);

    /**
     * 旋转框 NMS：AABB 之外并行提供旋转参数（cx/cy/w/h/angle 弧度），
     * 用旋转 IoU（多边形裁剪）抑制，返回保留下标。用于 OBB 任务。
     */
    std::vector<int32_t> nms_keep_rotated(const std::vector<Detection> &detections,
                                          const std::vector<float> &cx, const std::vector<float> &cy,
                                          const std::vector<float> &w, const std::vector<float> &h,
                                          const std::vector<float> &angle,
                                          float iou_threshold);
}
#endif

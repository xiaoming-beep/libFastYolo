#ifndef LIBFASTYOLO_YOLO_H
#define LIBFASTYOLO_YOLO_H
#if defined(_WIN32) && defined(LIBFASTYOLO_EXPORTS)
#define FASTYOLO_API __declspec(dllexport)
#elif defined(_WIN32)
#define FASTYOLO_API __declspec(dllimport)
#else
#define FASTYOLO_API
#endif
#include <fast_yolo/nms.h>
#include <fast_yolo/img_util.h>
#include <string>
#include <vector>
#include <memory>
// 成功
#define STATUS_SUCCESS                0

// 错误码（负数表示错误）
#define STATUS_MODEL_NOT_FOUND       -1   // 模型未找到
#define STATUS_MODEL_INIT_FAILED     -2   // 模型初始化失败
#define STATUS_NOT_INITIALIZED       -3   // 尚未初始化
#define STATUS_INFERENCE_FAILED      -4   // 推理失败
#define STATUS_TASK_UNSUPPORTED      -5   // 当前任务不支持该操作（如对非 DETECT 任务调用 track()）

namespace yolo {
    //计算大小
    int mul_size(const std::vector<int64_t> &arr);

    /**
     * 任务类型
     */
    enum class TaskType {
        AUTO = 0,     // 根据输出形状自动推断
        DETECT = 1,   // 目标检测
        SEGMENT = 2,  // 实例分割
        POSE = 3,     // 姿态关键点
        OBB = 4,      // 旋转目标检测
        CLASSIFY = 5, // 图像分类
        SEMSEG = 6,   // 语义分割
        DEPTH = 7,    // 深度估计
    };

    /**
     * 关键点（原图坐标，由 letterbox 逆变换反算）
     */
    struct Keypoint {
        float x;
        float y;
        float score;
    };

    /**
     * 结果构建
     */
    class YOLOResult {
    public:
        //x
        const int x;
        //y
        const int y;
        //宽度
        const int width;
        //高度
        const int height;
        //分类id
        const int clsId;
        //相似度
        const float score;

        //跟踪id（-1 表示未经跟踪，由 track() 赋值）
        int track_id = -1;

        // 以下字段仅特定任务填充（detect 时空置）：

        // POSE：关键点（原图坐标），大小 = num_keypoints
        std::vector<Keypoint> keypoints;

        // SEGMENT：实例掩码 ROI——长度 = width*height，0/1 二值；行跨距 = width，
        // 局部坐标原点 = 框左上角 (x,y)（旧"整幅原图"语义已废弃：框外恒 0，无信息损失）
        std::vector<uint8_t> mask;

        // OBB：旋转角（弧度）
        float angle = 0.0f;

        YOLOResult(const int x, const int y, const int width, const int height, const int cls_id, const float score)
            : x(x),
              y(y),
              width(width),
              height(height),
              clsId(cls_id),
              score(score) {
        }
    };

    /**
     * 图像分类结果
     */
    class ClassifyResult {
    public:
        // 置信度最高的类别
        int top_class = -1;
        // 最高类别分数
        float top_score = 0.0f;
        // 返回的前 K 个类别数（= config->top_k）
        int top_k = 0;
        // 前 top_k 类别 id（按分数降序）
        std::vector<int> top_ids;
        // 前 top_k 类别分数
        std::vector<float> top_scores;
        // 全部分类分数（softmax 后，长度为类别数）
        std::vector<float> scores;
    };

    /**
     * 稠密输出结果（语义分割 / 深度估计共用，按任务填充其一）
     */
    class DenseResult {
    public:
        // 原图分辨率
        int width = 0;
        int height = 0;
        // SEMSEG：逐像素类别 id（uint8，类别值须 < 256）
        std::vector<uint8_t> class_map;
        // DEPTH：逐像素深度（模型原始 float 值）
        std::vector<float> depth_map;
        // DEPTH：深度值域（用于归一化显示）
        float depth_min = 0.0f;
        float depth_max = 0.0f;
    };

    /**
     * yolo图片处理
     */
    class Letterbox {
        //边框大小
        const int box_size;
        //默认颜色
        const float def_color;
        //缩放比率
        float scale = 0;
        //缩放后的x
        float scaled_x = 0;
        //缩放后的y
        float scaled_y = 0;
        //中心裁剪模式（分类任务用：按最短边缩放后取中心 box×box，不填充）
        const bool center_crop_ = false;

    public:
        //像素
        std::vector<float> pixels;
        //目标图像宽高
        int dst_width = 0;
        int dst_height = 0;
        //原始图像宽度
        int img_width = 0;
        //原始图像高度
        int img_height = 0;

        explicit Letterbox(const int box_size = 640, const float color = 114, const bool center_crop = false)
            : box_size(box_size),
              def_color(color * INV_255),
              center_crop_(center_crop),
              pixels(box_size * box_size * 3) {
        }

        [[nodiscard]] bool center_crop() const {
            return center_crop_;
        }

        /**
         * 更新数据内容
         */
        void update(int width, int height);

        /**
         * 计算出yolo结果
         */
        [[nodiscard]] YOLOResult complete_result(const nms::Detection &detection) const;

        //填充uint8
        void fill_bgr(const uint8_t *data);

        // 缩放比率（letterbox 逆变换用）
        [[nodiscard]] float get_scale() const { return scale; }

        // 缩放后 x 偏移
        [[nodiscard]] float get_scaled_x() const { return scaled_x; }

        // 缩放后 y 偏移
        [[nodiscard]] float get_scaled_y() const { return scaled_y; }

        // 将 letterbox 坐标映射回原图坐标（小数）
        [[nodiscard]] float map_back_x(const float x) const { return (x - scaled_x) / scale; }

        [[nodiscard]] float map_back_y(const float y) const { return (y - scaled_y) / scale; }
    };

    /**
     *设备类
     */
    enum class InferDevice {
        CPU = 1,
        NPU = 2,
        VULKAN = 3,
        OPENCL = 4,
        //cuda
        CUDA = 6,
        //自动识别
        AUTO = 0,
        //全部
        ALL = 99
    };

    /**
     * 数据类型
     */
    enum class DataType {
        FLOAT32,
        FLOAT16,
        INT8,
        UINT8,
    };

    /**
     * 配置
     */
    class YOLOConfig {
    public:
        //模型路径
        const std::vector<char> model_data;
        //最低相似度
        const float min_score;
        //iou 值
        const float iou_threshold;
        //数据类型
        DataType data_type = DataType::FLOAT32;
        //推理设备
        InferDevice device = InferDevice::CPU;
        // CPU 线程数（GPU/NPU 可忽略）
        int num_threads = 1;
        // 多 GPU 时指定 ID
        int gpu_id = 0;
        //高精度
        int precision = 32;
        // 任务类型（AUTO 时 init 后按输出形状推断，见 yolo.cpp resolve_task）
        TaskType task = TaskType::AUTO;
        // POSE 等需要类别数（0 = 自动推断）
        int num_classes = 0;
        // POSE 关键点数（默认 COCO 17 点）
        int num_keypoints = 17;
        // CLASSIFY 返回前 K 个类别
        int top_k = 5;
        std::string input_name = "images";
        // 输出节点名（空 = 使用模型全部输出；非空 = 仅取该输出）
        std::string output_name = "";
        //输入形状
        std::vector<int64_t> input_shape;
        //输出形状（全部输出，init 回填）
        std::vector<std::vector<int64_t>> output_shapes;
        //输出元素类型（与 output_shapes 平行：ONNX 元素类型码 1=float32, 7=int64, init 回填）
        std::vector<int> output_dtypes;
        //首个输出形状（= output_shapes[0]，兼容便利字段）
        std::vector<int64_t> output_shape;

        explicit YOLOConfig(const std::vector<char> &model_data, const float min_score = 0.25,
                            const float iou_threshold = 0.45)
            : model_data(model_data),
              min_score(min_score),
              iou_threshold(iou_threshold) {
        }

        // ByteTrack 跟踪器参数（仅 track() 时使用）

        // 视频帧率，与 track_buffer 一起决定跟踪目标丢失后最多存活多少秒，建议 30
        int track_frame_rate = 30;

        // 丢失缓冲帧数，目标被遮挡/漏检时保留的帧数，超过后丢弃，建议 30（30fps 下 = 1s）
        int track_buffer = 30;

        // 低阈值，用于 ByteTrack 第二轮 IoU 关联，弱检测（≥此值但 < high_thresh）在此轮匹配，找回被遮挡目标，建议 0.4-0.5
        float track_thresh = 0.5f;

        // 高阈值，用于 ByteTrack 第一轮关联，高置信度检测（≥此值）优先匹配已有 track，建议 0.5-0.6
        float high_thresh = 0.6f;

        // IoU 匹配阈值，检测框与预测框的 IoU ≥ 此值才算匹配成功，越高越严格越不易串 ID，建议 0.7-0.8
        float match_thresh = 0.8f;
    };

    /**
     * 后端推理输出描述（纯 C 结构，ABI 契约）
     * data 由后端持有，仅在下次 yolo_infer_predict 调用前有效
     */
    struct YoloInferOutput {
        const void *data = nullptr;   // 输出数据指针（按 elem_type 解释）
        int64_t size = 0;             // 元素个数
        int elem_type = 1;            // ONNX 元素类型码：1=float32, 3=int8, 7=int64
        int64_t shape[8] = {0};       // 输出形状（rank 有效）
        int rank = 0;                 // 维度数
        // int8 输出的反量化参数：float = (q - zp) * scale（仅 elem_type==3 时有效）
        int32_t zp = 0;
        float scale = 1.0f;
    };

    /**
     * 掩码懒加载预算：调用方已为某 track 保留的最佳抠图面积（外接矩形面积）。
     * SEGMENT 跟踪时，仅当"表中无该 track（新船/尚无 best）"或"本帧框面积 >
     * bestArea"才为该框生成 ROI 掩码；平台期/远离期帧因此自动零掩码。
     */
    struct MaskBudget {
        int trackId = -1;
        float bestArea = 0.0f;
    };

    /**
     * yolo 推理
     */
    class YOLOInference {
    public:
        //输出大小
        int input_size = 0;
        //输出大小
        int output_size = 0;
        bool nms = false;

        explicit YOLOInference(const std::shared_ptr<YOLOConfig> &config)
            : config_(config) {
        }

        ~YOLOInference() {
            destroy();
        };

    /**
     * 推理（返回检测结果，track_id = -1）
     * 支持 DETECT / SEGMENT / POSE / OBB（按 config->task 分发）
     * need_masks=true 时 SEGMENT 对全部保留框生成 ROI 掩码（一次性消费场景，
     * 如离线抠图工具）；默认 false 只出框（懒加载：掩码按需另行请求）
     */
    std::vector<YOLOResult> predict(const Letterbox &letterbox, int *code,
                                     float score_threshold = -1.0f,
                                     float iou_threshold = -1.0f,
                                     bool need_masks = false) const;

        /**
         * 图像分类（config->task = CLASSIFY 或 AUTO）
         */
        ClassifyResult classify(const Letterbox &letterbox, int *code) const;

        /**
         * 语义分割（config->task = SEMSEG 或 AUTO），返回原图分辨率类别图
         */
        DenseResult semseg(const Letterbox &letterbox, int *code) const;

        /**
         * 深度估计（config->task = DEPTH 或 AUTO），返回原图分辨率 float 深度图
         */
        DenseResult depth(const Letterbox &letterbox, int *code) const;

        /**
         * 跟踪推理（返回带 track_id 的检测结果）
         * 仅支持 DETECT / SEGMENT / POSE / OBB（按 config->task 分发）
         * 内部先 predict，再执行 ByteTrack 多目标跟踪
         * SEGMENT 掩码懒加载：budgets 为 掩码预算表（调用方已保留 best 抠图面积，
         * 见 MaskBudget）；nullptr/空 → 整帧只出框不做掩码；表驱动按需生成
         */
        std::vector<YOLOResult> track(const Letterbox &letterbox, int *code,
                                       float score_threshold = -1.0f,
                                       float iou_threshold = -1.0f,
                                       const MaskBudget *budgets = nullptr,
                                       int budget_count = 0);

        /**
         * 初始化
         */
        int init();

        /**
         * 销毁内容
         */
        void destroy();

        /**
         * 解析任务类型（config->task == AUTO 时按输出形状推断，init 后调用有效）
         */
        TaskType resolve_task() const;

        /**
         * 模型输入形状（init 后由后端回填，如 {1, 3, 640, 640}）
         */
        [[nodiscard]] const std::vector<int64_t> &get_input_shape() const {
            return config_->input_shape;
        }

    private:
        // 保存配置指针
        const std::shared_ptr<YOLOConfig> config_;
        //是否初始化
        bool initialized_ = false;
        //内容指针
        void *context_ = nullptr;

        // NMS 工作缓冲（实例绑定：多线程/多句柄/嵌套调用均安全，且跨帧复用）
        mutable nms::NmsProcessor nms_processor_;

        // ByteTrack 跟踪器（由 track() 按需创建）
        void *tracker_ = nullptr;

        // 执行后端推理，填充输出数组（后端持有，下一次 predict 前有效），返回输出数
        int infer_outputs(const Letterbox &letterbox, YoloInferOutput *outs, int max_outs, int *code) const;
    };

    inline int mul_size(const std::vector<int64_t> &arr) {
        int size = 1;
        for (const auto i: arr) {
            size *= i;
        }
        return size;
    }

    extern "C" {
    void *yolo_infer_create(const YOLOConfig *config);

    void yolo_infer_init(void *inference, YOLOConfig *config, int *code);

    /**
     * 执行推理。核心层传入 outs[max_outs]，后端填充全部输出（超出 max_outs 截断）。
     * 返回实际输出数（失败返回 0 并写 *code）。
     */
    int yolo_infer_predict(void *inference,
                           const Letterbox *letterbox, YoloInferOutput *outs, int max_outs, int *code);

    void yolo_infer_destroy(void *inference);
    }
}
#endif

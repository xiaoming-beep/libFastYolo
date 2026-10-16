package com.fastyolo;

import lombok.Getter;

/**
 * YOLO 推理引擎——JNI 封装 C++ 侧 {@code yolo::YOLOInference}.
 *
 * <p>持有 C++ YOLOInference 对象的 native 指针（long nativeHandle）.
 * 实现了 AutoCloseable，推荐使用 try-with-resources:
 *
 * <pre>{@code
 * YoloConfig config = YoloConfig.builder()
 *     .modelData(Files.readAllBytes(Path.of("model.onnx")))
 *     .device(InferDevice.CPU)
 *     .numThreads(4)
 *     .build();
 *
 * try (YoloInference yolo = new YoloInference(config)) {
 *     yolo.init();
 *     Letterbox lb = new Letterbox();   // boxSize 需与模型输入一致，可用 getInputShape() 获取
 *     lb.update(imgWidth, imgHeight);
 *     // ... Java 侧缩放图像，获取 BGR 字节 ...
 *     lb.fillBGR(bgrBytes);
 *
 *     // 仅检测（trackId = -1）；SEGMENT/POSE/OBB 结果附带 mask/keypoints/angle
 *     YoloResult[] results = yolo.predict(lb);
 *
 *     // 检测 + ByteTrack 多目标跟踪（trackId >= 0，仅 DETECT 任务）
 *     YoloResult[] tracked = yolo.track(lb);
 *
 *     // 稠密任务（CLASSIFY / SEMSEG / DEPTH）
 *     ClassifyResult cls = yolo.classify(lb);
 *     SemSegResult sem = yolo.semseg(lb);
 *     DepthResult dep = yolo.depth(lb);
 * }
 * }</pre>
 *
 * <p>线程安全: 每个实例不线程安全（底层 C++ 引擎持有可变内部缓冲区
 * 和 ByteTrack 跟踪器状态）。多线程使用需各自创建实例或外挂锁.
 */
public class YoloInference implements AutoCloseable {

    static {
        System.loadLibrary("FastYolo");
    }

    /** C++ 侧 yolo::YOLOInference* 指针，0 表示已释放 */
    private long nativeHandle;
    /** 是否已关闭 */
    private volatile boolean closed;

    /** 是否已完成初始化 */
    @Getter
    private boolean initialized;

    /** 模型输入形状（init 后可用，如 [1, 3, 640, 640]） */
    @Getter
    private int[] inputShape = new int[0];

    /**
     * 创建推理引擎实例.
     * 构造时在 C++ 堆上创建 YOLOConfig + YOLOInference，
     * 但尚未初始化后端引擎（需调用 init()）.
     *
     * @param config 推理配置，modelData 会做防御性拷贝
     * @throws YoloException 如果 native 对象创建失败
     */
    public YoloInference(YoloConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("YoloConfig 不能为 null");
        }
        if (config.getModelData() == null || config.getModelData().length == 0) {
            throw new IllegalArgumentException("模型数据不能为空");
        }

        byte[] modelCopy = config.getModelData();

        this.nativeHandle = nativeCreate(
                modelCopy,
                config.getMinScore(),
                config.getIouThreshold(),
                config.getDevice().getNativeValue(),
                config.getGpuId(),
                config.getNumThreads(),
                config.getPrecision(),
                config.getInputName(),
                config.getOutputName(),
                config.getTrackFrameRate(),
                config.getTrackBuffer(),
                config.getTrackThresh(),
                config.getHighThresh(),
                config.getMatchThresh(),
                config.getTask().getNativeValue(),
                config.getNumClasses(),
                config.getNumKeypoints(),
                config.getTopK()
        );

        if (this.nativeHandle == 0) {
            throw new YoloException(YoloException.STATUS_MODEL_INIT_FAILED,
                    "Failed to create YOLO inference native instance");
        }
    }

    /**
     * 初始化后端推理引擎（ONNX / TensorRT / MNN）.
     * 幂等操作——重复调用不会重新初始化. 成功后可用 {@link #getInputShape()} 获取模型输入尺寸.
     *
     * @throws YoloException 初始化失败时抛出
     */
    public void init() {
        ensureOpen();
        int code = nativeInit(this.nativeHandle);
        if (code != YoloException.STATUS_SUCCESS) {
            throw YoloException.fromCode(code);
        }
        this.initialized = true;
        this.inputShape = nativeInputShape(this.nativeHandle);
    }

    /**
     * 解析后的任务类型（AUTO 时 init 后为按输出形状推断的结果）.
     *
     * @return TaskType，未初始化时返回 AUTO
     */
    public TaskType resolvedTask() {
        ensureOpen();
        return TaskType.fromNativeValue(nativeTask(this.nativeHandle));
    }

    // ==================== 检测（DETECT/SEGMENT/POSE/OBB） ====================

    /**
     * 执行目标检测（使用 config 中的默认阈值）.
     * 懒加载语义：SEGMENT 只返回检测框，不带掩码（mask == null）.
     * POSE/OBB 的附带数据（keypoints/angle）由结果对象携带.
     *
     * @param letterbox 已完成 fillBGR 的 Letterbox 实例
     * @return 检测结果数组（trackId = -1），可能为空
     */
    public YoloResult[] predict(Letterbox letterbox) {
        return predict(letterbox, -1.0f, -1.0f, false);
    }

    /**
     * 执行目标检测（覆盖 config 中的阈值）.
     * 懒加载语义：SEGMENT 只返回检测框，不带掩码（mask == null）.
     *
     * @param letterbox   已完成 fillBGR 的 Letterbox 实例
     * @param scoreThresh 置信度阈值，&lt;0 时使用 config 默认值
     * @param iouThresh   IoU 阈值，&lt;0 时使用 config 默认值
     * @return 检测结果数组，坐标已映射回原图空间
     */
    public YoloResult[] predict(Letterbox letterbox, float scoreThresh, float iouThresh) {
        return predict(letterbox, scoreThresh, iouThresh, false);
    }

    /**
     * 执行目标检测并显式请求 SEGMENT 掩码（一次性消费场景，如离线抠图）.
     *
     * @param needMask true 时 SEGMENT 对全部保留框生成 ROI 掩码（长度 =
     *                 width*height，0/1；局部原点 = 框左上角）；false 只出框
     */
    public YoloResult[] predict(Letterbox letterbox, float scoreThresh, float iouThresh,
                                boolean needMask) {
        ensureOpen();
        if (letterbox == null || letterbox.isClosed()) {
            throw new IllegalArgumentException("Letterbox 不能为 null 或已关闭");
        }
        return nativePredictMask(this.nativeHandle, letterbox.getNativeHandle(),
                scoreThresh, iouThresh, needMask);
    }

    // ==================== 稠密任务（CLASSIFY/SEMSEG/DEPTH） ====================

    /**
     * 图像分类（仅 CLASSIFY 任务；其他任务抛 {@link YoloException#STATUS_TASK_UNSUPPORTED}）.
     * 分类任务建议使用 {@code new Letterbox(boxSize, 114, true)} 中心裁剪模式预处理.
     *
     * @param letterbox 已完成 fillBGR 的 Letterbox 实例
     * @return 分类结果
     */
    public ClassifyResult classify(Letterbox letterbox) {
        ensureOpen();
        if (letterbox == null || letterbox.isClosed()) {
            throw new IllegalArgumentException("Letterbox 不能为 null 或已关闭");
        }
        return nativeClassify(this.nativeHandle, letterbox.getNativeHandle());
    }

    /**
     * 语义分割（仅 SEMSEG 任务），返回原图分辨率逐像素类别图.
     */
    public SemSegResult semseg(Letterbox letterbox) {
        ensureOpen();
        if (letterbox == null || letterbox.isClosed()) {
            throw new IllegalArgumentException("Letterbox 不能为 null 或已关闭");
        }
        return nativeSemSeg(this.nativeHandle, letterbox.getNativeHandle());
    }

    /**
     * 深度估计（仅 DEPTH 任务），返回原图分辨率 float32 深度图（模型原始值）.
     */
    public DepthResult depth(Letterbox letterbox) {
        ensureOpen();
        if (letterbox == null || letterbox.isClosed()) {
            throw new IllegalArgumentException("Letterbox 不能为 null 或已关闭");
        }
        return nativeDepth(this.nativeHandle, letterbox.getNativeHandle());
    }

    // ==================== 跟踪 ====================

    /**
     * 执行检测 + ByteTrack 多目标跟踪（使用 config 中的默认阈值）.
     * 懒加载语义：SEGMENT 只出框不带掩码（mask == null）.
     * 首次调用时会惰性创建 ByteTrack 跟踪器.
     *
     * @param letterbox 已完成 fillBGR 的 Letterbox 实例
     * @return 检测结果数组（trackId &gt;= 0）
     */
    public YoloResult[] track(Letterbox letterbox) {
        return track(letterbox, -1.0f, -1.0f, null, null);
    }

    /**
     * 执行检测 + ByteTrack 多目标跟踪（覆盖 config 中的阈值）.
     * 懒加载语义：SEGMENT 只出框不带掩码（mask == null）.
     *
     * @param letterbox   已完成 fillBGR 的 Letterbox 实例
     * @param scoreThresh 置信度阈值，&lt;0 时使用 config 默认值
     * @param iouThresh   IoU 阈值，&lt;0 时使用 config 默认值
     * @return 跟踪结果数组，trackId 为 ByteTrack 分配的稳定 ID
     */
    public YoloResult[] track(Letterbox letterbox, float scoreThresh, float iouThresh) {
        return track(letterbox, scoreThresh, iouThresh, null, null);
    }

    /**
     * 预算表驱动的跟踪（SEGMENT 掩码懒加载，推荐在线使用）.
     * 掩码只对"预算表中无该 track（新船/尚无 best）或 本帧框面积 &gt; bestArea"
     * 的跟踪框生成（重叠仲裁框一并生成）；其余帧/框零掩码成本.
     *
     * @param budgetTrackIds 第 i 项 = trackId（可传 null 表示整帧只出框）
     * @param budgetAreas    第 i 项 = 调用方已为该 track 保留的 best 抠图面积
     *                       （替换 best 后必须同步更新；track 结束时移除对应项）
     */
    public YoloResult[] track(Letterbox letterbox, float scoreThresh, float iouThresh,
                              int[] budgetTrackIds, float[] budgetAreas) {
        ensureOpen();
        if (letterbox == null || letterbox.isClosed()) {
            throw new IllegalArgumentException("Letterbox 不能为 null 或已关闭");
        }
        return nativeTrackMask(this.nativeHandle, letterbox.getNativeHandle(),
                scoreThresh, iouThresh, budgetTrackIds, budgetAreas);
    }

    /**
     * 释放 native 资源（后端引擎 + ByteTrack 跟踪器）.
     * 幂等——重复调用不会出错.
     */
    @Override
    public void close() {
        if (!closed) {
            closed = true;
            initialized = false;
            long h = nativeHandle;
            nativeHandle = 0;
            if (h != 0) {
                nativeDestroy(h);
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("YoloInference 已关闭");
        }
    }

    // ==================== JNI native 方法 ====================

    /**
     * C++ 侧: 构造 yolo::YOLOConfig + yolo::YOLOInference，返回指针.
     * 返回 0 表示构造失败.
     */
    private static native long nativeCreate(
            byte[] modelData,
            float minScore,
            float iouThreshold,
            int device,
            int gpuId,
            int numThreads,
            int precision,
            String inputName,
            String outputName,
            int trackFrameRate,
            int trackBuffer,
            float trackThresh,
            float highThresh,
            float matchThresh,
            int task,
            int numClasses,
            int numKeypoints,
            int topK
    );

    /** C++ 侧: 调用 YOLOInference::init()，返回状态码 */
    private native int nativeInit(long handle);

    /** C++ 侧: 返回模型输入形状（int[]，init 后有效） */
    private native int[] nativeInputShape(long handle);

    /** C++ 侧: 返回解析后的任务类型 native 值 */
    private native int nativeTask(long handle);

    /** C++ 侧: 调用 YOLOInference::predict()（懒加载：只出框），构建并返回 YoloResult[] */
    private native YoloResult[] nativePredict(
            long handle,
            long letterboxHandle,
            float scoreThresh,
            float iouThresh
    );

    /** C++ 侧: 调用 YOLOInference::predict(need_masks)，显式请求掩码 */
    private native YoloResult[] nativePredictMask(
            long handle,
            long letterboxHandle,
            float scoreThresh,
            float iouThresh,
            boolean needMask
    );

    /** C++ 侧: 调用 YOLOInference::classify()，构建 ClassifyResult */
    private native ClassifyResult nativeClassify(long handle, long letterboxHandle);

    /** C++ 侧: 调用 YOLOInference::semseg()，构建 SemSegResult */
    private native SemSegResult nativeSemSeg(long handle, long letterboxHandle);

    /** C++ 侧: 调用 YOLOInference::depth()，构建 DepthResult */
    private native DepthResult nativeDepth(long handle, long letterboxHandle);

    /** C++ 侧: 调用 YOLOInference::track()（懒加载：只出框），构建并返回 YoloResult[] */
    private native YoloResult[] nativeTrack(
            long handle,
            long letterboxHandle,
            float scoreThresh,
            float iouThresh
    );

    /** C++ 侧: 调用 YOLOInference::track(MaskBudget[])，预算表驱动掩码 */
    private native YoloResult[] nativeTrackMask(
            long handle,
            long letterboxHandle,
            float scoreThresh,
            float iouThresh,
            int[] budgetTrackIds,
            float[] budgetAreas
    );

    /** C++ 侧: 调用 destroy() + delete 释放 YOLOInference */
    private static native void nativeDestroy(long handle);
}

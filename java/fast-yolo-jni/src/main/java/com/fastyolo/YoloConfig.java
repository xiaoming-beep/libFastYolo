package com.fastyolo;

import lombok.Builder;
import lombok.Getter;

import java.util.Arrays;

/**
 * YOLO 推理配置，与 C++ yolo::YOLOConfig 对应.
 *
 * <p>推荐使用 Builder 模式构建:
 * <pre>{@code
 * YoloConfig config = YoloConfig.builder()
 *     .modelData(Files.readAllBytes(Path.of("model.onnx")))
 *     .device(InferDevice.CPU)
 *     .numThreads(4)
 *     .build();
 * }</pre>
 *
 * <p>modelData 在构造和读取时均做防御性拷贝，外部修改不影响内部状态.
 */
@Getter
@Builder
public class YoloConfig {

    /** 模型文件原始字节（ONNX / MNN / TensorRT engine 等） */
    private final byte[] modelData;

    // ==================== 检测阈值 ====================

    /** 最低置信度阈值，低于此值的检测框被过滤 */
    @Builder.Default
    private final float minScore = 0.25f;
    /** IoU 阈值，NMS 时高于此 IoU 的框被抑制 */
    @Builder.Default
    private final float iouThreshold = 0.45f;

    // ==================== 设备配置 ====================

    /** 推理设备 */
    @Builder.Default
    private final InferDevice device = InferDevice.CPU;
    /** 多 GPU 时指定使用的 GPU 编号 */
    @Builder.Default
    private final int gpuId = 0;
    /** CPU 推理时的线程数（GPU/NPU 忽略） */
    @Builder.Default
    private final int numThreads = 1;
    /** 推理精度位宽 */
    @Builder.Default
    private final int precision = 32;

    // ==================== 任务配置 ====================

    /** 任务类型（AUTO 时按输出形状自动推断） */
    @Builder.Default
    private final TaskType task = TaskType.AUTO;
    /** POSE 等任务的类别数（0 = 自动推断） */
    @Builder.Default
    private final int numClasses = 0;
    /** POSE 关键点数（默认 COCO 17 点） */
    @Builder.Default
    private final int numKeypoints = 17;
    /** CLASSIFY 返回前 K 个类别 */
    @Builder.Default
    private final int topK = 5;

    // ==================== 模型输入/输出名 ====================

    /** 模型输入节点名 */
    @Builder.Default
    private final String inputName = "images";
    /** 模型输出节点名（空 = 使用全部输出，多输出模型如 seg 需要） */
    @Builder.Default
    private final String outputName = "";

    // ==================== ByteTrack 跟踪器参数 ====================

    /** 视频帧率，影响跟踪器的时间衰减 */
    @Builder.Default
    private final int trackFrameRate = 30;
    /** 轨迹保持的最大帧数（丢失后多少帧内尝试恢复） */
    @Builder.Default
    private final int trackBuffer = 30;
    /** 跟踪置信度阈值（高分/低分检测的分界线） */
    @Builder.Default
    private final float trackThresh = 0.5f;
    /** 高置信度阈值，超过此值的检测直接初始化为新轨迹 */
    @Builder.Default
    private final float highThresh = 0.6f;
    /** IoU 匹配阈值，检测框与轨迹框 IoU 大于此值认为匹配 */
    @Builder.Default
    private final float matchThresh = 0.8f;

    /**
     * 获取模型数据的防御性拷贝，防止外部修改内部状态.
     */
    public byte[] getModelData() {
        return modelData.clone();
    }

    /**
     * 自定义 Builder，确保 modelData 写入时做防御性拷贝.
     */
    public static class YoloConfigBuilder {
        public YoloConfigBuilder modelData(byte[] data) {
            this.modelData = (data != null) ? data.clone() : null;
            return this;
        }
    }

    @Override
    public String toString() {
        return "YoloConfig{"
                + "minScore=" + minScore
                + ", iouThreshold=" + iouThreshold
                + ", device=" + device
                + ", numThreads=" + numThreads
                + ", modelSize=" + (modelData != null ? modelData.length : 0) + " bytes"
                + '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof YoloConfig)) return false;
        YoloConfig that = (YoloConfig) o;
        return Float.compare(that.minScore, minScore) == 0
                && Float.compare(that.iouThreshold, iouThreshold) == 0
                && gpuId == that.gpuId
                && numThreads == that.numThreads
                && precision == that.precision
                && trackFrameRate == that.trackFrameRate
                && trackBuffer == that.trackBuffer
                && Float.compare(that.trackThresh, trackThresh) == 0
                && Float.compare(that.highThresh, highThresh) == 0
                && Float.compare(that.matchThresh, matchThresh) == 0
                && Arrays.equals(modelData, that.modelData)
                && device == that.device
                && inputName.equals(that.inputName)
                && outputName.equals(that.outputName);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(modelData);
        result = 31 * result + Float.floatToIntBits(minScore);
        result = 31 * result + Float.floatToIntBits(iouThreshold);
        result = 31 * result + device.hashCode();
        result = 31 * result + inputName.hashCode();
        result = 31 * result + outputName.hashCode();
        return result;
    }
}

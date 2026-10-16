package com.fastyolo;

import lombok.Getter;
import lombok.AllArgsConstructor;

/**
 * YOLO 检测/跟踪结果（DETECT/SEGMENT/POSE/OBB 共用）.
 * 坐标已映射回原始图像空间（由 native 侧 {@code letterbox.complete_result()} 完成）.
 * 默认构造时 trackId 为 -1（仅检测，未跟踪）.
 *
 * <p>按任务类型填充可选字段（其余为 null/0）：
 * <ul>
 *   <li>SEGMENT：{@link #getMask()} 原图分辨率二值掩码（0/1）</li>
 *   <li>POSE：{@link #getKeypoints()} 关键点（原图坐标）</li>
 *   <li>OBB：{@link #getAngle()} 旋转角（弧度）</li>
 * </ul>
 */
@Getter
@AllArgsConstructor
public class YoloResult {
    /** 边界框左上角 x（原图坐标） */
    private final int x;
    /** 边界框左上角 y（原图坐标） */
    private final int y;
    /** 边界框宽度（原图坐标） */
    private final int width;
    /** 边界框高度（原图坐标） */
    private final int height;
    /** 分类 ID */
    private final int clsId;
    /** 置信度 [0, 1] */
    private final float score;
    /** 跟踪 ID：-1 表示未经跟踪（predict），&gt;=0 为 ByteTrack 分配的 ID */
    private final int trackId;

    // ==================== 任务附带数据（SEGMENT/POSE/OBB，由 JNI 填充） ====================

    /** POSE：关键点（原图坐标），非 POSE 任务为 null */
    private Keypoint[] keypoints;

    /**
     * SEGMENT：ROI 二值掩码——长度 = width*height，0/1；行跨距 = width，
     * 局部坐标原点 = 框左上角 (x,y)。仅 predict(needMask=true) /
     * track(MaskBudget[]) 按需生成；其余调用为 null。
     */
    private byte[] mask;

    /** OBB：旋转角（弧度），非 OBB 为 0 */
    private float angle;

    /**
     * 构造检测结果（trackId 指定，任务附带数据为空）.
     */
    public YoloResult(int x, int y, int width, int height, int clsId, float score, int trackId) {
        this(x, y, width, height, clsId, score, trackId, null, null, 0.0f);
    }

    /**
     * 构造检测结果（无跟踪信息，trackId 默认为 -1）.
     */
    public YoloResult(int x, int y, int width, int height, int clsId, float score) {
        this(x, y, width, height, clsId, score, -1);
    }
}

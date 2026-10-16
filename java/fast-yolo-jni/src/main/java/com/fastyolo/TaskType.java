package com.fastyolo;

/**
 * 任务类型（值必须与 C++ yolo::TaskType 一致）。
 *
 * <p>AUTO 时按模型输出形状自动推断，大多数模型无需显式指定；
 * 歧义场景（如 v8 风格 OBB 与 DETECT 形状相近）建议显式指定。
 */
public enum TaskType {
    /** 按输出形状自动推断 */
    AUTO(0),
    /** 目标检测 */
    DETECT(1),
    /** 实例分割 */
    SEGMENT(2),
    /** 姿态关键点 */
    POSE(3),
    /** 旋转目标检测 */
    OBB(4),
    /** 图像分类 */
    CLASSIFY(5),
    /** 语义分割 */
    SEMSEG(6),
    /** 深度估计 */
    DEPTH(7);

    private final int nativeValue;

    TaskType(int nativeValue) {
        this.nativeValue = nativeValue;
    }

    public int getNativeValue() {
        return nativeValue;
    }

    public static TaskType fromNativeValue(int value) {
        for (TaskType t : values()) {
            if (t.nativeValue == value) {
                return t;
            }
        }
        return AUTO;
    }
}

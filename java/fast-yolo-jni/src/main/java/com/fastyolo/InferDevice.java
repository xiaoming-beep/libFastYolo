package com.fastyolo;

import lombok.Getter;
import lombok.AllArgsConstructor;

/**
 * 推理设备枚举，与 C++ yolo::InferDevice 一一对应.
 * nativeValue 作为 JNI 传参，与 C++ 侧值完全一致.
 */
@Getter
@AllArgsConstructor
public enum InferDevice {
    /** 自动选择可用设备 */
    AUTO(0),
    /** CPU 推理（所有后端都支持） */
    CPU(1),
    /** NPU 推理 */
    NPU(2),
    /** Vulkan 推理（MNN 后端） */
    VULKAN(3),
    /** OpenCL 推理（MNN 后端） */
    OPENCL(4),
    /** CUDA GPU 推理（TensorRT / MNN-CUDA 后端） */
    CUDA(6),
    /** 所有设备 */
    ALL(99);

    /** 与 C++ yolo::InferDevice 对应的原始值 */
    private final int nativeValue;

    /**
     * 根据 native 侧传来的 int 值还原枚举.
     * 未匹配到任何设备时默认返回 AUTO.
     */
    public static InferDevice fromNativeValue(int nativeValue) {
        for (InferDevice device : values()) {
            if (device.nativeValue == nativeValue) {
                return device;
            }
        }
        return AUTO;
    }
}

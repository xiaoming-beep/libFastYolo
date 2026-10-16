package com.fastyolo;

import lombok.Getter;

/**
 * YOLO 图像预处理——Letterbox 缩放 + 填充.
 *
 * <p>封装了 C++ 侧 {@code yolo::Letterbox} 的完整功能.
 * 典型使用流程:
 * <pre>{@code
 * // 1. 创建 Letterbox（默认 640x640，灰色填充）
 * Letterbox lb = new Letterbox(640, 114);
 *
 * // 2. 更新尺寸——计算缩放比例、目标宽高
 * lb.update(imgWidth, imgHeight);
 *
 * // 3. Java 侧将原图缩放到 lb.getDstWidth() × lb.getDstHeight()
 * // 4. 获取缩放后图像的 BGR 字节，调用 fillBGR 填充 planar float 像素
 * lb.fillBGR(resizedBGR);
 *
 * // 5. 将 letterbox 传给 YoloInference.predict() 或 track()
 * }</pre>
 *
 * <p>Letterbox 实例可跨帧复用，每帧只需 update → fillBGR.
 * 实现了 AutoCloseable，支持 try-with-resources.
 */
@Getter
public class Letterbox implements AutoCloseable {

    static {
        System.loadLibrary("FastYolo");
    }

    /** C++ 侧 yolo::Letterbox* 指针，0 表示已释放或未创建 */
    private long nativeHandle;
    /** 是否已关闭 */
    private volatile boolean closed;

    /** 缩放后图像宽度 */
    private int dstWidth;
    /** 缩放后图像高度 */
    private int dstHeight;
    /** 原始图像宽度（调用 update 后生效） */
    private int imgWidth;
    /** 原始图像高度（调用 update 后生效） */
    private int imgHeight;

    /**
     * 创建默认 Letterbox（box_size=640, 灰色填充=114, 填充模式）.
     */
    public Letterbox() {
        this(640, 114, false);
    }

    /**
     * 创建 Letterbox（填充模式）.
     *
     * @param boxSize 目标正方形边长，默认 640
     * @param color   填充颜色值 [0, 255]，默认 114（灰色）
     */
    public Letterbox(int boxSize, float color) {
        this(boxSize, color, false);
    }

    /**
     * 创建 Letterbox.
     *
     * @param boxSize    目标正方形边长，默认 640
     * @param color      填充颜色值 [0, 255]，默认 114（灰色）
     * @param centerCrop true 时为中心裁剪模式：按最短边缩放后取中心 boxSize×boxSize
     *                   （分类任务用，与 ultralytics Resize+CenterCrop 一致）；false 为填充模式
     */
    public Letterbox(int boxSize, float color, boolean centerCrop) {
        this.nativeHandle = nativeCreate(boxSize, color, centerCrop);
        if (this.nativeHandle == 0) {
            throw new YoloException(YoloException.STATUS_MODEL_INIT_FAILED,
                    "Failed to create Letterbox native instance");
        }
    }

    /**
     * 根据原始图像宽高计算缩放比例和 padding.
     * 调用后 dstWidth/dstHeight/imgWidth/imgHeight 字段会被 native 侧更新.
     *
     * @param width  原始图像宽度
     * @param height 原始图像高度
     */
    public void update(int width, int height) {
        ensureOpen();
        nativeUpdate(this.nativeHandle, width, height);
    }

    /**
     * 将缩放后的 BGR 图像数据填充到 letterbox planar float 缓冲区.
     * 数据长度应为 dstWidth * dstHeight * 3 字节.
     *
     * @param data BGR 顺序的 uint8 像素数据
     */
    public void fillBGR(byte[] data) {
        ensureOpen();
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("BGR data must not be null or empty");
        }
        nativeFillBGR(this.nativeHandle, data);
    }

    /**
     * 通过 native 指针地址直接填充 BGR 数据，避免 byte[] 拷贝.
     *
     * <p>适用于 OpenCV Mat 等通过 {@code dataAddr()} 暴露指针的场合:
     * <pre>{@code
     * Mat resizedImg = new Mat();
     * // ... 缩放原图到 lb.getDstWidth() × lb.getDstHeight()
     * letterbox.fillBGR(resizedImg.dataAddr());
     * }</pre>
     *
     * @param dataPtr BGR uint8 像素数据的内存地址（须保证在调用期间有效）
     */
    public void fillBGR(long dataPtr) {
        ensureOpen();
        if (dataPtr == 0) {
            throw new IllegalArgumentException("BGR data pointer must not be null");
        }
        nativeFillBGRAddr(this.nativeHandle, dataPtr);
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            long h = nativeHandle;
            nativeHandle = 0;
            if (h != 0) {
                nativeDestroy(h);
            }
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Letterbox 已关闭");
        }
    }

    // ==================== JNI native 方法 ====================

    /** 创建 C++ Letterbox 对象并预分配 pixels 缓冲区 */
    private static native long nativeCreate(int boxSize, float color, boolean centerCrop);

    /** 调用 C++ Letterbox::update() 并同步 dstWidth/dstHeight/imgWidth/imgHeight 到 Java 字段 */
    private native void nativeUpdate(long handle, int width, int height);

    /** 调用 C++ Letterbox::fill_bgr() 完成 BGR→planar float 转换 */
    private native void nativeFillBGR(long handle, byte[] data);

    /** 通过指针地址调用 C++ Letterbox::fill_bgr()，零拷贝路径 */
    private native void nativeFillBGRAddr(long handle, long dataPtr);

    /** 释放 C++ Letterbox 对象 */
    private static native void nativeDestroy(long handle);
}

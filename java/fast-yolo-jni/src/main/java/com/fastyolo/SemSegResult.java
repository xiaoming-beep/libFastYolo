package com.fastyolo;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * 语义分割结果：原图分辨率逐像素类别图。
 * 类别值 < 256（uint8 存储，JNI 转 int）。
 */
@Getter
@AllArgsConstructor
public class SemSegResult {
    /** 原图宽度 */
    private final int width;
    /** 原图高度 */
    private final int height;
    /** 逐像素类别 id，长度 width * height */
    private final int[] classMap;
}

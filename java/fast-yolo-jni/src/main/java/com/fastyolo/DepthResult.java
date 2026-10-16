package com.fastyolo;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * 深度估计结果：原图分辨率 float32 深度图（模型原始值）+ 值域。
 */
@Getter
@AllArgsConstructor
public class DepthResult {
    /** 原图宽度 */
    private final int width;
    /** 原图高度 */
    private final int height;
    /** 逐像素深度，长度 width * height */
    private final float[] depthMap;
    /** 深度最小值（有效区域） */
    private final float depthMin;
    /** 深度最大值（有效区域） */
    private final float depthMax;
}

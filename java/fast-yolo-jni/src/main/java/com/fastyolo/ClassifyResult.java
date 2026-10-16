package com.fastyolo;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * 图像分类结果。
 */
@Getter
@AllArgsConstructor
public class ClassifyResult {
    /** 置信度最高的类别 id */
    private final int topClass;
    /** 最高类别分数 */
    private final float topScore;
    /** 返回的前 K 个类别数（= 配置 topK） */
    private final int topK;
    /** 前 topK 类别 id（按分数降序） */
    private final int[] topIds;
    /** 前 topK 类别分数 */
    private final float[] topScores;
    /** 全部分类分数（softmax 后，长度为类别数） */
    private final float[] scores;
}

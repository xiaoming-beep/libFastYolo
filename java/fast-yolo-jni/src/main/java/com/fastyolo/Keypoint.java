package com.fastyolo;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * 姿态关键点（原图坐标，由 native 侧 letterbox 逆变换反算）。
 */
@Getter
@AllArgsConstructor
public class Keypoint {
    /** 关键点 x（原图坐标） */
    private final float x;
    /** 关键点 y（原图坐标） */
    private final float y;
    /** 关键点置信度 [0, 1] */
    private final float score;
}

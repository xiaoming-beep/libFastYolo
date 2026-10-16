// 池化。yolo11n 只有一种形态：MaxPool 5x5 stride=1 pad=2（SPPF 结构）。
// 但按通用形式写，顺便支持别的配置。
//
// in/out: [N, C, H, W]

__kernel void maxpool(__global const float* in,
                      __global float* out,
                      const int4 inShape,
                      const int4 outShape,
                      const int2 ksize,
                      const int2 stride,
                      const int2 pad,
                      const int countIncludePad) {
    const int total = outShape.s0 * outShape.s1 * outShape.s2 * outShape.s3;
    const int gid = get_global_id(0);
    if (gid >= total) {
        return;
    }
    const int ow = gid % outShape.s3;
    int t = gid / outShape.s3;
    const int oh = t % outShape.s2;
    t /= outShape.s2;
    const int c = t % outShape.s1;
    const int n = t / outShape.s1;

    __global const float* ip = in + (((long)n * inShape.s1 + c) * inShape.s2) * inShape.s3;

    float best = -INFINITY;
    for (int r = 0; r < ksize.x; ++r) {
        const int ih = oh * stride.x - pad.x + r;
        if (ih < 0 || ih >= inShape.s2) {
            continue;
        }
        for (int s = 0; s < ksize.y; ++s) {
            const int iw = ow * stride.y - pad.y + s;
            if (iw < 0 || iw >= inShape.s3) {
                continue;
            }
            best = fmax(best, ip[ih * inShape.s3 + iw]);
        }
    }
    // 全部越界（理论上不会发生，因为 pad < k）时给 0
    out[gid] = (best == -INFINITY) ? 0.0f : best;
}

// 全局平均池化（cls 模型用）
__kernel void global_avgpool(__global const float* in,
                             __global float* out,
                             const int4 inShape) {
    const int nc = get_global_id(0);       // 每个 work-item 一个 (n, c)
    if (nc >= inShape.s0 * inShape.s1) {
        return;
    }
    const long plane = (long)inShape.s2 * inShape.s3;
    __global const float* ip = in + (long)nc * plane;
    float sum = 0.0f;
    for (int i = 0; i < plane; ++i) {
        sum += ip[i];
    }
    out[nc] = sum / (float)plane;
}

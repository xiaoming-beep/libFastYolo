// 通用「带步长块拷贝」。一个内核覆盖 4 个算子：
//
//   Concat    : shape=输入形状, srcStride=输入自然步长, dstStride=输出自然步长,
//               dstOffset=本输入在输出里的起始偏移
//   Split     : shape=本路输出形状, srcStride=输入自然步长, srcOffset=本路起点
//   Slice     : shape=输出形状, srcStride=输入自然步长, srcOffset=切片起点
//   Transpose : shape=输出形状, srcStride=按 perm 重排后的输入步长
//
// 全部按 4 维处理（不足的维度 shape 填 1、stride 填 0）。

__kernel void copy_strided(__global const float* src,
                           __global float* dst,
                           const int4 shape,
                           const int4 srcStride,
                           const int4 dstStride,
                           const int srcOffset,
                           const int dstOffset,
                           const int total) {
    const int gid = get_global_id(0);
    if (gid >= total) {
        return;
    }
    const int i3 = gid % shape.s3;
    int t = gid / shape.s3;
    const int i2 = t % shape.s2;
    t /= shape.s2;
    const int i1 = t % shape.s1;
    const int i0 = t / shape.s1;

    const int si = srcOffset + i0 * srcStride.s0 + i1 * srcStride.s1 + i2 * srcStride.s2 +
                   i3 * srcStride.s3;
    const int di = dstOffset + i0 * dstStride.s0 + i1 * dstStride.s1 + i2 * dstStride.s2 +
                   i3 * dstStride.s3;
    dst[di] = src[si];
}

// ---------------------------------------------------------------------------
// pixel shuffle 2x：把 [N, C*4, H, W] 重排成 [N, C, 2H, 2W]
//
// 用途：ConvTranspose(k=2, s=2, p=0) 等价于「1x1 卷积 C_in->C_out*4」+ 这个 shuffle，
// YOLO 的 seg / depth 头就是这一种形态（实测三个模型参数完全一致）。
//
// 映射关系：
//   src (n, oc*4 + dh*2 + dw, h, w)  ->  dst (n, oc, 2h+dh, 2w+dw)
// 即把每个输出通道的 2x2 子块拆到空间上去。
// 每个源元素只读一次、每个目标元素只写一次，是纯搬运。
//
// bias 在这里加：1x1 卷积输出有 C*4 个通道，但它们共用同一个 oc 的 bias，
// 所以在 shuffle 时按 oc 加最方便（省掉一次 bias 重排）。
// ---------------------------------------------------------------------------
__kernel void pixel_shuffle2x(__global const float* src,
                              __global float* dst,
                              __global const float* bias,
                              const int hasBias,
                              const int C, const int H, const int W, const int N) {
    const int perN = C * 4 * H * W;   // 单个 batch 的元素数（源、目标大小相同）
    const long total = (long)N * perN;
    const long gid = get_global_id(0);
    if (gid >= total) {
        return;
    }
    const int n = (int)(gid / perN);
    // 注意：不能叫 local —— 它是 OpenCL 的地址空间限定符关键字，
    // 拿来当变量名会直接编译报错（"expected identifier or '('"）
    const int inner = (int)(gid - (long)n * perN);

    const int plane = 4 * H * W;                  // 每个 oc 的输出元素数 = (2H)*(2W)
    const int oc = inner / plane;
    const int rem = inner - oc * plane;
    const int oh = rem / (2 * W);
    const int ow = rem - oh * (2 * W);
    const int dh = oh & 1;
    const int dw = ow & 1;
    const int h = oh >> 1;
    const int w = ow >> 1;

    const int sc = oc * 4 + dh * 2 + dw;
    float v = src[(long)n * perN + (sc * H + h) * W + w];
    if (hasBias != 0) {
        v += bias[oc];
    }
    dst[gid] = v;
}

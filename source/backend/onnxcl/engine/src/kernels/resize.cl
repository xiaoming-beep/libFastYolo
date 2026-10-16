// Resize。yolo11n 只用 nearest + coordinate_transformation_mode=asymmetric + nearest_mode=floor。
//
// asymmetric: x_orig = x_resized / scale      （scale = out/in）
// floor     : 取 floor
//
// in/out: [N, C, H, W]

__kernel void resize_nearest(__global const float* in,
                             __global float* out,
                             const int4 inShape,
                             const int4 outShape,
                             const float scaleH,   // outH / inH
                             const float scaleW) {
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

    int ih = (int)floor((float)oh / scaleH);
    int iw = (int)floor((float)ow / scaleW);
    // 防御性 clamp（scales 不是整数比时可能算出边界外）
    ih = min(max(ih, 0), inShape.s2 - 1);
    iw = min(max(iw, 0), inShape.s3 - 1);

    __global const float* ip = in + (((long)n * inShape.s1 + c) * inShape.s2) * inShape.s3;
    out[gid] = ip[ih * inShape.s3 + iw];
}

// 双线性，half_pixel（depth / sem 模型用，M6 再验）
__kernel void resize_bilinear(__global const float* in,
                              __global float* out,
                              const int4 inShape,
                              const int4 outShape,
                              const float scaleH,
                              const float scaleW,
                              const int alignCorners) {
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

    float y, x;
    if (alignCorners != 0) {
        const float sh = (outShape.s2 > 1) ? (float)(inShape.s2 - 1) / (float)(outShape.s2 - 1) : 0.0f;
        const float sw = (outShape.s3 > 1) ? (float)(inShape.s3 - 1) / (float)(outShape.s3 - 1) : 0.0f;
        y = oh * sh;
        x = ow * sw;
    } else {
        y = ((float)oh + 0.5f) / scaleH - 0.5f;
        x = ((float)ow + 0.5f) / scaleW - 0.5f;
    }
    y = fmax(y, 0.0f);
    x = fmax(x, 0.0f);

    const int y0 = (int)floor(y);
    const int x0 = (int)floor(x);
    const int y1 = min(y0 + 1, inShape.s2 - 1);
    const int x1 = min(x0 + 1, inShape.s3 - 1);
    const float fy = y - (float)y0;
    const float fx = x - (float)x0;

    __global const float* ip = in + (((long)n * inShape.s1 + c) * inShape.s2) * inShape.s3;
    const float v00 = ip[y0 * inShape.s3 + x0];
    const float v01 = ip[y0 * inShape.s3 + x1];
    const float v10 = ip[min(y0 + 1, inShape.s2 - 1) * inShape.s3 + x0];
    const float v11 = ip[y1 * inShape.s3 + x1];

    const float top = v00 + (v01 - v00) * fx;
    const float bot = v10 + (v11 - v10) * fx;
    out[gid] = top + (bot - top) * fy;
}

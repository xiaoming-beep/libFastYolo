// Softmax。把张量看成 [outer, axisSize, inner] 三维，在 axisSize 维上做归一化。
// 这样任意 axis 都能覆盖：
//   axis = 最后一维        -> inner = 1
//   axis = 中间某一维      -> inner = 后面所有维的乘积
//
// yolo11n 用到两种：axis=-1（inner=1）和 axis=1（inner=4*8400）。

__kernel void softmax(__global const float* in,
                      __global float* out,
                      const int outer,
                      const int axisSize,
                      const int inner) {
    const int o = get_global_id(0);
    const int i = get_global_id(1);
    if (o >= outer || i >= inner) {
        return;
    }
    const long base = (long)o * axisSize * inner + i;
    __global const float* p = in + base;
    __global float* q = out + base;

    float mx = -INFINITY;
    for (int k = 0; k < axisSize; ++k) {
        mx = fmax(mx, p[(long)k * inner]);
    }
    float sum = 0.0f;
    for (int k = 0; k < axisSize; ++k) {
        sum += exp(p[(long)k * inner] - mx);
    }
    const float inv = 1.0f / sum;
    for (int k = 0; k < axisSize; ++k) {
        q[(long)k * inner] = exp(p[(long)k * inner] - mx) * inv;
    }
}

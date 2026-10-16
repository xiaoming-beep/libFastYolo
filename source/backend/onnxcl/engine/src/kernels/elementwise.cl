// 二元逐元素算子。同一个内核覆盖三种形态：
//   · same-shape  —— strides 就是自然步长
//   · 标量        —— strides 全 0
//   · 一般广播    —— 被广播的维度 stride 给 0
//
// outShape 和 stride 都已 pad 到 4 维（缺的高维填 1 / 0）。

// op: 0=add 1=sub 2=mul 3=div 4=pow 5=max 6=min 7=fmod
__kernel void binary_op(__global const float* a,
                        __global const float* b,
                        __global float* out,
                        const int4 outShape,
                        const int4 aStride,
                        const int4 bStride,
                        const int total,
                        const int op) {
    const int gid = get_global_id(0);
    if (gid >= total) {
        return;
    }
    // 线性下标 -> 4 维下标
    const int i3 = gid % outShape.s3;
    int t = gid / outShape.s3;
    const int i2 = t % outShape.s2;
    t /= outShape.s2;
    const int i1 = t % outShape.s1;
    const int i0 = t / outShape.s1;

    const int ia = i0 * aStride.s0 + i1 * aStride.s1 + i2 * aStride.s2 + i3 * aStride.s3;
    const int ib = i0 * bStride.s0 + i1 * bStride.s1 + i2 * bStride.s2 + i3 * bStride.s3;
    const float x = a[ia];
    const float y = b[ib];

    float r;
    switch (op) {
        case 0: r = x + y; break;
        case 1: r = x - y; break;
        case 2: r = x * y; break;
        case 3: r = x / y; break;
        case 4: r = pow(x, y); break;
        case 5: r = fmax(x, y); break;
        case 6: r = fmin(x, y); break;
        // 8 = 整数除法（截断）。ONNX 的 Div 对 int32/int64 输入做的是
        // 整数除法，不是浮点除法；用输入张量的 dtype 判断走哪条路。
        case 8: r = trunc(x / y); break;
        default: r = fmod(x, y); break;
    }
    out[gid] = r;
}

// 一元算子
// op: 0=sigmoid 1=exp 2=sqrt 3=log 4=cos 5=sin 6=tanh 7=relu 8=abs 9=neg
//     10=floor 11=ceil 12=erf 13=silu 14=relu6 15=sinh 16=cosh
__kernel void unary_op(__global const float* in, __global float* out,
                       const int total, const int op) {
    const int gid = get_global_id(0);
    if (gid >= total) {
        return;
    }
    const float x = in[gid];
    float r;
    switch (op) {
        case 0: r = 1.0f / (1.0f + exp(-x)); break;
        case 1: r = exp(x); break;
        case 2: r = sqrt(x); break;
        case 3: r = log(x); break;
        case 4: r = cos(x); break;
        case 5: r = sin(x); break;
        case 6: r = tanh(x); break;
        case 7: r = fmax(x, 0.0f); break;
        case 8: r = fabs(x); break;
        case 9: r = -x; break;
        case 10: r = floor(x); break;
        case 11: r = ceil(x); break;
        case 12: r = erf(x); break;
        case 13: r = x / (1.0f + exp(-x)); break;   // SiLU
        case 14: r = fmin(fmax(x, 0.0f), 6.0f); break;
        case 15: r = sinh(x); break;
        default: r = cosh(x); break;
    }
    out[gid] = r;
}

// Clip：min/max 由 host 传入
__kernel void clip_op(__global const float* in, __global float* out,
                      const int total, const float lo, const float hi) {
    const int gid = get_global_id(0);
    if (gid >= total) {
        return;
    }
    out[gid] = fmin(fmax(in[gid], lo), hi);
}

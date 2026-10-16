// 带 batch 的矩阵乘，同时充当 ONNX 的 Gemm。
//
// MatMul: transB=0、hasC=0
// Gemm  : transB 看属性；第三个输入是偏置 C（形状 [N]，按列广播）；alpha/beta 是缩放因子
//
// 【为什么必须支持 transB】YOLO 分类头的 Gemm 是 transB=1，也就是权重按 [N,K] 存、
// 要算 A x B^T。忽略它的后果不是“算得不准”，而是把 N 取错（1280 而不是 1000），
// 往输出缓冲里多写 280 个元素 —— 越界加数据错乱，最终表现为输出全是 NaN。
__kernel void batched_matmul(__global const float* A,   // [batch, M, K]
                             __global const float* B,   // transB=0: [batch, K, N]；transB=1: [batch, N, K]
                             __global const float* C,   // [N] 偏置（hasC=0 时传占位）
                             __global float* Y,         // [batch, M, N]
                             const int batch,
                             const int M,
                             const int K,
                             const int N,
                             const int transB,
                             const int hasC,
                             const float alpha,
                             const float beta) {
    const long total = (long)batch * M * N;
    const long gid = get_global_id(0);
    if (gid >= total) {
        return;
    }
    const int n = (int)(gid % N);
    long t = gid / N;
    const int m = (int)(t % M);
    const int b = (int)(t / M);

    __global const float* a = A + ((long)b * M + m) * K;
    __global const float* bb = B + (long)b * K * N;

    float acc = 0.0f;
    if (transB != 0) {
        // B 是 [N, K]，要取它的第 n 行做点积 —— 连续访存，比转置版快
        __global const float* bn = bb + (long)n * K;
        for (int k = 0; k < K; ++k) {
            acc += a[k] * bn[k];
        }
    } else {
        for (int k = 0; k < K; ++k) {
            acc += a[k] * bb[(long)k * N + n];
        }
    }
    acc *= alpha;
    if (hasC != 0) {
        acc += beta * C[n];
    }
    Y[gid] = acc;
}

// 归约：在最后一维上取最大值/求和（ReduceMax / ReduceMean）
// op: 0 = max, 1 = sum
__kernel void reduce_last_dim(__global const float* in,
                              __global float* out,
                              const int outer,
                              const int axisSize,
                              const int op) {
    const int o = get_global_id(0);
    if (o >= outer) {
        return;
    }
    __global const float* p = in + (long)o * axisSize;
    float acc = (op == 0) ? -INFINITY : 0.0f;
    for (int k = 0; k < axisSize; ++k) {
        acc = (op == 0) ? fmax(acc, p[k]) : (acc + p[k]);
    }
    out[o] = acc;
}

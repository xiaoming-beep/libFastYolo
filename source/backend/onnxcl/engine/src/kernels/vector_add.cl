// M1 冒烟测试用内核：验证「源码编译 → 参数绑定 → 启动 → 读回 → 计时」整条链路。
// 同时也是后面 elementwise 算子的模板。

__kernel void vector_add(__global const float* a,
                         __global const float* b,
                         __global float* c,
                         const int n) {
    const int i = get_global_id(0);
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// M4 会用来做显存带宽体检：纯拷贝，不带计算
__kernel void vector_copy(__global const float* a, __global float* c, const int n) {
    const int i = get_global_id(0);
    if (i < n) {
        c[i] = a[i];
    }
}

// 用 local memory 做向量加。
// 注意：__local 必须写成【参数】而不是函数体内静态声明——
// 真实的 conv/gemm 内核需要 host 侧按 tile 大小动态设置 local 内存尺寸，
// 静态声明会导致 setLocal 报 CL_INVALID_ARG_INDEX。
__kernel void vector_add_local(__global const float* a,
                               __global const float* b,
                               __global float* c,
                               const int n,
                               __local float* tile) {
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    tile[lid] = (gid < n) ? a[gid] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (gid < n) {
        c[gid] = tile[lid] + b[gid];
    }
}

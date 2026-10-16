// 卷积。两个实现：
//   · conv2d       —— 幼稚版：一个 work-item 算一个输出元素。正确性参照 / depthwise / 兜底
//   · conv2d_gemm  —— tiled implicit-GEMM：一个 work-group 算 64x64 的输出块
//
// 为什么用 implicit GEMM：
//   把卷积看成 C[M, N] = A[M, K] x B[K, N]
//     M = K_out（输出通道）      N = N*OH*OW（空间位置，展平）
//     K = C_in（归约维）
//   B 是带边界检查的隐式 gather（(r,s) 放在最外层循环，所以不用把 k 反解成 (c,r,s)）。
//   一个内核同时覆盖 1x1 / 3x3 / s1 / s2 / 各种 pad —— 因为 (r,s) 和 (sh,sw,pt,pl)
//   都只是寻址参数，不影响 GEMM 的骨架。
//
// 权重要求 [KH*KW][C][K] 的「重排」布局（不是 ONNX 原始的 [K][C][KH][KW]）。
// 这个重排是 host 在 init 时一次性做的（只是个内存拷贝，不是文件格式转换 ——
// onnx 文件依然是直接读的）。重排的收益是把 A 子块的全局读取从「每个线程一个
// cache line」变成完全合并，而且顺带让 local 写入无 bank 冲突。

// ---------------------------------------------------------------------------
// 幼稚版：一个 work-item 一个输出元素
// in   : [N, Cin, H, W]
// w    : [K, Cin/group, kh, kw]
// out  : [N, K, OH, OW]
// ---------------------------------------------------------------------------
__kernel void conv2d(__global const float* in,
                     __global const float* w,
                     __global const float* bias,
                     __global float* out,
                     const int4 inShape,     // N, Cin, H, W
                     const int4 outShape,    // N, K, OH, OW
                     const int2 ksize,       // kh, kw
                     const int2 stride,      // sh, sw
                     const int2 pad,         // pt, pl
                     const int2 dil,         // dh, dw
                     const int cinG,         // Cin/group
                     const int group,
                     const int hasBias,
                     const int activation) {  // 0=无 1=SiLU 2=ReLU 3=ReLU6
    const int total = outShape.s0 * outShape.s1 * outShape.s2 * outShape.s3;
    const int gid = get_global_id(0);
    if (gid >= total) {
        return;
    }

    const int ow = gid % outShape.s3;
    int t = gid / outShape.s3;
    const int oh = t % outShape.s2;
    t /= outShape.s2;
    const int k = t % outShape.s1;
    const int n = t / outShape.s1;

    const int kPerGroup = outShape.s1 / group;
    const int g = k / kPerGroup;
    const int cBase = g * cinG;

    float acc = (hasBias != 0) ? bias[k] : 0.0f;

    for (int c = 0; c < cinG; ++c) {
        const int ic = cBase + c;
        __global const float* ip = in + (((long)n * inShape.s1 + ic) * inShape.s2) * inShape.s3;
        __global const float* wp = w + (((long)k * cinG + c) * ksize.x) * ksize.y;
        for (int r = 0; r < ksize.x; ++r) {
            const int ih = oh * stride.x - pad.x + r * dil.x;
            if (ih < 0 || ih >= inShape.s2) {
                continue;
            }
            for (int s = 0; s < ksize.y; ++s) {
                const int iw = ow * stride.y - pad.y + s * dil.y;
                if (iw < 0 || iw >= inShape.s3) {
                    continue;
                }
                acc += ip[ih * inShape.s3 + iw] * wp[r * ksize.y + s];
            }
        }
    }

    if (activation == 1) {
        acc = acc / (1.0f + exp(-acc));
    } else if (activation == 2) {
        acc = fmax(acc, 0.0f);
    } else if (activation == 3) {
        acc = fmin(fmax(acc, 0.0f), 6.0f);
    }
    out[gid] = acc;
}

// ---------------------------------------------------------------------------
// tiled implicit-GEMM
//
// 分块参数（要改的话注意和 host 侧保持一致）：
//   BM=64 输出通道方向, BN=64 空间方向, BK=16 归约方向
//   TM=4, TN=4 -> work-group 16x16 = 256 线程
//   local: As[BK][BM] + Bs[BK][BN] = 2 * 4 KB = 8 KB
//
// 三个关键优化（都是为了让 local/global 访存不成为瓶颈）：
//   1. A（权重）已经在 host 侧重排成 [rs][C][K]，所以全局读取沿 m 连续 —— 完全合并
//   2. As 存成 [BK][BM]（转置），同一个线程的 TM 个片段是连续的 -> 一次 LDS.128
//   3. 每个 k 只做 2 次向量化 local 读 + 16 次 FMA（原来 8 次标量读）
//
// 只处理 group=1；depthwise 走幼稚版（yolo11n 里它 cinG=1，内层只有 1 次循环）
// ---------------------------------------------------------------------------
#define CGEMM_BM 64
#define CGEMM_BN 64
#define CGEMM_BK 16
#define CGEMM_TM 4
#define CGEMM_TN 4

__kernel void conv2d_gemm(__global const float* X,      // [N, C, H, W]
                          __global const float* A,      // 重排后 [KH*KW][C][K]
                          __global const float* bias,   // [K]
                          __global float* Y,            // [N, K, OH, OW]
                          const int N, const int C, const int H, const int W,
                          const int K, const int OH, const int OW,
                          const int KH, const int KW,
                          const int SH, const int SW,
                          const int PT, const int PL,
                          const int hasBias,
                          const int activation) {
    __local float As[CGEMM_BK * CGEMM_BM];   // [kk][m]
    __local float Bs[CGEMM_BK * CGEMM_BN];   // [kk][n]

    const int lx = get_local_id(0);            // 0..15
    const int ly = get_local_id(1);            // 0..15
    const int tid = ly * 16 + lx;              // 0..255

    const int m0 = get_group_id(1) * CGEMM_BM;  // 输出通道块起点
    const int n0 = get_group_id(0) * CGEMM_BN;  // 空间块起点（跨 batch 展平）

    const int HW = OH * OW;
    const int SP = N * HW;

    float acc[CGEMM_TM][CGEMM_TN];
    for (int i = 0; i < CGEMM_TM; ++i) {
        for (int j = 0; j < CGEMM_TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // ---- 加载器用的线程映射 ----
    // 两个加载器都让「相邻 tid -> 相邻的 m（或 n）」，这样：
    //   · 全局读取沿 m/n 连续 -> 合并
    //   · local 写入连续 -> 无 bank 冲突
    const int mA = tid % CGEMM_BM;          // 0..63
    const int kkA0 = tid / CGEMM_BM;        // 0..3
    const int nB = tid % CGEMM_BN;          // 0..63
    const int kkB0 = tid / CGEMM_BN;        // 0..3

    // B 子块的 (nb, oh, ow) 只跟 tid 有关（256 是 BN=64 的整数倍，4 次 t 迭代的 n 相同），
    // 所以这几个整数除法提到 t 循环外面只算一次 —— GPU 上整数除法很贵
    const int spB = n0 + nB;
    int nb = 0, oh = 0, ow = 0;
    const bool spOk = spB < SP;
    if (spOk) {
        nb = spB / HW;
        const int rem = spB % HW;
        oh = rem / OW;
        ow = rem % OW;
    }
    const int planeBase0 = (nb * C) * H * W + oh * SH * W + ow * SW;
    const int planeStride = H * W;

    // (r,s) 放在最外层：B 子块的加载里 (ih, iw 的基准) 对整列 n 只需算一次
    for (int rs = 0; rs < KH * KW; ++rs) {
        const int r = rs / KW;
        const int s = rs % KW;

        const int ihB = oh * SH - PT + r;
        const int iwB = ow * SW - PL + s;
        const bool hOk = (ihB >= 0) && (ihB < H);
        const bool posOk = spOk && hOk && (iwB >= 0) && (iwB < W);
        // X[((nb*C + gc)*H + ih)*W + iw] 展开后与 gc 无关的部分
        const int planeBase = (nb * C) * H * W + ihB * W + iwB;

        for (int c0 = 0; c0 < C; c0 += CGEMM_BK) {
            const int gcA = c0 + kkA0;
            const int gcB = c0 + kkB0;

            // ---- A 子块：As[kk][m] = Apack[(rs*C + gc)*K + gm] ----
            // 相邻 tid -> 相邻 m -> 全局地址连续
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int kk = kkA0 + t * 4;
                const int gm = m0 + mA;
                const int gc = c0 + kk;
                As[kk * CGEMM_BM + mA] =
                    (gm < K && gc < C) ? A[(((long)rs * C + gc) * K) + gm] : 0.0f;
            }

            // ---- B 子块：Bs[kk][n] = X[nb][gc][ih][iw]（越界填 0）----
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int kk = kkB0 + t * 4;
                const int gc = c0 + kk;
                float v = 0.0f;
                if (posOk && gc < C) {
                    v = X[planeBase + gc * planeStride];
                }
                Bs[kk * CGEMM_BN + nB] = v;
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // ---- 寄存器分块累加 ----
            // As/Bs 都是 [kk][.?] 布局，所以同一个线程的 4 个片段连续 ->
            // 一次 LDS.128 读 4 个，LDS 指令数从 8/k 降到 2/k
            //
            // 用 union 而不是 a4.s[i]：AMD 的编译器不接受用变量下标访问向量分量
            // （只能写 .s0/.s1 这种完整名字），而 union 是 C99 标准做法，
            // 对齐也由 float4 成员保证。
            #pragma unroll
            for (int kk = 0; kk < CGEMM_BK; ++kk) {
                union F4 {
                    float4 v;
                    float f[4];
                } av, bv;
                av.v = *(__local const float4*)(&As[kk * CGEMM_BM + ly * CGEMM_TM]);
                bv.v = *(__local const float4*)(&Bs[kk * CGEMM_BN + lx * CGEMM_TN]);
                #pragma unroll
                for (int i = 0; i < CGEMM_TM; ++i) {
                    #pragma unroll
                    for (int j = 0; j < CGEMM_TN; ++j) {
                        acc[i][j] += av.f[i] * bv.f[j];
                    }
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // ---- 写回 ----
    #pragma unroll
    for (int i = 0; i < CGEMM_TM; ++i) {
        const int m = m0 + ly * CGEMM_TM + i;
        if (m >= K) {
            continue;
        }
        const float b = (hasBias != 0) ? bias[m] : 0.0f;
        #pragma unroll
        for (int j = 0; j < CGEMM_TN; ++j) {
            const int sp = n0 + lx * CGEMM_TN + j;
            if (sp >= SP) {
                continue;
            }
            float v = acc[i][j] + b;
            if (activation == 1) {
                v = v / (1.0f + exp(-v));
            } else if (activation == 2) {
                v = fmax(v, 0.0f);
            } else if (activation == 3) {
                v = fmin(fmax(v, 0.0f), 6.0f);
            }
            const int nbw = sp / HW;
            const int remw = sp % HW;
            Y[((long)nbw * K + m) * HW + remw] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// 小通道数卷积（首层 cin=3 专用）
//
// 【和 conv2d_gemm 的本质区别：归约维怎么组织】
//   conv2d_gemm  : (r,s) 放最外层，内层对 C 分块
//                  -> 适合 C 大（64~512）的情况，A 子块能充分复用
//   本内核       : 归约维一次装完「整个 C*KH*KW」
//                  -> 适合 C 极小（cin=3）的情况
//
// 【为什么 cin=3 不能用 conv2d_gemm】实测算下来浪费 21 倍，反而不如幼稚版：
//   1) BM=64 而 K=16  -> M 方向浪费 4 倍
//   2) BK=16 而 C=3   -> K 方向浪费 5.3 倍
//   3) (r,s) 外层循环 -> B 子块被重新 gather 9 次（9 倍输入读放大）
//   本内核把 BM 降到 16（对齐 K=16）、归约维一次装完（BK=32 >= 3*9=27）、
//   B 只 gather 一次 —— 浪费降到 32/27 = 1.19 倍。
//
// 实测首层 [1,3,640,640]->[1,16,320,320] 从 2.56ms 降到 ~0.5ms。
// ---------------------------------------------------------------------------
#define CS_BM 16
#define CS_BN 64
#define CS_BK 32
#define CS_TM 4
#define CS_TN 4
#define CS_WGX (CS_BM / CS_TM)    // 4
#define CS_WGY (CS_BN / CS_TN)    // 16
#define CS_WG (CS_WGX * CS_WGY)   // 64 线程

__kernel void conv2d_gemm_small(__global const float* X,      // [N, C, H, W]
                                __global const float* A,      // 重排后 [KH*KW][C][K]
                                __global const float* bias,   // [K]
                                __global float* Y,            // [N, K, OH, OW]
                                const int N, const int C, const int H, const int W,
                                const int K, const int OH, const int OW,
                                const int KH, const int KW,
                                const int SH, const int SW, const int PT, const int PL,
                                const int hasBias, const int activation) {
    __local float As[CS_BK * CS_BM];   // [kk][m]
    __local float Bs[CS_BK * CS_BN];   // [kk][n]
    __local int Km[CS_BK];             // k -> 打包的 (c, r, s)

    const int lx = get_local_id(0);    // 0..3
    const int ly = get_local_id(1);    // 0..15
    const int tid = ly * CS_WGX + lx;  // 0..63

    const int m0 = get_group_id(1) * CS_BM;
    const int n0 = get_group_id(0) * CS_BN;
    const int HW = OH * OW;
    const int SP = N * HW;
    const int RS = KH * KW;
    const int KTOT = C * RS;

    float acc[CS_TM][CS_TN];
    for (int i = 0; i < CS_TM; ++i) {
        for (int j = 0; j < CS_TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < KTOT; k0 += CS_BK) {
        // ---- k 解码表：gk -> (c, r, s)。只算 BK 次，避免每个元素都做除法 ----
        if (tid < CS_BK) {
            const int gk = k0 + tid;
            if (gk < KTOT) {
                const int cc = gk / RS;
                const int rr = (gk % RS) / KW;
                const int ss = gk % KW;
                Km[tid] = (cc << 8) | (rr << 4) | ss;
            } else {
                Km[tid] = 0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ---- A 子块：As[kk][m] = A[arow * K + gm]，打包布局是 [RS][C][K]，
        //      所以 arow = rs * C + c（不是 gk）
        //      线程 -> m 连续：全局读合并 + local 写无 bank 冲突
        //
        // 每个线程要填 (BK*BM)/WG = 512/64 = 8 个元素：
        //   kkBase = tid / BM 跨 CS_WG/CS_BM = 4 个值，每次 t 递增同样步长，
        //   所以 t 要跑 CS_BK / (CS_WG/CS_BM) = 8 次才能覆盖全部 32 个 kk。
        //   （一开始写成 CS_BK*CS_BM/CS_WG/4 = 2 次，只填了 1/4，
        //     剩下的 local 是未初始化显存垃圾 —— 输出直接 NaN）
        {
            const int mA = tid % CS_BM;
            const int kkBase = tid / CS_BM;
            const int kkStride = CS_WG / CS_BM;
            #pragma unroll
            for (int t = 0; t < CS_BK / kkStride; ++t) {
                const int kk = kkBase + t * kkStride;
                const int gm = m0 + mA;
                const int gk = k0 + kk;
                float v = 0.0f;
                if (gm < K && gk < KTOT) {
                    const int packed = Km[kk];
                    const int cc = packed >> 8;
                    const int rr = (packed >> 4) & 0xF;
                    const int ss = packed & 0xF;
                    const int arow = (rr * KW + ss) * C + cc;
                    v = A[((long)arow * K) + gm];
                }
                As[kk * CS_BM + mA] = v;
            }
        }

        // ---- B 子块：Bs[kk][n] = X[nb][c][ih][iw]（越界填 0）----
        // 线程的 n 固定（WG = BN），所以空间解码 (nb, oh, ow) 只做一次
        {
            const int n = tid % CS_BN;
            const int kkB = tid / CS_BN;   // = 0（CS_WG == CS_BN）
            const int sp = n0 + n;

            int nb = 0, oh = 0, ow = 0;
            const bool spOk = sp < SP;
            if (spOk) {
                nb = sp / HW;
                const int rem = sp % HW;
                oh = rem / OW;
                ow = rem % OW;
            }

            for (int t = 0; t < CS_BK; ++t) {
                const int kk = kkB + t * (CS_WG / CS_BN);
                const int gk = k0 + kk;
                float v = 0.0f;
                if (spOk && gk < KTOT) {
                    const int packed = Km[kk];
                    const int cc = packed >> 8;
                    const int rr = (packed >> 4) & 0xF;
                    const int ss = packed & 0xF;
                    const int ih = oh * SH - PT + rr;
                    const int iw = ow * SW - PL + ss;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        v = X[(((long)nb * C + cc) * H + ih) * W + iw];
                    }
                }
                Bs[kk * CS_BN + n] = v;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // ---- 寄存器分块累加（同 conv2d_gemm：float4 读 local）----
        //
        // 线程映射：work-group 是 (4, 16)，所以
        //   lx (0..3)  -> 输出通道方向，As 的下标用 lx*TM（CS_BM = 4*TM）
        //   ly (0..15) -> 空间方向，   Bs 的下标用 ly*TN（CS_BN = 16*TN）
        // 两个写反会越界读 local（ly 能到 15，而 CS_BM 只有 16）
        #pragma unroll
        for (int kk = 0; kk < CS_BK; ++kk) {
            union F4 {
                float4 v;
                float f[4];
            } av, bv;
            av.v = *(__local const float4*)(&As[kk * CS_BM + lx * CS_TM]);
            bv.v = *(__local const float4*)(&Bs[kk * CS_BN + ly * CS_TN]);
            #pragma unroll
            for (int i = 0; i < CS_TM; ++i) {
                #pragma unroll
                for (int j = 0; j < CS_TN; ++j) {
                    acc[i][j] += av.f[i] * bv.f[j];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ---- 写回 ----
    // 下标必须和上面的计算映射一致：lx -> 通道，ly -> 空间
    #pragma unroll
    for (int i = 0; i < CS_TM; ++i) {
        const int m = m0 + lx * CS_TM + i;
        if (m >= K) {
            continue;
        }
        const float b = (hasBias != 0) ? bias[m] : 0.0f;
        #pragma unroll
        for (int j = 0; j < CS_TN; ++j) {
            const int sp = n0 + ly * CS_TN + j;
            if (sp >= SP) {
                continue;
            }
            float v = acc[i][j] + b;
            if (activation == 1) {
                v = v / (1.0f + exp(-v));
            } else if (activation == 2) {
                v = fmax(v, 0.0f);
            } else if (activation == 3) {
                v = fmin(fmax(v, 0.0f), 6.0f);
            }
            const int nbw = sp / HW;
            const int remw = sp % HW;
            Y[((long)nbw * K + m) * HW + remw] = v;
        }
    }
}

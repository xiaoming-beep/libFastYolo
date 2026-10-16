# onnxcl 项目状态 / 交接文档

> **写于 2026-10-15。这份文档是项目的单一事实来源**，新会话读它就能直接继续，
> 不需要重新摸索。配套设计文档：[`onnxcl.md`](onnxcl.md)（讲设计取舍）。

---

## 0. 一句话目标

**用 OpenCL 直接跑 `.onnx`，不做任何格式转换**（不转 mnn / ort），
作为一个后端整合进 `libFastYolo`。

已完成。当前状态：**10 个模型全部跑通并数值验证通过。**

---

## 1. 当前状态总览

### 1.1 模型支持（全部 10 个）

| 模型 | 任务 | 数值验证 |
|---|---|---|
| yolo11n | DETECT | ✅ 余弦 1.00000000 |
| yolo26n (opset11) | DETECT | ✅ 300/300 |
| che.onnx（用户的真实模型） | DETECT | ✅ 余弦 1.00000000 |
| yolo11n-cls | CLASSIFY | ✅ 余弦 1.00000000 |
| yolo26n-cls | CLASSIFY | ✅ 余弦 1.00000000 |
| yolo11n-seg | SEGMENT | ✅ 两个输出都 1.00000000 |
| yolo26n-seg | SEGMENT | ✅ 掩码 1.0；检测 74/11400（并列）|
| yolo26n-depth | DEPTH | ✅ 归一化误差 0.005 |
| yolo26n-sem | SEMSEG | ✅ 1/1048576（1 个并列像素）|
| yolo26n-obb | OBB | ✅ 可运行（差异源于模型退化，见 §6）|

**"并列"是什么意思**：该模型的类别分数大量相等（obb 那个全是 0），
TopK 选哪 N 个本来就是任意的（ONNX 规范未定义 tie-break）。**不是引擎 bug**，
已用中间张量对拍证明两个引擎的分数完全一致。

### 1.2 性能（che.onnx，用户的模型）

```
onnxruntime CPU    33.8 ms
我们的引擎（JNI 墙钟） 23.16 ms     ← 快 1.5x
  ├─ Conv GPU      约 13.5 ms
  ├─ 其它算子 GPU   约 3 ms
  └─ host 开销      约 7 ms
初始化             673 ms（内核编译缓存命中）
```

---

## 2. 目录与文件位置

### 2.1 主目录（引擎本体，唯一活跃的开发位置）

```
C:\Users\Administrator\CLionProjects\libFastYolo\
├── CMakeLists.txt                      ← 加了 option(ONNXCL_BACKEND) + elseif 分支
├── docs\backend\onnxcl.md              ← 设计文档
├── docs\backend\onnxcl-engine.md       ← 本文（实现状态 / 交接）
├── source\backend\onnxcl\
│   ├── CMakeLists.txt                  ← 后端模块（option ONNXCL_TEST）
│   ├── yolo_onnxcl.cpp                 ← 适配层：4 个 yolo_infer_* C ABI
│   ├── gen_kernels.py                  ← 把 .cl 嵌进 cpp（含 MSVC 字面量分片）
│   ├── test_onnxcl.cpp                 ← 集成验证程序（支持 BGR 和原始张量两种输入）
│   ├── tools\verify_backend.py         ← 与 onnxruntime 对拍（支持非 f32 输出）
│   └── engine\                         ← 自包含引擎
│       ├── include\onnxcl\
│       │   ├── cl_api.h                ← OpenCL 运行时加载器（LoadLibrary/dlopen）
│       │   ├── cl_runtime.h            ← context/queue/内核缓存/编译产物缓存
│       │   ├── pb_reader.h             ← 通用 protobuf wire-format 读取器
│       │   ├── onnx_model.h            ← ONNX 结构
│       │   ├── graph.h                 ← 图 IR
│       │   └── executor.h              ← 执行器
│       ├── src\
│       │   ├── onnx\onnx_loader.cpp
│       │   ├── graph\{graph_build,shape_infer,fusion}.cpp
│       │   ├── cl\{cl_api,cl_runtime}.cpp
│       │   ├── exec\{executor,cpu_ops}.cpp     ← cpu_ops.cpp = CPU 回退算子
│       │   └── kernels\*.cl                    ← 8 个内核文件
│       └── third_party\OpenCLHeaders\
└── build-onnxcl\  build-jni\                   ← 构建目录（svn:ignore）
```

### 2.2 用户的测试工程（部署目标）

```
D:\works\codes\my_codes\MyYoloTest\
├── dll\                    ← 从 build-jni/bin/Release 拷进来（main() 从这里加载）
├── native\windows\         ← 同上（另一份副本）
├── model\                  ← che.onnx（用户真实模型）、yolo26n.onnx（opset11）
└── com\MyYoloTestApp.java  ← 用户的测试程序
```

### 2.3 分析/调试工具（在独立原型工程里）

```
D:\test\opencl\
├── REPORT.md               ← 早期：MNN 抽取可行性分析
├── DESIGN.md               ← 早期：引擎设计文档
├── onnxcl\                 ← 独立原型（旧引擎副本，主要用于工具）
│   ├── tools\analyze_onnx_ops.py       统计 ONNX 算子
│   ├── tools\analyze_onnx_structure.py 结构 + FLOPs + Conv 配置
│   ├── tools\analyze_fusion.py         融合机会 + 大小张量划分
│   ├── tools\cl_device_info.py         OpenCL 设备能力（ctypes）
│   ├── tools\cl_bandwidth.cpp          显存带宽实测
│   ├── tools\edge_cases_test.py        边界用例（Slice 步长/group conv/Mod）
│   └── tools\make_debug_model.py       ★ 把中间张量挂成输出，与 onnxruntime 对拍
└── tools\                  ← 更多分析脚本
```

### 2.4 临时数据

`D:\tmp\` —— 测试模型、输入张量、输出结果（`*_out0.bin`、`*_pixels.bin`）。

---

## 3. 构建与验证

### 3.1 构建

```bash
# MinGW（快，日常迭代用）
cd C:/Users/Administrator/CLionProjects/libFastYolo
cmake -S . -B build-onnxcl -G "MinGW Makefiles" -DONNXCL_BACKEND=ON -DONNXCL_TEST=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-onnxcl -j4

# MSVC + JNI（部署用）
cmake -S . -B build-jni -G "Visual Studio 17 2022" -A x64 \
      -DONNXCL_BACKEND=ON -DJNI_BACKEND=ON -DJAVA_HOME=D:/works/software/jdk/jdk21
cmake --build build-jni --config Release -j4
```

**注意事项**：
- **不需要 OpenCL SDK**（引擎走运行时加载）
- MinGW 产物的运行时 DLL 需要 `libgcc_s_seh-1.dll` / `libstdc++-6.dll` 在 PATH 或同目录
- 杀软会短暂锁住新生成的 exe（报 "Device or resource busy"），重试即可
- Python 的 `/tmp` 和 MSYS bash 的 `/tmp` **不是同一个目录**，脚本里要用显式路径

### 3.2 验证

```bash
# 跑模型并 dump 输出
./build-onnxcl/bin/test_onnxcl.exe <model.onnx> <bgr.raw> <w> <h> <输出前缀>
#   · 若输入文件字节数 == 模型输入元素数×4 -> 当原始 float32 张量用（合成用例）
#   · 否则当 BGR 图像走 Letterbox（集成测试）
#   · 环境变量 ONNXCL_PROFILE=1 打印逐算子 + 逐节点耗时

# 和一色 onnxruntime 对拍
python source/backend/onnxcl/tools/verify_backend.py <model.onnx> <输出前缀>

# 边界用例
python D:/test/opencl/onnxcl/tools/edge_cases_test.py
```

### 3.3 环境变量

| 变量 | 作用 |
|---|---|
| `ONNXCL_PROFILE=1` | 打开逐算子/逐层耗时统计（关闭时零开销） |
| `ONNXCL_NO_CACHE=1` | 关闭内核编译产物缓存（排查编译问题用） |
| `ONNXCL_CACHE_DIR=<dir>` | 指定缓存目录（默认 `%LOCALAPPDATA%\onnxcl_cache`） |
| `ONNXCL_KERNELS=<dir>` | 从磁盘读 .cl 而不是用内嵌版（改内核免重编） |
| `ONNXCL_CONV=naive\|gemm` | 强制卷积内核选择（对拍用） |
| `ONNXCL_REPEAT=N` | 重复推理 N 次。**抓「第二次才崩」类 bug 的必备手段**，也兼做微基准（核显单次噪声 ±15%，取最小值） |

---

## 4. 架构与关键设计决策

| 决策 | 理由 |
|---|---|
| **ONNX 直读，手写 protobuf** | 不引 libprotobuf（MinGW 下编译痛苦），只需读 6 个 message，约 160 行 |
| **NCHW 布局，不做 NC4HW4** | 和 ONNX 一致 → 图里不需要任何 layout 转换算子（省掉 MNN 那 412 个 ConvertTensor） |
| **权重按需重排 `[KH*KW][C][K]`** | 让 GEMM 的 A 子块全局读取合并。是 init 时的内存拷贝，**不是文件格式转换** |
| **视图算子零拷贝** | `Reshape/Flatten/Unsqueeze/Squeeze/Cast/Alias` 在图 IR 层做 alias 归并 |
| **后处理小算子走 CPU**（`cpu_ops.cpp`） | TopK/GatherElements/Gather/Expand/Tile/ArgMax 张量小、且大量用 int64 索引 |
| **SiLU 融合进 Conv epilogue** | `Mul(x, Sigmoid(x))` 是纯带宽受限，融合后省掉 150+ 次启动 |
| **OpenCL 运行时加载** | MSVC 无 OpenCL.lib；MinGW 可直链但为了统一 |
| **内核编译产物磁盘缓存** | AMD 编译器慢（8 个内核 ~3.5s）；缓存键含源码+选项+设备+驱动版本 |

### 4.1 内核清单（8 个 .cl 文件）

| 文件 | 内核 | 覆盖 |
|---|---|---|
| `conv.cl` | `conv2d` | 幼稚版（正确性参照 / depthwise / 兜底） |
| | `conv2d_gemm` | 大分块 GEMM（BM=64,BN=64,BK=16,TM=TN=4，work-group 16×16） |
| | `conv2d_gemm_small` | 小分块 GEMM（BM=16,BN=64,BK=32,TM=TN=4，work-group 4×16） |
| `elementwise.cl` | `binary_op` | Add/Sub/Mul/Div/Pow/Max/Min/Mod + **整数除法**（op code 8） |
| | `unary_op` / `clip_op` | Sigmoid/Exp/Cos/Sin/SiLU/... |
| `copy.cl` | `copy_strided` | **Concat/Split/Slice/Transpose** 统一归结为带步长块拷贝 |
| | `pixel_shuffle2x` | ConvTranspose(k=2,s=2,p=0) 的后半段 |
| `pool.cl` | `maxpool` / `global_avgpool` | |
| `resize.cl` | `resize_nearest` / `resize_bilinear` | |
| `softmax.cl` | `softmax` | 任意 axis（看成 `[outer, axisSize, inner]`） |
| `matmul.cl` | `batched_matmul` | MatMul **和 Gemm**（transB/C/alpha/beta） |
| `vector_add.cl` | `vector_add` / `vector_copy` / `vector_copy4` | 自检 + 带宽测量 |

### 4.2 算子分发（`executor.cpp` 的 `runNode`）

- 视图算子（含 `Cast`/`Constant`）→ 空操作
- `Conv` → `chooseConvKernel` 选 Naive/Gemm/GemmSmall
- 二元/一元 → 内核码分派（**注意 Div 对整数类型要走整数除法**）
- **`Constant` 必须显式跳过**（见 §6）
- 其余见 `runNode` 的 if 链

---

## 5. 性能现状与诊断结论（重要）

### 5.1 che.onnx 的逐层效率分布

```
效率区间      层数   总耗时ms   占比     区间等效GFLOPS
<15%          11    3.594   24.1%        120
15-30%        16    4.773   32.0%        367
30-45%        26    5.042   33.8%        593
>45%          10    1.503   10.1%        890
合计          63   14.912                 436    （峰值 1690 GFLOPS）
```

### 5.2 三条已证实的结论

1. **Winograd 不值得做**（对 che.onnx）
   只有 >45% 效率的那 10 层适用，占 10% 时间，**最多省 0.19 ms**。
   （3×3s1 占 64.9% FLOPs，但它们恰好都在高效率区间。）

2. **真正的问题是 27 个低效层（占 56% 时间）**，两类原因：
   - `[1,256,20,20]→[1,64,20,20]`：**只有 7 个 work-group**，GPU 90% 空转（效率 2.3%）
   - `[1,16,160,160]→[1,16,160,160]` / `[1,48,160,160]→[1,32,160,160]`：
     **im2col 的 B 子块加载开销 > 计算本身**（通道太少，每线程要扛 32 次带边界检查的加载）

3. **【已试过并回退】"并行度不足就换小分块"的判据**
   实测**慢 60%**（13.19 → 21.19 ms）。原因：小分块 work-group 只有 64 线程，
   每线程扛 32 次 B 子块加载（大分块只要 4 次），加载开销远超并行度收益。
   **代码里已留注释说明这条路不通。**

### 5.3 已落地的优化

| 优化 | 效果 |
|---|---|
| tiled implicit-GEMM 卷积 | 137.7 → ~31 ms |
| Conv+SiLU 融合（78 处） | GPU −31% |
| 关掉默认 profiling | 墙钟 −18ms |
| 权重按需重排 + float4 LDS | 23.8 → 21.9 ms |
| 首层卷积专用内核（cin=3） | 2.56 → 0.52 ms |
| Concat 走 DMA 拷贝 | 1.34 → 0.11 ms |
| 内核编译缓存 | 初始化 3.7s → 0.67s |
| `-cl-mad-enable` | 约 9%（且不减精度） |

### 5.4 下一步的正确方向

**不是**改分块尺寸，而是**降低每 work-group 的固定开销**：

1. **先建微基准**（单层重复跑 N 次取中位数）——
   当前单次测量噪声 ±15%，5% 的改动根本量不出来。**这是所有后续优化的前提。**
2. B 子块输入驻留 local memory（消除重复 gather）
3. 改成不依赖 im2col 的直接卷积（每个输入值读一次，寄存器内复用）
4. fp16（理论 2x，但要改全套）

---

## 6. 踩过的坑（完整清单）

### 编译/工具链

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | `.bat` 文件写 UTF-8 中文注释 | cmd.exe 按 GBK 读，注释被拆成乱码当命令执行 | **.bat 必须纯 ASCII** |
| 2 | MSVC 单字符串字面量上限（约 16380 字节） | `C2026: 字符串太大，已截断尾部字符` | gen_kernels.py 把源码切成多段相邻字面量 |
| 3 | `local` 是 OpenCL 保留关键字（地址空间限定符） | 内核编译报 `expected identifier or '('` | 变量改名 |
| 4 | **MSVC：定义上不能凭空加 `__declspec(dllexport)`** | `C2375: 重定义；不同的链接` | 和 MNN/RKNN 后端一致：定义上不加导出宏，交给构建系统；且定义要放 `namespace yolo` |
| 5 | MinGW 给产物加 `lib` 前缀 | Java `System.loadLibrary("FastYolo")` 找不到 | Windows 上加 `PREFIX ""` |
| 6 | MSVC 需要 `OpenCL.lib`（无 SDK 就没有） | 链接失败 | 改成运行时 `LoadLibrary` |
| 7 | `jni.h` 平台子目录写死 `include/linux` | Windows 编不过 | 按平台判断 win32/darwin/linux |
| 8 | 杀软锁住新生成的 exe | `Device or resource busy` | 等待重试 |
| 9 | Python 的 `/tmp` ≠ MSYS bash 的 `/tmp` | 脚本找不到刚生成的文件 | 用显式绝对路径 |

### 运行时/ABI

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 10 | **ABI 边界没包 try/catch** | 异常逸出 → 宿主进程 terminate（`0xC0000409`），且 stdout 块缓冲导致**崩溃时日志全丢** | 4 个 `yolo_infer_*` 全部包 try/catch；排查时加 `setvbuf(_IONBF)` |
| 11 | **单例在进程退出时析构** | 同样 `0xC0000409`（此时 OpenCL 运行时可能已卸载，且在 `DLL_PROCESS_DETACH` 持有 loader lock） | `ClRuntime::instance()` 改成**故意 leak**（`new` 出来不析构） |
| 12 | `java.library.path` 不解析依赖 DLL | `UnsatisfiedLinkError: Can't find dependent libraries` | 把 native 目录加进 `PATH` |
| 13 | 适配层只支持 float32 输出 | 分割类别图（uint8）被拒 | 加 `exportTensor` 按 ONNX 声明类型转换（uint8/int8/int32/int64） |
| 14 | 测试程序写文件假定 float32 | 遇 uint8 输出按 4 倍读 → **越界崩溃** | 按 `elem_type` 决定元素宽度 |
| 25 | **临时缓冲池的所有权契约没定清楚**：`acquireScratch` 把缓冲记入 `scratchPool_`，调用方却调 `rt.release(mid)`，**忘了删池子里的条目** | **第 1 次推理正常、第 2 次必崩**（use-after-free，退出码 139）。只在走 ConvTranspose 的模型（seg/depth）上出现 | 缓冲归池所有，**调用方不要 release**，析构时统一释放（见 `acquireScratch` 的注释）。已加回归测试 |

> **坑 25 为什么难抓**：它需要「同一进程内跑两次推理」才能复现 —— 单次推理的冒烟测试
> 永远是绿的。所以 `test_onnxcl` 和 Java 测试都加了重复推理参数（`ONNXCL_REPEAT` / 第 6 个命令行参数）。
> **教训：任何带缓存的资源（缓冲池/句柄/内核）都要问一句「第二次还会不会用对」。**

### 数值/语义（最重要的一类）

| # | 坑 | 症状 | 发现方式 |
|---|---|---|---|
| 15 | **`Div` 对整数输入必须是整数除法** | `188.6` vs 参考 `188` | **★ 中间张量对拍** |
| 16 | **`Gemm` 忽略 `transB`** | 输出**全是 NaN** | NaN 输出（N 算成 1280 而不是 1000 → **越界写**） |
| 17 | **`Gemm` 忽略偏置 `C`**（第三个输入） | 数值偏移 | 同上 |
| 18 | **`Constant` 节点没被跳过** | `不支持算子 Constant` | 边界用例（真实模型用 initializer，很多导出工具用 Constant） |
| 19 | `Tile`/`Gather` 形状推导缺失 | 构图失败 | 直接报错 |
| 20 | 空张量（`dims=[0]`，Resize 的 `roi`）被形状校验误杀 | 构图失败 | 改成「常量允许空张量、激活值不允许」 |
| 21 | TopK 并列行为不确定 | 同输入可能不同输出 | 代码审查 → 并列按索引升序确定化 |
| 22 | `Concat`/`Split` 的 offset 用原始 rank 下标索引 4 维补齐步长数组 | `cos` 只有 0.96（差 84 倍） | 对拍 |
| 23 | 融合判据写成「x 只有 1 个消费者」 | SiLU 融合从未生效（`Mul(x, Sigmoid(x))` 里 x 本来就有 2 个消费者） | 性能无提升 → 查 profile |
| 24 | 首层卷积分块浪费 21 倍 | 单层占 13.8% GPU 时间 | 逐层 profile |

### 调试方法论（★ 最有价值的资产）

**`tools/make_debug_model.py`** —— 把模型里任意中间张量挂成输出（非 f32 的自动插 Cast），
然后 onnxruntime 和 `test_onnxcl` 各跑一遍，**逐张量对拍**。

**坑 #15 是唯一必须靠它才能发现的**（语义错误，只看最终输出会误判成"并列问题"）。
以后查任何"数值对不上"都应该先上这个工具。

**诊断直觉**：
- **NaN 输出 ≈ 越界写**（不是"算得不准"）—— 坑 #16 就是靠这个直觉定位的
- **不要用 `max_rel` 当判据**，参考值接近 0 时无意义。用 `diff <= atol + rtol*|ref|`
- **测量噪声大时取最小值**（噪声源：核显共享内存、系统其它进程）

---

## 7. 关于"差异"的定性结论（避免重复调查）

| 现象 | 结论 |
|---|---|
| yolo26n-obb 有 41/300 行不同 | **不是 bug**。该模型类别分数**全是 0**（两个引擎一致），21504 个候选全部并列，TopK 选谁任意。已用中间张量对拍证明 |
| yolo26n-seg 检测输出 74/11400 不符 | 同上（并列） |
| yolo26n-sem 1/1048576 个像素不同 | 同上（ArgMax 遇到相等分数的并列像素） |

**判断方法**：如果"值"类中间张量完全一致、只有"索引/选择"类不同 → 就是并列问题。

---

## 8. 待办

### 8.1 文档（优先，已落后 3 轮）

[`onnxcl.md`](onnxcl.md) 需要补：
- 8 个新算子（TopK/GatherElements/Gather/Expand/Tile/ArgMax/Cast/ConvTranspose）
- CPU 回退通道的设计
- 边界支持（Slice 步长、group/depthwise conv、Constant 跳过、非 f32 输出）
- 性能诊断结论（§5）
- 本次新增的坑（§6 的 15-21）

### 8.2 性能（见 §5.4）

先建微基准，再做输入驻留 local / 直接卷积 / fp16。**Winograd 不要做。**

### 8.3 未覆盖的边界（都不影响现有 10 个模型）

| 项 | 现状 |
|---|---|
| `ConvTranspose` 其它参数（k≠2/s≠2/pad≠0） | 明确报错 |
| `Gemm` 的 `transA=1` | 明确报错 |
| `Slice` 超过 4 维 | 报错（`copy_strided` 是 4 维） |
| 索引 > 2²⁴ 的模型 | 引擎内部统一 f32，会丢精度（已在 `cpu_ops.cpp` 文件头注明） |
| 动态 shape / 控制流（If/Loop/Scan） | 设计上不支持 |

**全部是「遇到就报明确错误」，不会静默算错** —— 这是刻意保持的。

---

## 9. 快速上手（新会话第一步）

```bash
# 1. 确认现状能跑
cd C:/Users/Administrator/CLionProjects/libFastYolo
cmake --build build-onnxcl -j4
./build-onnxcl/bin/test_onnxcl.exe D:/test/opencl/test/yolo11n.onnx \
    D:/test/opencl/test_sample_bgr.raw 1280 720 D:/tmp/quick
python source/backend/onnxcl/tools/verify_backend.py \
    D:/test/opencl/test/yolo11n.onnx D:/tmp/quick
# 期望：余弦相似度 1.00000000

# 2. 看性能分布
ONNXCL_PROFILE=1 ./build-onnxcl/bin/test_onnxcl.exe \
    D:/works/codes/my_codes/MyYoloTest/model/che.onnx \
    D:/tmp/che_frame_bgr.raw 2560 1440 D:/tmp/prof

# 3. 部署（改了引擎之后）
cmake --build build-jni --config Release -j4
cp build-jni/bin/Release/{FastYolo.dll,yolo_onnxcl.dll} \
   /d/works/codes/my_codes/MyYoloTest/dll/
cp build-jni/bin/Release/{FastYolo.dll,yolo_onnxcl.dll} \
   /d/works/codes/my_codes/MyYoloTest/native/windows/
```

**改代码前请先跑一遍步骤 1**，确认基线是绿的再动手。

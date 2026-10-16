# onnxcl 后端方案

`onnx` 文件由 OpenCL 直接执行，**不需要转成 mnn / ort 等其他格式**，也不需要 ONNX Runtime
或 MNN 这类第三方推理库。

- 后端目录：`source/backend/onnxcl/`
- 引擎本体：`source/backend/onnxcl/engine/`（自包含，约 5000 行）
- 顶层开关：`-DONNXCL_BACKEND=ON`

---

## 1. 为什么做这个

现有 `onnx` 后端走 ONNX Runtime，优点是成熟、支持 EP 多；但要跑在 GPU 上得配 CUDA/cuDNN，
在只有核显（或不想装 CUDA）的开发机上用不上。MNN 后端则要求模型先经 `mnnconvert` 转一次格式，
每换一个模型就要再转一次。

本后端的目标很单一：**模型文件丢进来就能在 OpenCL 上跑，没有中间格式、没有 SDK 依赖。**

适用面刻意做窄了 —— 只支持 YOLO 系列（静态 shape、无控制流的算子图），
换来的是实现量小、可读、可改。

---

## 2. 编译

### Windows

```bat
cmake -S . -B build-onnxcl -G "MinGW Makefiles" -DONNXCL_BACKEND=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-onnxcl -j4
```

MSVC 同理（去掉 `-G`，用 Visual Studio 生成器）。

**不需要安装 OpenCL SDK。** 系统只要有显卡驱动自带的 OpenCL 运行时即可：
Windows 是 `C:\Windows\System32\OpenCL.dll`，Linux 是 `libOpenCL.so.1`。
原因见下面 §4.1。

### 和 JNI 一起编

```bash
cmake -S . -B build -DONNXCL_BACKEND=ON -DJNI_BACKEND=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

#### Windows（MSVC，已在本机验证）

```bat
cmake -S . -B build-jni -G "Visual Studio 17 2022" -A x64 ^
      -DONNXCL_BACKEND=ON -DJNI_BACKEND=ON ^
      -DJAVA_HOME=D:/works/software/jdk/jdk21
cmake --build build-jni --config Release -j4
```

产出 `build-jni/bin/Release/FastYolo.dll` + `yolo_onnxcl.dll`。

为此对 CMakeLists 做了两处修正（原来只适用于 Linux）：

1. **`jni.h` 的平台子目录**：原来是写死的 `include/linux`，Windows 上要 `include/win32`
   （macOS 是 `include/darwin`），现在按平台判断。
2. **jvm 库的位置**：Windows 用 `${JAVA_HOME}/lib/jvm.lib`（JDK 自带导入库，
   20/21 都带；MinGW 下如果发行版没发导入库则退回直接链 `bin/server/jvm.dll`）。
   Linux 是 `lib/server/libjvm.so`，同时兼顾一部分发行版的 `lib/<arch>/server`。

另外给 Windows 加了 `PREFIX ""`：Java 侧是 `System.loadLibrary("FastYolo")`，
Windows 下会去找 `FastYolo.dll`；MinGW 默认产出 `libFastYolo.dll`，Java 找不到。
（MSVC 本来就不加前缀，不受影响。）

#### ⚠️ 部署坑位：依赖 DLL 的搜索路径

`java.library.path` **只用来找 `FastYolo.dll` 本身，不用来解析它的依赖**。
`FastYolo.dll` 依赖 `yolo_onnxcl.dll`（后端是独立 DLL），而 Windows 解析依赖只搜
「宿主 exe 目录 + 系统目录 + 当前目录 + PATH」。直接跑会报：

```
UnsatisfiedLinkError: FastYolo.dll: Can't find dependent libraries
```

解决：把 native 目录加进 `PATH`（或放进 JVM 的 `bin` 目录）：

```bat
set "PATH=<native目录>;%PATH%"
java -Djava.library.path="<native目录>" -cp ... YourMain
```

> 注：`onnx` 后端依赖 `onnxruntime.dll` 是同一个问题，处理方式相同。

### 依赖

只有两个：**C++17 以上编译器**、**Python3**（仅构建期用，把 `.cl` 内核源码嵌进二进制）。
运行期零依赖（除了系统 OpenCL 运行时）。

---

## 3. 目录结构

```
source/backend/onnxcl/
├── CMakeLists.txt              后端模块（按后端拆分的老规矩）
├── yolo_onnxcl.cpp             适配层：实现 yolo_infer_create/init/predict/destroy 四个 C ABI
├── gen_kernels.py              构建期：把 .cl 内核源码嵌进 kernels_generated.cpp
├── test_onnxcl.cpp             集成验证程序（-DONNXCL_TEST=ON 时构建）
├── tools/
│   └── verify_backend.py       集成验证：和 onnxruntime 对拍
└── engine/                     引擎本体（自包含，不依赖本仓库其他部分）
    ├── include/onnxcl/
    │   ├── cl_api.h            OpenCL 运行时加载器（LoadLibrary/dlopen）
    │   ├── cl_runtime.h        context/queue/program 缓存/work-group 自适应
    │   ├── pb_reader.h         通用 protobuf wire-format 读取器
    │   ├── onnx_model.h        ONNX 结构定义
    │   ├── graph.h             图 IR
    │   └── executor.h          执行器
    ├── src/
    │   ├── onnx/onnx_loader.cpp       onnx.proto 6 个 message 的解析 + half/bf16/int 转换
    │   ├── graph/graph_build.cpp      ONNX -> IR、拓扑排序、常量物化
    │   ├── graph/shape_infer.cpp      各算子形状规则
    │   ├── graph/fusion.cpp           图融合（SiLU 折进 Conv）
    │   ├── cl/cl_api.cpp              OpenCL 运行时加载
    │   ├── cl/cl_runtime.cpp          设备/context/queue/内核编译
    │   ├── exec/executor.cpp          显存分配 + 逐算子调度
    │   └── kernels/*.cl               8 个 OpenCL 内核文件
    └── third_party/OpenCLHeaders/     OpenCL 头文件（自带，不需要 SDK）
```

---

## 4. 设计要点

### 4.1 OpenCL 走运行时加载，不依赖 OpenCL.lib

Windows + MSVC 下没有装 OpenCL SDK 就没有导入库（`OpenCL.lib`），系统只自带 `OpenCL.dll`。
MinGW 可以直链 DLL，MSVC 不行。Linux 也不是所有发行版默认带 `-dev` 包。

所以统一改成运行时加载（`LoadLibrary` / `dlopen`），好处：

- 任何编译器、任何平台都不需要 SDK，只要有系统 OpenCL 运行时就能编译和运行
- 运行时不在了能给明确的中文报错，而不是"加载库就失败"
- 和 MNN 的 `MNN_USE_LIB_WRAPPER` 是同一个思路

实现见 `engine/include/onnxcl/cl_api.h`：用一个 X-macro 维护函数指针表，
`clXxx(...)` 的调用靠宏重定向到指针。加新函数只要在 `ONNXCL_CL_FUNCS` 里加一行。

**坑位**：重定向宏必须放在 `cl_api.h`（所有使用方都能看到），不能只放
`cl_runtime.cpp` —— 因为 `cl_runtime.h` 里有 inline 的 `Kernel::setRaw()` 会调
`clSetKernelArg`，而 inline 函数是在使用方（如 `executor.cpp`）展开的。
加载器自身（`cl_api.cpp`）定义 `ONNXCL_CL_API_NO_REDIRECT` 跳过这段。

### 4.2 内存传模型字节，不是路径

`YOLOConfig::model_data` 是 `std::vector<char>`，接口给的就是模型字节。
引擎的 `ModelProto::loadFromMemory` 直接吃字节，和文件路径走同一条解析路径。

### 4.3 不跨 ABI 边界抛异常（**必须遵守**）

四个 `yolo_infer_*` 全部包 `try/catch`。这不是可选项：

C++ 异常一旦从 `extern "C"` 边界逸出，宿主进程直接 terminate。实测症状是
**主函数全部走完、"完成"也打印了，但进程退出码是 `0xC0000409`**；而且因为 stdout
是块缓冲，崩溃时整个缓冲区丢失，栈上什么日志都看不到，非常难查。
宿主是 JNI 时就是整个 JVM 挂掉。

### 4.4 单例故意 leak，不在进程退出时析构

`ClRuntime::instance()` 返回的是 `new` 出来的对象，不析构：

析构函数会去 `clReleaseKernel/Program/CommandQueue/Context`，而进程退出时
（1）OpenCL 运行时可能已经被卸载，函数指针就是野指针；（2）析构发生在
`DLL_PROCESS_DETACH` 里，此时持有 loader lock，在锁内调外部库很容易死锁或崩溃。

实测不加这一条就是上面那个 `0xC0000409`。进程退出时操作系统会回收全部资源（含显存），
leak 是安全且推荐的做法。

### 4.5 张量布局：NCHW + 权重重排

激活值内部统一 NCHW（和 ONNX 一致），所以图里**不需要任何 layout 转换算子**
（对比 MNN 需要 412 个 `ConvertTensor` 做 NCHW↔NC4HW4）。

权重在 init 时重排成 `[KH*KW][C][K]`，让卷积 GEMM 的 A 子块全局读取沿 k 方向连续。
这只是一次性的内存拷贝，**不是文件格式转换** —— onnx 文件本身始终是直读的。

### 4.6 只写了 8 个内核文件

模型分析显示算子形态很规整，所以做了最大限度的内核复用：

| 内核 | 覆盖的算子 | 关键设计 |
|---|---|---|
| `conv2d` | Conv（幼稚版，正确性参照 / depthwise / 兜底） | |
| `conv2d_gemm` | Conv（1×1 / 3×3 / s1 / s2 / group） | tiled implicit-GEMM，带 `activation` 参数供融合 |
| `binary_op` | Add / Sub / Mul / Div / Pow / Max / Min / Mod | **带 strides 的广播**，一个内核覆盖 same-shape / 标量 / 广播 |
| `unary_op` | Sigmoid / Exp / Cos / Sin / Tanh / Relu / SiLU … | op code 分派 |
| `copy_strided` | **Concat / Split / Slice / Transpose** | 全部归结为"带步长的块拷贝" |
| `maxpool` | MaxPool | |
| `resize_nearest` | Resize | |
| `softmax` | Softmax（任意 axis） | 看成 `[outer, axisSize, inner]` |
| `batched_matmul` | MatMul / Gemm | |

### 4.7 视图算子零拷贝

`Reshape` / `Flatten` / `Unsqueeze` / `Squeeze` 在图 IR 层做 alias 归并：
输出直接共享输入的显存，既不分配内存也不启动内核。

### 4.8 SiLU 融合

YOLO 里有大量 `Mul(x, Sigmoid(x))`。它是纯显存带宽受限的（一个张量读 2 遍写 1 遍），
融进 Conv 的 epilogue 后这部分显存往返直接归零。

判据要特别小心：**不能写成「x 只有 1 个消费者」** —— SiLU 的模式里 x 同时被
Sigmoid 和 Mul 读，本来就是 2 个消费者。正确条件是"这两个节点之外没人再读 x"。

---

## 5. 验证

### 集成验证（推荐）

```bash
# 1. 编译时带上验证程序
cmake -S . -B build-onnxcl -DONNXCL_BACKEND=ON -DONNXCL_TEST=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-onnxcl -j4

# 2. 准备输入：任意 JPEG -> raw BGR（避免在 C++ 里引解码器）
python -c "
from PIL import Image; import numpy as np
im = Image.open('your.jpg').convert('RGB').resize((1280,720), Image.BILINEAR)
open('sample_bgr.raw','wb').write(np.asarray(im,dtype=np.uint8)[:,:,::-1].tobytes())
"

# 3. 跑：真实走一遍「模型字节 -> 后端 -> Letterbox -> 输出」
./build-onnxcl/bin/test_onnxcl yolo11n.onnx sample_bgr.raw 1280 720 /tmp/t

# 4. 把 test_onnxcl dump 出来的 Letterbox 像素喂给 onnxruntime 对拍
python source/backend/onnxcl/tools/verify_backend.py yolo11n.onnx /tmp/t
```

判据用 `diff <= atol + rtol*|ref|`（`atol=rtol=1e-4`），
不是单纯的 `max_rel` —— 参考值接近 0 时相对误差没有意义。

### 最近一次结果（AMD Radeon 680M / gfx1035，yolo11n）

```
前处理输出: [1,3,640,640]（来自 libFastYolo 的 Letterbox）
onnxruntime 输出: [1,84,8400]

  余弦相似度     : 1.00000000
  最大绝对误差   : 1.529e-02   (出现在参考值 ≈340 处，相对误差 4.5e-05)
  平均绝对误差   : 4.341e-06
  归一化误差     : 0.445 (阈值 1.0)   超阈元素: 0/705600
  数值范围       : 与 onnxruntime 完全一致
```

误差量级就是 fp32 不同累加顺序的正常水平。

### 性能

```
端到端墙钟     : ~22 ms/帧 (640x640, ≈45 FPS)
内核启动次数   : 199 次
显存占用       : ~158 MB
初始化         : ~3.5 s（OpenCL 内核编译，一次性）
```

初始化那 3.5 秒是 AMD 的 OpenCL 编译器慢，不是 bug。**后续应该做编译产物
二进制缓存**（`clGetProgramInfo` 取 `CL_PROGRAM_BINARIES` 落盘，
按"源码 hash + 设备名"做 key）—— 符号已经在 `cl_api.h` 里预留了。

---

## 6. 已知限制

| 项 | 说明 |
|---|---|
| 任务类型 | 只验证了 **DETECT（yolo11n）**。其余 7 种任务（seg/pose/obb/cls/semseg/depth）形状推导已通过，但执行器还缺 `GatherElements`/`TopK`/`Expand`/`Mod`/`ArgMax`/`Cast`/`ConvTranspose` 等算子 |
| 输出类型 | 只支持 **float32 输出**。非 float32 会在 create 阶段明确报错，而不是给一个类型对不上的指针让上层错读 |
| 输入 | 只支持 float32 输入，且必须是静态 shape |
| 设备 | 只有 OpenCL 一条路。显式指定 CUDA/VULKAN/NPU 会报错（AUTO/CPU/ALL 都会走 OpenCL） |
| Slice | 只支持单位步长 |
| MatMul | 只处理 batch 维完全相同的情况 |
| 广播 | 只实现了 ONNX 实际用到的组合，遇到不支持会**报错而不是静默算错** |
| 显存 | 不做复用（yolo11n 峰值约 158 MB） |
| MinGW 产物 | `libFastYolo.dll` 等会带 `lib` 前缀，且运行时需要 `libgcc_s_seh-1.dll` / `libstdc++-6.dll` 在旁边（MSVC 和 Linux 无此问题） |

### 可能的后续优化方向

按性价比排序：

1. **内核编译产物落盘缓存** —— 直接干掉 3.5 秒初始化
2. **首层卷积（cin=3）专用内核** —— 它单独占 14% 的 GPU 时间（cin=3 时 GEMM 的分块浪费比达 21 倍，反而更慢）
3. **Concat 特化** —— 通道拼接本质是连续拷贝，现在走的是通用 strided copy
4. Conv 双缓冲 / 更激进的分块 —— 目前 583 GFLOPS（峰值约 1.7 TFLOPS 的 34%）

---

## 7. 排查记录（踩过的坑）

都是实际发生过的，记下来免得重犯：

1. **`cl_api.h` 里重定向宏的位置** —— 只放 `cl_runtime.cpp` 会导致 `executor.cpp`
   去链真的 `clSetKernelArg`（因为 `Kernel::setRaw` 是头文件里的 inline），
   而我们没有也不需要 `OpenCL.lib`，链接直接失败。

2. **X-macro 生成的成员名带 `cl` 前缀** —— `X(name) decltype(&::name) name`
   产出的是 `clGetPlatformIDs` 而不是 `GetPlatformIDs`，宏右值要写
   `::onnxcl::clApi.clXxx`。

3. **`HMODULE` 和 `void*` 不能隐式互转** —— 公共头不能引 `windows.h`，
   所以句柄只能存成 `void*`；加载辅助函数收 `void*`、内部再转 `HMODULE`。

4. **`.bat` 文件必须是纯 ASCII** —— cmd.exe 在中文 Windows 上按 GBK 读 `.bat`，
   UTF-8 的中文注释会被拆成乱码当命令执行（`set` 赋值失败、参数变空）。
   这个坑我踩了两次，写中文注释注释到 Windows 脚本上时要特别注意。

5. **MSVC：定义上不能凭空加 `__declspec(dllexport)`** ——
   `yolo.h` 里声明 `yolo_infer_*` 时没有 dllexport，如果定义上加一个，MSVC 会报：
   ```
   error C2375: "yolo::yolo_infer_create": 重定义；不同的链接
   ```
   GCC/MinGW 宽容地放过，所以是 MSVC 帮我们挑出来的问题。正确做法和 MNN/RKNN/
   openvino 后端一致：**定义上不加导出宏**，导出交给构建系统
   （MSVC 靠 `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON`；
   MinGW 没这个机制，要在 CMakeLists 里显式 `-Wl,--export-all-symbols`）。
   另一个细节：这四个函数要定义在 `namespace yolo` 里（和 `yolo.h` 的声明同作用域），
   而不是全局 `extern "C"`。

6. **`Letterbox` 的 `pixels` 是 planar RGB**（`RRR...GGG...BBB`，0~1 归一化），
   输入是 BGR uint8 —— `fill_bgr` 内部已经转好了，后端直接传 `pixels` 即可，不要再换通道。

7. **stdout 块缓冲会吃掉崩溃前的日志** —— 排查时给测试程序加
   `setvbuf(stdout, nullptr, _IONBF, 0)`，否则崩了什么都看不到。

8. **控制台中文乱码不用管** —— Windows 控制台默认 GBK，而程序输出 UTF-8。
   只是显示问题，可以 `chcp 65001` 或重定向到文件看。

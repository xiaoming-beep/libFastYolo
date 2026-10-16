# 故障排查汇总

> 症状 → 原因 → 处理。

## GPU / ONNX-CUDA

| 症状 | 原因 | 处理 |
|---|---|---|
| `[yolo_onnx] CUDA EP unavailable, fallback to CPU: ... Failed to load shared library` | 系统找不到 `libcudnn.so.9`（或驱动缺失） | 补 cuDNN（[目录式部署](build/python-sdk.md#4-onnx-cuda-的部署形态推荐目录式而非-slim-wheel)）；`ldd providers_cuda.so` 确认还缺什么 |
| Conv 节点报 `CUDNN_FE failure 11: CUDNN_BACKEND_API_FAILED`（GTX 10/16 系老卡） | cuDNN 9.x 新版本放弃 Pascal/Maxwell | 换 cuDNN 9.1（见[版本矩阵](appendix/version-matrix.md#3-cudnn-9-的显卡架构门槛)） |
| `Failed to load ... error: libcublasLt.so.11: cannot open shared object file` | 用了 CUDA 11 构建的 ORT 包（如官方 1.18 C++ 包） | 换 CUDA 12 构建（1.21–1.26） |
| pip 版 CUDA EP 警告 `Require cuDNN 9.* and CUDA 13.*` | onnxruntime-gpu ≥ 1.27 是 CUDA 13 构建 | 锁 `onnxruntime-gpu==1.26.*` 或升级 CUDA 13 |
| CUDA EP enabled 后报 `Invalid handle. Cannot load symbol cublasLtGetVersion` | ORT 进程内加载 cuBLAS/cuDNN 组合与老卡（Pascal）环境不匹配；GTX 1060 实测 1.23.2 / 1.24.4 / 1.29.0 均复现 | 先按版本矩阵锁 cuDNN 9.1；仍失败时用 CPU EP 验证逻辑（后处理与 EP 无关），GPU 场景换 RTX 20 系以上验证 |

## JNI / Java（Windows）

| 症状 | 原因 | 处理 |
|---|---|---|
| `onnxruntime.dll: 动态链接库(DLL)初始化例程失败` | JDK/bin 自带旧版 VC++ 运行时抢先加载 | 复制 `C:\Windows\System32\` 的 `vcruntime140[_1].dll`、`msvcp140[_1].dll` 到 `%JAVA_HOME%\bin\`（JVM 未运行时操作） |
| `FastYolo.dll: 找不到指定的模块/程序` | 三 DLL 不同目录，或缺 VC++ Redistributable | 全部放同一目录；装 VC++ 2022 Redist |
| `UnsatisfiedLinkError: Can't find dependent libraries` | `java.library.path` 不解析依赖 DLL | 把 native 目录加进 `PATH`（见 [Windows 编译](build/windows.md#4-dll-依赖与部署)） |
| `is not a valid Win32 application` | DLL 位数与 JVM 不匹配 | 统一 x64 |
| CMake 报 `JAVA_HOME not set` / `libjvm not found` | 未设 JAVA_HOME 或 JDK 不含 dev 组件 | `-DJAVA_HOME=...` |
| MSVC 编译 `yolo::STATUS_SUCCESS` 报 C2589 | `STATUS_SUCCESS` 宏与 winnt.h 冲突 | JNI 代码用字面量 `0`，勿写 `yolo::STATUS_XXX` |
| Java 调 `track(budgetIds, budgetAreas)` / `predict(…, needMask)` 报 `UnsatisfiedLinkError` | 新增 native 方法未写入 `javac -h` 生成头 → 符号被 C++ 改名（`_Z…`） | 按 [JNI 文档](build/jni.md#2-重新生成-jni-头文件)重新生成头并重编 |
| Git Bash 下 `dumpbin /exports` 无输出/报错 | MSYS 把 `/exports` 转成 Windows 路径 | 命令前加 `MSYS_NO_PATHCONV=1` |

## Python

| 症状 | 原因 | 处理 |
|---|---|---|
| `AttributeError: ... undefined symbol: fastyolo_create` | 构建未加 `-DPYTHON_BACKEND=ON`（或重新 configure 时被缓存覆盖为 OFF） | `cmake -S . -B <build> -DPYTHON_BACKEND=ON …` 后重编 |
| `python scripts\build_wheel.py --full` 报错找不到 openvino | `--full` 实现依赖本机 openvino 包 | 先 `pip install openvino`；GPU 部署走目录式 |
| 构建后 libs/ 里 CUDA 运行时消失 | `collect_artifacts.py` 清理与当前模式不匹配的文件 | 用 `--runtime full`（必要时加 `--cudnn-lib`）重新收集 |
| import 报"未找到 libFastYolo 动态库" | 未构建或 libs/ 被清空 | 重跑构建脚本 |

## 掩码 / SEGMENT

| 症状 | 原因 | 处理 |
|---|---|---|
| 掩码长度不再是 `imgW*imgH` / 下游数组越界 | 2026-09 起掩码为 ROI（长度 = 框 `width*height`），旧整幅假设代码未迁移 | 见 [迁移指南](guides/segment-lazy-migration.md)；索引公式 `(px-x)+(py-y)*width` |

## onnxcl（OpenCL）

| 症状 | 原因 | 处理 |
|---|---|---|
| 退出码 `0xC0000409` 且无日志 | ABI 边界异常逸出 / 单例在进程退出析构 | 确认 4 个 `yolo_infer_*` 均包 try/catch；`ClRuntime::instance()` 故意 leak；调试时 `setvbuf(stdout, nullptr, _IONBF, 0)` |
| MinGW 产物运行缺 DLL | 需要 `libgcc_s_seh-1.dll` / `libstdc++-6.dll` | 放 PATH 或同目录（MSVC 无此问题） |
| 初始化慢（AMD 约 3.5s） | OpenCL 内核编译一次性开销 | 正常；已支持内核编译产物缓存（`ONNXCL_NO_CACHE=1` 可关，`ONNXCL_CACHE_DIR` 可改） |
| 某算子在界面上报不支持 | 引擎刻意"遇到就报明确错误"而非静默算错 | 见 [onnxcl 已知限制](backend/onnxcl.md#6-已知限制) |

## 构建 / 版本控制

| 症状 | 原因 | 处理 |
|---|---|---|
| `postprocess.cpp` 等文件"缺失/异常"（SVN） | 本地未版本化同名文件 obstruct 了 `svn update` | 删本地挡路文件再 update；或从 `.svn/pristine/<sha1前两位>/<sha1>.svn-base` 还原 |
| Windows 构建/加载时"另一个程序正在使用此文件" | 杀软扫描大 DLL（WinError 32） | SDK loader 已内置指数退避重试，稍候重跑脚本 |
| `.bat` 脚本报语法错误/参数变空 | cmd.exe 按 GBK 读 `.bat`，UTF-8 中文注释被当命令 | **`.bat` 必须纯 ASCII** |
| GitHub 下载超时 | 网络不通 | 用 `https://gh-proxy.com/` 前缀；pip 走清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |

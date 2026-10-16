# MNN 后端

> 顶层开关：`-DMNN_BACKEND=ON`
> 适配层：`source/backend/MNN/yolo_mnn.cpp`。

MNN 支持 CPU / OpenCL / Vulkan / AUTO（经 `MapInferDevice()` 映射）。
模型需先用 MNN 自带的 `mnnconvert` 转成 `.mnn`（该后端内部仍以 ONNX 字节流接口对外，
转换细节以 MNN 版本为准）。换一个模型就要再转一次——这是 MNN 相对 onnxcl / ONNX 的额外步骤。

---

## 1. Linux

- 仓库自带 `source/backend/MNN/lib/libMNN.so` 为 **CPU 构建**（无 OpenCL/CUDA 符号）；
  `device=OPENCL/VULKAN` 需自行编译 MNN 并替换该 `.so`（如 `-DMNN_OPENCL=ON`）。
- 构建：

  ```bash
  cmake -S . -B build -DMNN_BACKEND=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)
  ```

- 非 CPU 设备会自动启用磁盘 cache（`./cachefile`），多实例并发时注意工作目录。

## 2. Windows（完整自编译流程，含 JNI）

> 来源：Windows MNN-JNI 编译记录（MNN 3.6.1 + VS2022 BuildTools）。

### 2.1 编译 MNN（启用 OpenCL）

```bash
cmake -G "Visual Studio 17 2022" -A x64 -B build \
  -DMNN_BUILD_DEMO=OFF -DMNN_BUILD_BENCHMARK=OFF -DMNN_BUILD_CONVERTER=OFF \
  -DMNN_BUILD_TRAIN=OFF -DMNN_SEP_BUILD=OFF -DMNN_OPENCL=ON -DMNN_USE_SYSTEM_LIB=OFF
cmake --build build --config Release -j
```

产物 `build/Release/MNN.dll + MNN.lib`（`MNN_SEP_BUILD=OFF` 所以是单库）。
**不需要 `OpenCL.lib`**：`MNN_USE_SYSTEM_LIB=OFF` 使 MNN 运行时 `LoadLibrary("OpenCL.dll")`，
编译期零依赖。

### 2.2 拷贝进仓库

```bash
cp MNN.lib MNN.dll → source/backend/MNN/lib/
```

（Linux 用的 `libMNN.so` 保留无妨。）

### 2.3 生成 JNI 头文件

见 [JNI 文档](../build/jni.md#2-重新生成-jni-头文件)。⚠️ 生成头文件必须用 **JDK ≥ 17**
（Java 工程是 17 + Lombok）。

### 2.4 构建 libFastYolo

此处 C++ 库用 JDK8 的 `jvm.lib` 链接（历史环境偏好，与生成头的 JDK 版本无关）：

```bash
export JAVA_HOME=/d/works/software/jdk/jdk8
cmake -G "Visual Studio 17 2022" -A x64 -B build-mnn-jni -DMNN_BACKEND=ON -DJNI_BACKEND=ON
cmake --build build-mnn-jni --config Release -j
```

### 2.5 收尾

`MNN.dll` 被 POST_BUILD 拷到 `build-mnn-jni/bin/`（注意**不在** `bin/Release/`），
手动 `cp bin/MNN.dll bin/Release/`，保证三 DLL 同目录：`FastYolo.dll → yolo_mnn.dll → MNN.dll`。

验证导出（Git Bash 下 `dumpbin` 的 `/exports` 会被 MSYS 当路径，加 `MSYS_NO_PATHCONV=1`）：

```bash
MSYS_NO_PATHCONV=1 dumpbin /exports FastYolo.dll | find /c "Java_com_fastyolo"   # 应 ≥ 15
```

# JNI / Java 绑定

> 顶层开关：`-DJNI_BACKEND=ON`（需 `JAVA_HOME`）。
> 实现：`source/jni/yolo_jni.cpp`、`source/jni/jni_utils.h`。

JNI 实现直接打进主库（`libFastYolo.so` / `FastYolo.dll`），不需要单独的后端 JNI 库。

---

## 1. 结构

- Java 工程：`java/fast-yolo-jni/`（Maven，`mvn clean package` → `target/fast-yolo-jni-1.0.0.jar`）。
  类：`InferDevice / YoloException / YoloConfig / YoloResult / Letterbox / YoloInference`
  （后两者持 native handle，`AutoCloseable`）。
- CMake 用 `JAVA_HOME`（env 或 `-DJAVA_HOME=` 传入）找 `include/` 与 `libjvm/jvm.lib`，
  找不到直接 FATAL_ERROR。平台差异（`jni_md.h` 子目录、jvm 库位置）已按
  win32 / darwin / linux 自动处理。
- 调用流程与线程模型：`Letterbox` 可跨帧复用；`predict/track` 结果坐标已映射回原图；
  实例**非线程安全**，多线程各自建实例。
- native 方法：`nativePredict / nativePredictMask / nativeTrack / nativeTrackMask`
  （后两者为 SEGMENT 掩码懒加载，见 [迁移指南](../guides/segment-lazy-migration.md)）。

---

## 2. 重新生成 JNI 头文件

**任何 native 方法签名变更后必须执行**，否则新符号会被 C++ 改名（`_Z…`），
Java 端报 `UnsatisfiedLinkError`。

```bash
# 生成头文件必须用 JDK ≥ 17（Java 工程 target=17 + Lombok，需 processorpath 挂 lombok jar）
JDK21_BIN=<jdk17+/bin>
LOMBOK=lombok-<与pom一致>.jar   # 正式做法：mvn dependency:copy -Dartifact=org.projectlombok:lombok:<ver>
SRC=java/fast-yolo-jni/src/main/java
"$JDK21_BIN/javac" -h source/include/fast_yolo -cp "$LOMBOK" -processorpath "$LOMBOK" \
  -d /tmp/jclasses "$SRC"/com/fastyolo/*.java
# 生成 com_fastyolo_Letterbox.h / com_fastyolo_YoloInference.h，与 fast_yolo_jni.h 一同入库
```

⚠️ 头文件必须与**提交版 Java 源码**一致；签名变了而没重新生成，运行时会
`GetMethodID` / `UnsatisfiedLinkError`。另外 5 个数据类
（TaskType / Keypoint / ClassifyResult / SemSegResult / DepthResult）的字段构造器签名
必须匹配 `source/jni/jni_utils.h` 中的 JNI 签名字符串。

---

## 3. Java 侧加载顺序

依赖链：`FastYolo → yolo_* → 后端运行时`。

```java
// Windows / ONNX CPU：按序显式加载
System.load(dir + "/onnxruntime.dll");
System.load(dir + "/yolo_onnx.dll");
System.load(dir + "/FastYolo.dll");

// Linux ONNX CPU 静态版：只需加载主库（onnxruntime 已静态链入，
// libyolo_onnx.so 经 $ORIGIN 自动解析）
System.load("/path/libFastYolo.so");

// Linux ONNX-CUDA 自包含版：同样只需加载主库，但整个目录必须含
// libonnxruntime*.so、providers_cuda/shared、（可选）libcudnn*.so.9 或系统提供 cuDNN；
// jvm.dll 在 Java 进程内已存在，无需加载。

// Windows onnxcl / MNN：Java 只加载主库，但依赖 DLL 要靠 PATH 解析（见 Windows 编译文档）
```

---

## 4. 编译命令

```bash
# Linux
cmake -S . -B build -DJNI_BACKEND=ON -DJAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
      -DONNX_BACKEND=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

```bat
:: Windows（MSVC）
cmake -S . -B build-msvc -G "Visual Studio 17 2022" -A x64 ^
      -DJNI_BACKEND=ON -DJAVA_HOME=D:/works/software/jdk/jdk21
cmake --build build-msvc --config Release -j
```

# Windows 编译

> 通用前提、VS 生成器用法、DLL 依赖。**后端特有参数见 [backend/](../backend/README.md)。**

---

## 1. 环境要求

- **VS2022（或 BuildTools）+ "使用 C++ 的桌面开发"工作负载**。
- **VC++ 2022 Redistributable** 必须安装——缺失时 `onnxruntime.dll` 拒绝加载。
- JDK（仅 JNI 时）：`set JAVA_HOME=D:\...\jdk21`。
  ⚠️ JDK 自带 VC++ 运行时可能过旧导致 onnxruntime 初始化失败，处理见 [故障排查](../troubleshooting.md)。
- Python 3（仅 onnxcl 后端构建期需要，用于把 `.cl` 内核源码嵌进二进制）。

---

## 2. 通用构建

```bat
:: 多配置 VS 生成器：CMAKE_BUILD_TYPE 无效，靠 --config
cmake -B build-msvc -G "Visual Studio 17 2022" -DJNI_BACKEND=ON
cmake --build build-msvc --config Release -j
```

产物在 `build-msvc\bin\Release\`（Debug 为 `bin\Debug\`，主库名带 `d` 后缀，如 `FastYolod.dll`）。

MinGW 也可（`-G "MinGW Makefiles"`），注意 MinGW 默认给产物加 `lib` 前缀，仓库在 Windows
上统一去掉了（`PREFIX ""`），否则 Java `System.loadLibrary("FastYolo")` 找不到。

### CLion 配置

Settings → CMake 添加配置，Generator 选 `Visual Studio 17 2022`，Options 填
`-DJNI_BACKEND=ON` 等即可。

---

## 3. 后端选择（Windows 可用）

| 后端 | 开关 | 文档 |
|---|---|---|
| ONNX Runtime CPU | 默认 | [onnx.md](../backend/onnx.md#4-windows--cpu) |
| OpenVINO（Python SDK 默认） | `-DOPENVINO_BACKEND=ON` + `-DOPENVINO_ROOT=` | [openvino.md](../backend/openvino.md#2-windowspython-sdk-默认路线) |
| MNN | `-DMNN_BACKEND=ON`（需自编 MNN） | [mnn.md](../backend/mnn.md#2-windows完整自编译流程含-jni) |
| onnxcl（OpenCL） | `-DONNXCL_BACKEND=ON` | [onnxcl.md](../backend/onnxcl.md) |

> ONNX-CUDA 与 RKNN 不支持 Windows。

---

## 4. DLL 依赖与部署

产物按依赖链依次加载：`FastYolo.dll → yolo_<后端>.dll → 后端运行时`。**三件套必须同目录**。

| 后端 | 需要同目录的 DLL |
|---|---|
| ONNX | `FastYolo.dll` + `yolo_onnx.dll` + `onnxruntime.dll` |
| OpenVINO | `FastYolo.dll` + `yolo_openvino.dll` + OpenVINO 运行时 |
| MNN | `FastYolo.dll` + `yolo_mnn.dll` + `MNN.dll` |
| onnxcl | `FastYolo.dll` + `yolo_onnxcl.dll`（运行时只需系统 `OpenCL.dll`） |

⚠️ `java.library.path` **只用来找 `FastYolo.dll` 本身，不用来解析它的依赖**。直接跑可能报：

```
UnsatisfiedLinkError: FastYolo.dll: Can't find dependent libraries
```

解决：把 native 目录加进 `PATH`（或放进 JVM 的 `bin` 目录）：

```bat
set "PATH=<native目录>;%PATH%"
java -Djava.library.path="<native目录>" -cp ... YourMain
```

---

## 5. Python SDK 一把梭

```bat
python -m pip install cmake openvino
python\scripts\build_windows.bat
```

见 [Python SDK](python-sdk.md)。

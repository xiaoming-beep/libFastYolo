# TensorRT 后端

> 顶层开关：`-DTENSORRT_BACKEND=ON`
> 适配层：`source/backend/TensorRT/yolo_tensorrt.cpp`。

TensorRT 只支持 CUDA，且**模型必须是序列化的 engine**（后端直接 `deserializeCudaEngine`），
不能直接吃 `.onnx`。engine 绑定 TRT 版本与具体显卡，换卡 / 升级 TRT 需重新生成。

---

## 1. 依赖

- 本机 CUDA toolkit（CMake 自动搜 `/usr/local/cuda*`）
- TensorRT SDK（`/usr/src/tensorrt` 或系统路径的 `nvinfer` / `nvinfer_plugin`）

## 2. 编译（Linux）

```bash
cmake -S . -B build -DTENSORRT_BACKEND=ON -DJNI_BACKEND=ON \
      -DJAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## 3. 模型转换

先用 `trtexec` 把 ONNX 转成 engine：

```bash
trtexec --onnx=yolo11n.onnx --saveEngine=yolo11n.engine
```

## 4. 说明

- Windows 理论可用但**未在仓库验证**。
- 嫌 TensorRT 安装重时：Linux GPU 场景优先考虑 **ONNX-CUDA**（同为 N 卡加速，免装 TRT、
  模型免转换），见 [onnx.md](onnx.md#3-linux--cuda-gpu默认开启)。

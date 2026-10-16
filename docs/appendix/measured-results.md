# 本机实测记录

## 1. ONNX-CUDA（2026-09-01/02）

**环境**

- Ubuntu，NVIDIA driver 580.173.02，CUDA 12.8（`/usr/local/cuda-12.8`），
  系统另有 cuDNN 8.9.7（本方案用不到）。
- 构建：`cmake -S . -B build-onnx-gpu -DPYTHON_BACKEND=ON`
  → 日志 `Using ONNX Runtime GPU (CUDA EP)`。

**验证**（`python/examples/demo.py --device CUDA`，真实摄像头照片，yolo11n）

- `CUDA` 与 `CPU` 两种 device 的检测结果逐位一致（boat 0.654 / cake 0.546，坐标相同）；
- 系统缺 cuDNN9 时自动回退 CPU（日志含 fallback 原因），推理不中断；
- `providers_cuda.so` 打 `$ORIGIN` + cuDNN 9.1 同目录后，**零环境变量自包含运行成功**。

**Python 侧冒烟**（ORT 1.21 wheel 交叉验证）：

- yolo11n 10.3ms/帧、yolo11n-seg 12.4ms/帧、yolo26n 9.9ms/帧（GTX 1060 5GB，640×640）；
- 与 CPU 输出 maxdiff ≤ 2e-3（浮点级），yolo26 输出仅 TopK 排序差异。

`collect_artifacts.py --backend onnx --runtime full --cudnn-lib <dir>` 一条命令收出完整 GPU 部署包。

## 2. SEGMENT 掩码懒加载升级（2026-09-10）

**环境**

- Hygon C86-3G（8C16T，AVX2）、GTX 1060 5GB、ORT 1.23.2（CUDA 12 构建；GPU EP 因
  `cublasLtGetVersion` 兼容问题未能在该老卡启用，CPU EP 下完成全部后处理测量——
  后处理为纯 CPU，结论与 EP 无关）。

**纯后处理基准**（`MyFastYolo/bench_postonly`）

| 项 | 结果 |
|---|---|
| 懒帧（decode + NMS） | 0.52–0.54ms/帧（≈1900fps 上限，0.027 核@50fps） |
| ROI 掩码帧（全量） | face 1 目标 2.46ms、4K 1 大目标 4.37ms、4K 4 目标 4.43ms |
| `findContours` 净耗时（ROI） | face +0.20ms、4K 4 目标 +0.38ms（旧整幅 4 目标 14.9ms） |
| 模拟轨迹（每 40 帧评估 3 帧） | 4K 4 目标平均 0.81ms/帧（含轮廓 0.84ms）≈0.041 核@50fps；旧实现 ~30ms/帧 ≈1.5 核@50fps |

**正确性**

- ROI 掩码非零像素与旧整幅掩码逐位一致（face 183379）；
- `MyFastYolo/test_lazy_track` 预算语义 4 场景通过；
- Python SDK 实跑（懒帧 / `need_masks=True` / track 连续性）通过；
- Java → JNI 联调（懒帧、预算 {0}、预算 {best}、无预算、`predict(needMask)`）5 场景通过。

## 3. onnxcl（AMD Radeon 680M / gfx1035，yolo11n）

```
前处理输出: [1,3,640,640]（来自 libFastYolo 的 Letterbox）
onnxruntime 输出: [1,84,8400]

  余弦相似度     : 1.00000000
  最大绝对误差   : 1.529e-02   (出现在参考值 ≈340 处，相对误差 4.5e-05)
  平均绝对误差   : 4.341e-06
  归一化误差     : 0.445 (阈值 1.0)   超阈元素: 0/705600
  数值范围       : 与 onnxruntime 完全一致
```

**性能**：端到端 ~22 ms/帧（640×640，≈45 FPS）；内核启动 199 次；显存 ~158 MB；
初始化 ~3.5s（AMD OpenCL 编译器慢，已加编译产物缓存）。

**che.onnx（用户真实模型）**：onnxruntime CPU 33.8ms，本引擎（JNI 墙钟）23.16ms（快 1.5×），
初始化 673ms（缓存命中）。

> 当前状态：**10 个模型全部跑通并数值验证**（yolo11n / yolo26n / che / cls / seg / depth / sem / obb）。
> 细节见 [onnxcl.md](../backend/onnxcl.md) 与 [onnxcl-engine.md](../backend/onnxcl-engine.md)。

## 4. RKNN（RK3576，NPU 850MHz，i8，640×640，bus.jpg，2026-09-15）

| 模型 | 图 | 全流水（含解码+NMS） | 备注 |
|---|---|---|---|
| **yolov8n** | 优化图 9 输出 + int8 直读 | **48.1 FPS** | 5 目标，分数与官方 demo `process_i8` 完全一致 |
| yolo11n | 优化图 9 输出 + int8 直读 | 43.9 FPS | |
| yolov8n | 标准图（strip sigmoid） | 38.6 FPS | 兜底路线 |
| yolo26n-seg | 标准图（strip sigmoid） | 32.2 FPS | SEGMENT 自动识别 |
| yolo26n | 标准图（原生） | 42.8 FPS | 框为归一化坐标，解码适配待做 |

详见 [RKNN 后端](../backend/rknn.md)。

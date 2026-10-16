# SEGMENT 掩码懒加载 + ROI 迁移指南（2026-09）

> 影响范围：`predict` / `track` 的 SEGMENT 掩码 API（C++ / JNI / Python 三套）。
> 旧"整幅原图分辨率掩码"语义**已废弃**，下游代码必须迁移。

---

## 1. 背景

4K 输入是 OCR 下游的硬需求。旧实现每帧为每个保留框生成**整幅原图分辨率掩码**
（4K 单框 8.8MB 清零 + 写回），叠加下游"整幅掩码找轮廓/连通域"，实测 4K 4 目标后处理
≈30ms/帧（50fps 下 ≈1.5 个核心）。而掩码的真实用途往往只是为每条轨迹挑一张"面积最大"
的抠图，平台期/远离期帧的掩码根本不影响结果。

## 2. 新设计（要点）

1. **两阶段**：每帧只跑 `decode_segment_boxes`（解码 + NMS，≈0.5ms"懒帧"）；
   `build_segment_masks` 只对"确实要消费"的框生成 ROI 掩码（"掩码帧"）。
2. **MaskBudget 预算表**：调用方每帧传 `{trackId → 已保留 best 抠图面积}`，库内 ByteTrack
   之后精确判定：**无记录（新目标）或 `框面积 > bestArea`** 才生成；框面积是抠图面积的
   严格上界，跳过无损、无帧滞后。被触发框的重叠跟踪框一并生成（保住重叠仲裁语义）。
3. **ROI 掩码**：长度 = 框 `width*height`，行跨距 = 框宽，原点 = 框左上角；旧整幅掩码
   框外恒 0，故无信息损失（已逐位验证）。
4. **API 不兼容变更**：`predict(…, need_masks)` 默认只出框；`track(…, budgets, n)` 预算驱动；
   `mask_w/mask_h` 等整幅假设全部移除。

## 3. 实测（纯后处理，与推理设备无关）

本机 Hygon 8C16T / ORT 1.23.2：

| 场景 | 懒帧 | 掩码帧（ROI 全量） | +findContours(ROI) | 模拟轨迹（每 40 帧评估 3 帧） |
|---|---|---|---|---|
| face 1202×823，1 目标 | 0.54ms | 2.46ms | +0.20ms | 0.68ms（0.034 核@50fps） |
| 4K，1 大目标 | 0.52ms | 4.37ms | +0.35ms | 0.81ms（0.041 核@50fps） |
| 4K，4 目标 | 0.52ms | 4.43ms | +0.38ms | 0.81ms（0.041 核@50fps） |

对照旧实现（4K 4 目标）：掩码 15.0ms + 整幅轮廓 14.9ms ≈ **30ms/帧 → 1.5 核@50fps**。
ROI 无损校验：face 场景非零像素 183379 与旧整幅完全一致。

---

## 4. 掩码索引

ROI 掩码长度 = `width * height`，行跨距 = `width`，原点 = 框左上角 `(x, y)`：

```
// 原图点 (px, py) 对应掩码下标
idx = (px - x) + (py - y) * width;   // 0 表示框外/背景，1 表示前景
```

旧代码若用 `idx = px + py * imgW`（整幅假设），必须改成上面的公式。

---

## 5. C++ 迁移

```cpp
// ---- 旧：predict 总是出整幅掩码 ----
auto dets = yolo.predict(img);                 // det.mask.size() == imgW*imgH

// ---- 新：默认只出框 ----
auto dets = yolo.predict(img);                 // det.mask.empty()

// 离线/单帧需要掩码：显式要
auto dets = yolo.predict(img, /*need_masks=*/true);   // ROI 掩码

// ---- 在线逐帧：预算表驱动（推荐）----
std::vector<int>    budgetIds   = { /* 每条已跟踪目标 id */ };
std::vector<float>  budgetAreas = { /* 已保留 best 抠图面积 */ };
auto tracked = yolo.track(img, budgetIds.data(), budgetAreas.size(),
                          budgetAreas.data());   // 只对需要消费的框生成 ROI 掩码
```

> `track()` 默认只出框；掩码/关键点/角度载荷与检测框一致保留。

## 6. Java / JNI 迁移

```java
// 旧
List<YoloResult> r = yolo.predict(img);                 // 掩码整幅

// 新：显式要掩码
List<YoloResult> r = yolo.predict(img, /*needMask=*/true);   // ROI

// 在线逐帧：预算表驱动
int[]   ids    = { /* track ids */ };
float[] areas  = { /* best 面积 */ };
List<YoloResult> t = yolo.track(img, ids, areas);
```

⚠️ 签名变更后必须按 [JNI 文档](../build/jni.md#2-重新生成-jni-头文件)重新生成
`com_fastyolo_YoloInference.h`，否则报 `UnsatisfiedLinkError`。

## 7. Python 迁移

```python
# 旧：mask 是原图分辨率
dets = yolo.predict(img)
for d in dets:
    print(d.mask.shape)          # (imgH, imgW)

# 新：默认不出掩码
dets = yolo.predict(img)         # d.mask is None / 空

# 离线取掩码
dets = yolo.predict(img, need_masks=True)
for d in dets:
    h, w = d.mask.shape           # ROI：= (box.height, box.width)
    # 原图点 (px, py) -> d.mask[py - d.box.y, px - d.box.x]
```

Python 无预算表 API，在线逐帧场景请走 JNI 预算表路径或自行在 Python 侧按
`need_masks` 决策。

## 8. 下游集成（ctfo-ship-pixel / ctfo-ship-common）

合入补丁（SegCutout / TypeCropCollector / common Java 类 + 调用序说明）位于
`libFastYolo/.working/pixel-lazy-patch/`（该目录为工作副本，随升级同步）。

---

## 9. 升级检查清单

- [ ] 下游所有读取掩码的代码改成 ROI 索引公式（第 4 节）；
- [ ] 需要掩码的调用显式传 `need_masks=True` / `needMask=true`；
- [ ] 在线逐帧场景改用 `track(..., budgets)` 预算表；
- [ ] 重新生成 JNI 头文件（若 Java native 方法签名有变）；
- [ ] 以 `-DPYTHON_BACKEND=ON` 重新构建 Python SDK 并重出 wheel；
- [ ] 跑三语言冒烟（Python 懒帧 / `need_masks=True` / track 连续性；Java 懒帧、预算 {0}、
      预算 {best}、无预算、`predict(needMask)`）。

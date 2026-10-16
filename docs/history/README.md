# 历史编译笔记归档说明

本 README 早期版本整合了仓库原有的多份分散编译笔记，原文件已并入 `docs/` 各专题文档，
这里仅作来源说明与去留记录。

| 归档文件 | 原内容去向 |
|---|---|
| `编译指南.md` | [build/linux.md](../build/linux.md)、[build/windows.md](../build/windows.md)、[appendix/version-matrix.md](../appendix/version-matrix.md)、[appendix/measured-results.md](../appendix/measured-results.md) |
| `jni编译.txt` | [build/jni.md](../build/jni.md)（头文件生成、加载顺序、Windows VC++ 运行时坑） |
| `mnn-jni-windows编译上下文.md` | [backend/mnn.md](../backend/mnn.md)（Windows MNN + OpenCL 完整流程） |
| `python安装.md` | [build/python-sdk.md](../build/python-sdk.md) + [`python/README.md`](../../python/README.md) |
| `nms测试环境上下文.md` | [guides/api.md](../guides/api.md)、[appendix/measured-results.md](../appendix/measured-results.md) |
| `ai-session.md` | [guides/api.md](../guides/api.md)（三层抽象、ByteTrack、关键类） |
| `日志.txt` | 纯构建日志，已删除 |

> 注：历史文档中的 `jni方案.md` 在仓库中已不存在；JNI 接口与调用流程以
> [build/jni.md](../build/jni.md) 与 [guides/segment-lazy-migration.md](../guides/segment-lazy-migration.md) 为准。

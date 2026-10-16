/*
 * libFastYolo JNI 实现
 *
 * 将 C++ yolo::YOLOInference / yolo::Letterbox 的完整功能暴露给 Java 层.
 *
 * 文件结构:
 *   - Letterbox 相关:   4 个函数   (nativeCreate / Update / FillBGR / Destroy)
 *   - YoloInference 相关: 10 个函数 (Create / Init / InputShape / Task / Predict /
 *                                   Classify / SemSeg / Depth / Track / Destroy)
 *
 * 依赖:
 *   - fast_yolo_jni.h         JNI 函数声明（javac -h 生成，聚合头）
 *   - fast_yolo/yolo.h        公共 C++ API (YOLOInference, Letterbox, YOLOConfig, YOLOResult)
 *   - fast_yolo/img_util.h    图像工具 (read_bin_file 等)
 *   - jni_utils.h             JNI 辅助 (异常抛出、结果类构造缓存)
 */
// ---- 关键：yolo.h 必须先于 <jni.h> 引入 ----
// MinGW 的某些头会间接定义 STATUS_SUCCESS 等 NTSTATUS 宏，
// 与 yolo.h 中的 #define 冲突。先引入 yolo.h 可以确保
// STATUS_SUCCESS=0 等值被正确设置。
#include <fast_yolo/yolo.h>
#include <fast_yolo/img_util.h>

#include <fast_yolo/fast_yolo_jni.h>  // ← 包含 <jni.h>
#include "jni_utils.h"

#include <cstring>
#include <memory>
#include <vector>

// ============================================================================
// 全局缓存 — 避免每次 JNI 调用都查找字段/方法 ID
// ============================================================================

/** Letterbox Java 字段 ID 缓存（用于 nativeUpdate 后同步字段到 Java） */
struct LetterboxFields {
    jfieldID dstWidth;
    jfieldID dstHeight;
    jfieldID imgWidth;
    jfieldID imgHeight;
    bool initialized = false;
};

static LetterboxFields g_lbFields;

/** 惰性初始化 Letterbox 字段 ID */
static bool initLetterboxFields(JNIEnv *env, jobject thiz) {
    if (g_lbFields.initialized) return true;

    jclass clazz = env->GetObjectClass(thiz);
    if (!clazz) return false;

    g_lbFields.dstWidth  = env->GetFieldID(clazz, "dstWidth",  "I");
    g_lbFields.dstHeight = env->GetFieldID(clazz, "dstHeight", "I");
    g_lbFields.imgWidth  = env->GetFieldID(clazz, "imgWidth",  "I");
    g_lbFields.imgHeight = env->GetFieldID(clazz, "imgHeight", "I");

    env->DeleteLocalRef(clazz);

    if (!g_lbFields.dstWidth || !g_lbFields.dstHeight
        || !g_lbFields.imgWidth || !g_lbFields.imgHeight) {
        return false;
    }
    g_lbFields.initialized = true;
    return true;
}

// ============================================================================
// com_fastyolo_Letterbox — JNI 实现
// ============================================================================

/**
 * 创建 C++ yolo::Letterbox 实例.
 *
 * @param boxSize 目标正方形边长（默认 640）
 * @param color   填充颜色 [0, 255]（默认 114 灰色）
 * @param centerCrop true 为中心裁剪模式（分类任务用，不填充）
 * @return C++ 对象指针（转为 jlong），失败返回 0
 */
JNIEXPORT jlong JNICALL
Java_com_fastyolo_Letterbox_nativeCreate(JNIEnv *env, jclass /*clazz*/,
                                          jint boxSize, jfloat color, jboolean centerCrop) {
    auto *lb = new (std::nothrow) yolo::Letterbox(static_cast<int>(boxSize),
                                                  static_cast<float>(color),
                                                  centerCrop == JNI_TRUE);
    if (!lb) {
        throwYoloException(env, -2);
        return 0;
    }
    return reinterpret_cast<jlong>(lb);
}

/**
 * 调用 Letterbox::update()，计算缩放比例和 padding，
 * 并将 C++ 侧计算结果同步回 Java 字段.
 */
JNIEXPORT void JNICALL
Java_com_fastyolo_Letterbox_nativeUpdate(JNIEnv *env, jobject thiz,
                                          jlong handle, jint width, jint height) {
    auto *lb = reinterpret_cast<yolo::Letterbox *>(handle);
    if (!lb) {
        throwYoloException(env, -3);
        return;
    }

    // 执行 C++ update
    lb->update(static_cast<int>(width), static_cast<int>(height));

    // 同步字段到 Java 层
    if (!initLetterboxFields(env, thiz)) return;

    env->SetIntField(thiz, g_lbFields.dstWidth,  static_cast<jint>(lb->dst_width));
    env->SetIntField(thiz, g_lbFields.dstHeight, static_cast<jint>(lb->dst_height));
    env->SetIntField(thiz, g_lbFields.imgWidth,  static_cast<jint>(lb->img_width));
    env->SetIntField(thiz, g_lbFields.imgHeight, static_cast<jint>(lb->img_height));
}

/**
 * 调用 Letterbox::fill_bgr()，
 * 将 Java 侧传入的 BGR uint8 字节转为 planar float 像素并做 letterbox 填充.
 */
JNIEXPORT void JNICALL
Java_com_fastyolo_Letterbox_nativeFillBGR(JNIEnv *env, jobject /*thiz*/,
                                           jlong handle, jbyteArray data) {
    auto *lb = reinterpret_cast<yolo::Letterbox *>(handle);
    if (!lb) {
        throwYoloException(env, -3);
        return;
    }

    // 获取字节数组的原始指针（可能拷贝也可能直接映射，取决于 JVM 实现）
    jbyte *bytes = env->GetByteArrayElements(data, nullptr);
    if (!bytes) return;

    // 调用 C++ 填充函数（自动选择 horizontal / vertical / center_crop 路径）
    lb->fill_bgr(reinterpret_cast<const uint8_t *>(bytes));

    // JNI_ABORT = 不拷贝回 Java 侧（只读访问）
    env->ReleaseByteArrayElements(data, bytes, JNI_ABORT);
}

/**
 * 通过 native 指针地址调用 Letterbox::fill_bgr()，零拷贝路径.
 *
 * 适用于 OpenCV Mat::dataAddr() 等场景，避免 byte[] 的 GetByteArrayElements 开销.
 * 调用方保证 dataPtr 在调用期间有效，且指向 dstWidth * dstHeight * 3 字节的 BGR 数据.
 */
JNIEXPORT void JNICALL
Java_com_fastyolo_Letterbox_nativeFillBGRAddr(JNIEnv *env, jobject /*thiz*/,
                                               jlong handle, jlong dataPtr) {
    auto *lb = reinterpret_cast<yolo::Letterbox *>(handle);
    if (!lb) {
        throwYoloException(env, -3);
        return;
    }
    if (!dataPtr) {
        throwYoloException(env, -1);
        return;
    }

    lb->fill_bgr(reinterpret_cast<const uint8_t *>(dataPtr));
}

/**
 * 释放 C++ yolo::Letterbox 对象.
 */
JNIEXPORT void JNICALL
Java_com_fastyolo_Letterbox_nativeDestroy(JNIEnv * /*env*/, jclass /*clazz*/,
                                           jlong handle) {
    delete reinterpret_cast<yolo::Letterbox *>(handle);
}


// ============================================================================
// com_fastyolo_YoloInference — JNI 实现
// ============================================================================

/**
 * 在 C++ 堆上构造 yolo::YOLOConfig + yolo::YOLOInference.
 *
 * 从 jbyteArray 拷贝模型数据到 std::vector<char>（一次性开销），
 * 设置所有配置字段后创建推理引擎.
 *
 * @return C++ YOLOInference* 转为 jlong，失败返回 0
 */
JNIEXPORT jlong JNICALL
Java_com_fastyolo_YoloInference_nativeCreate(JNIEnv *env, jclass /*clazz*/,
                                              jbyteArray modelData,
                                              jfloat minScore, jfloat iouThreshold,
                                              jint device, jint gpuId,
                                              jint numThreads, jint precision,
                                              jstring inputName, jstring outputName,
                                              jint trackFrameRate, jint trackBuffer,
                                              jfloat trackThresh, jfloat highThresh,
                                              jfloat matchThresh,
                                              jint task, jint numClasses,
                                              jint numKeypoints, jint topK) {
    // ---- 1. jbyteArray → std::vector<char> ----
    jsize modelLen = env->GetArrayLength(modelData);
    if (modelLen <= 0) {
        throwYoloException(env, -1);
        return 0;
    }

    std::vector<char> modelVec(static_cast<size_t>(modelLen));
    env->GetByteArrayRegion(modelData, 0, modelLen,
                            reinterpret_cast<jbyte *>(modelVec.data()));

    // ---- 2. 构造 YOLOConfig ----
    auto config = std::make_shared<yolo::YOLOConfig>(
        modelVec,
        static_cast<float>(minScore),
        static_cast<float>(iouThreshold));

    // 设备与精度
    config->device      = static_cast<yolo::InferDevice>(device);
    config->gpu_id      = static_cast<int>(gpuId);
    config->num_threads = static_cast<int>(numThreads);
    config->precision   = static_cast<int>(precision);

    // 任务配置
    config->task         = static_cast<yolo::TaskType>(task);
    config->num_classes  = static_cast<int>(numClasses);
    config->num_keypoints = static_cast<int>(numKeypoints);
    config->top_k        = static_cast<int>(topK);

    // 输入/输出节点名
    const char *inName = env->GetStringUTFChars(inputName, nullptr);
    config->input_name  = std::string(inName);
    env->ReleaseStringUTFChars(inputName, inName);

    const char *outName = env->GetStringUTFChars(outputName, nullptr);
    config->output_name = std::string(outName);
    env->ReleaseStringUTFChars(outputName, outName);

    // ByteTrack 跟踪器参数
    config->track_frame_rate = static_cast<int>(trackFrameRate);
    config->track_buffer     = static_cast<int>(trackBuffer);
    config->track_thresh     = static_cast<float>(trackThresh);
    config->high_thresh      = static_cast<float>(highThresh);
    config->match_thresh     = static_cast<float>(matchThresh);

    // ---- 3. 构造 YOLOInference ----
    auto *infer = new (std::nothrow) yolo::YOLOInference(config);
    if (!infer) {
        throwYoloException(env, -2);
        return 0;
    }

    return reinterpret_cast<jlong>(infer);
}

/**
 * 调用 YOLOInference::init() 初始化后端引擎.
 *
 * @return 状态码: STATUS_SUCCESS(0) 成功，负数表示错误
 */
JNIEXPORT jint JNICALL
Java_com_fastyolo_YoloInference_nativeInit(JNIEnv *env, jobject /*thiz*/,
                                            jlong handle) {
    auto *infer = reinterpret_cast<yolo::YOLOInference *>(handle);
    if (!infer) {
        throwYoloException(env, -3);
        return -3;
    }

    int code = infer->init();
    if (code != 0) {
        throwYoloException(env, code);
    }
    return code;
}

/**
 * 返回模型输入形状（int[]，init 后有效）.
 */
JNIEXPORT jintArray JNICALL
Java_com_fastyolo_YoloInference_nativeInputShape(JNIEnv *env, jobject /*thiz*/,
                                                  jlong handle) {
    auto *infer = reinterpret_cast<yolo::YOLOInference *>(handle);
    if (!infer) {
        throwYoloException(env, -3);
        return nullptr;
    }
    const auto &shape = infer->get_input_shape();
    jintArray arr = env->NewIntArray(static_cast<jsize>(shape.size()));
    if (arr && !shape.empty()) {
        std::vector<jint> tmp(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
            tmp[i] = static_cast<jint>(shape[i]);
        }
        env->SetIntArrayRegion(arr, 0, static_cast<jsize>(shape.size()), tmp.data());
    }
    return arr;
}

/**
 * 返回解析后的任务类型 native 值（AUTO 时按输出形状推断）.
 */
JNIEXPORT jint JNICALL
Java_com_fastyolo_YoloInference_nativeTask(JNIEnv *env, jobject /*thiz*/,
                                            jlong handle) {
    auto *infer = reinterpret_cast<yolo::YOLOInference *>(handle);
    if (!infer) {
        throwYoloException(env, -3);
        return 0;
    }
    return static_cast<jint>(infer->resolve_task());
}

/**
 * 内部辅助: 将 C++ vector<YOLOResult> 转为 Java YoloResult[].
 */
static jobjectArray buildResultArray(JNIEnv *env,
                                      const std::vector<yolo::YOLOResult> &results) {
    if (!initYoloResultCache(env)) return nullptr;

    YoloResultCache &cache = getYoloResultCache();
    jobjectArray arr = env->NewObjectArray(
        static_cast<jsize>(results.size()), cache.clazz, nullptr);
    if (!arr) return nullptr;

    for (size_t i = 0; i < results.size(); ++i) {
        jobject obj = buildYoloResult(env, results[i]);
        if (jniHasException(env)) return nullptr;
        env->SetObjectArrayElement(arr, static_cast<jsize>(i), obj);
        env->DeleteLocalRef(obj);
    }
    return arr;
}

/**
 * 执行目标检测 / 实例分割 / 姿态 / 旋转框（按任务类型返回附带数据）.
 *
 * 流程: Letterbox → 后端推理 → 任务解码 → 坐标反算回原图 → 构建 Java YoloResult[].
 *
 * scoreThresh / iouThresh < 0 时使用 config 中的默认值.
 */
JNIEXPORT jobjectArray JNICALL
Java_com_fastyolo_YoloInference_nativePredict(JNIEnv *env, jobject /*thiz*/,
                                               jlong handle, jlong letterboxHandle,
                                               jfloat scoreThresh, jfloat iouThresh) {
    auto *infer = reinterpret_cast<yolo::YOLOInference *>(handle);
    auto *lb    = reinterpret_cast<yolo::Letterbox *>(letterboxHandle);

    if (!infer || !lb) {
        throwYoloException(env, -3);
        return nullptr;
    }

    int code;
    auto results = infer->predict(*lb, &code,
                                   static_cast<float>(scoreThresh),
                                   static_cast<float>(iouThresh));

    if (code != 0) {
        throwYoloException(env, code);
        return nullptr;
    }

    return buildResultArray(env, results);
}

/**
 * 图像分类（仅 CLASSIFY 任务）.
 */
JNIEXPORT jobject JNICALL
Java_com_fastyolo_YoloInference_nativeClassify(JNIEnv *env, jobject /*thiz*/,
                                                jlong handle, jlong letterboxHandle) {
    auto *infer = reinterpret_cast<yolo::YOLOInference *>(handle);
    auto *lb    = reinterpret_cast<yolo::Letterbox *>(letterboxHandle);

    if (!infer || !lb) {
        throwYoloException(env, -3);
        return nullptr;
    }

    int code;
    const yolo::ClassifyResult r = infer->classify(*lb, &code);
    if (code != 0) {
        throwYoloException(env, code);
        return nullptr;
    }
    if (!initDenseResultCache(env)) return nullptr;
    return buildClassifyResult(env, r);
}

/**
 * 语义分割（仅 SEMSEG 任务），返回原图分辨率类别图.
 */
JNIEXPORT jobject JNICALL
Java_com_fastyolo_YoloInference_nativeSemSeg(JNIEnv *env, jobject /*thiz*/,
                                              jlong handle, jlong letterboxHandle) {
    auto *infer = reinterpret_cast<yolo::YOLOInference *>(handle);
    auto *lb    = reinterpret_cast<yolo::Letterbox *>(letterboxHandle);

    if (!infer || !lb) {
        throwYoloException(env, -3);
        return nullptr;
    }

    int code;
    const yolo::DenseResult r = infer->semseg(*lb, &code);
    if (code != 0) {
        throwYoloException(env, code);
        return nullptr;
    }
    if (!initDenseResultCache(env)) return nullptr;
    return buildSemSegResult(env, r);
}

/**
 * 深度估计（仅 DEPTH 任务），返回原图分辨率 float32 深度图.
 */
JNIEXPORT jobject JNICALL
Java_com_fastyolo_YoloInference_nativeDepth(JNIEnv *env, jobject /*thiz*/,
                                             jlong handle, jlong letterboxHandle) {
    auto *infer = reinterpret_cast<yolo::YOLOInference *>(handle);
    auto *lb    = reinterpret_cast<yolo::Letterbox *>(letterboxHandle);

    if (!infer || !lb) {
        throwYoloException(env, -3);
        return nullptr;
    }

    int code;
    const yolo::DenseResult r = infer->depth(*lb, &code);
    if (code != 0) {
        throwYoloException(env, code);
        return nullptr;
    }
    if (!initDenseResultCache(env)) return nullptr;
    return buildDepthResult(env, r);
}

/**
 * 显式请求掩码的 predict（SEGMENT：needMask=true 时对全部保留框生成 ROI 掩码；
 * false 只出框——懒加载默认）。其余任务忽略该标志。
 */
JNIEXPORT jobjectArray JNICALL
Java_com_fastyolo_YoloInference_nativePredictMask(JNIEnv *env, jobject /*thiz*/,
                                                   jlong handle, jlong letterboxHandle,
                                                   jfloat scoreThresh, jfloat iouThresh,
                                                   jboolean needMask) {
    auto *infer = reinterpret_cast<yolo::YOLOInference *>(handle);
    auto *lb    = reinterpret_cast<yolo::Letterbox *>(letterboxHandle);

    if (!infer || !lb) {
        throwYoloException(env, -3);
        return nullptr;
    }

    int code;
    auto results = infer->predict(*lb, &code,
                                   static_cast<float>(scoreThresh),
                                   static_cast<float>(iouThresh),
                                   needMask == JNI_TRUE);

    if (code != 0) {
        throwYoloException(env, code);
        return nullptr;
    }

    return buildResultArray(env, results);
}

/**
 * 预算表驱动的 track（SEGMENT 掩码懒加载）：budgetIds/budgetAreas 平行数组，
 * 第 i 项 = {trackId, 已保留 best 抠图面积}；null 数组 = 整帧只出框。
 * 掩码只对"无 best 记录或框面积 > bestArea"的跟踪框（及其重叠框）生成。
 */
JNIEXPORT jobjectArray JNICALL
Java_com_fastyolo_YoloInference_nativeTrackMask(JNIEnv *env, jobject /*thiz*/,
                                                 jlong handle, jlong letterboxHandle,
                                                 jfloat scoreThresh, jfloat iouThresh,
                                                 jintArray budgetIds, jfloatArray budgetAreas) {
    auto *infer = reinterpret_cast<yolo::YOLOInference *>(handle);
    auto *lb    = reinterpret_cast<yolo::Letterbox *>(letterboxHandle);

    if (!infer || !lb) {
        throwYoloException(env, -3);
        return nullptr;
    }

    std::vector<yolo::MaskBudget> budgets;
    if (budgetIds != nullptr && budgetAreas != nullptr) {
        const jsize nIds = env->GetArrayLength(budgetIds);
        const jsize nAreas = env->GetArrayLength(budgetAreas);
        const jsize n = nIds < nAreas ? nIds : nAreas;
        if (n > 0) {
            std::vector<jint> ids(static_cast<size_t>(n));
            std::vector<jfloat> areas(static_cast<size_t>(n));
            env->GetIntArrayRegion(budgetIds, 0, n, ids.data());
            env->GetFloatArrayRegion(budgetAreas, 0, n, areas.data());
            budgets.reserve(static_cast<size_t>(n));
            for (jsize i = 0; i < n; ++i) {
                budgets.push_back({ids[static_cast<size_t>(i)], areas[static_cast<size_t>(i)]});
            }
        }
    }

    int code;
    auto results = infer->track(*lb, &code,
                                 static_cast<float>(scoreThresh),
                                 static_cast<float>(iouThresh),
                                 budgets.empty() ? nullptr : budgets.data(),
                                 static_cast<int>(budgets.size()));

    if (code != 0) {
        throwYoloException(env, code);
        return nullptr;
    }

    return buildResultArray(env, results);
}

/**
 * 执行检测 + ByteTrack 多目标跟踪（仅 DETECT 任务）.
 *
 * 流程: predict() → ByteTrack update（两级关联）→ 回填 track_id.
 * 首次调用时惰性创建 ByteTrack 跟踪器.
 */
JNIEXPORT jobjectArray JNICALL
Java_com_fastyolo_YoloInference_nativeTrack(JNIEnv *env, jobject /*thiz*/,
                                             jlong handle, jlong letterboxHandle,
                                             jfloat scoreThresh, jfloat iouThresh) {
    auto *infer = reinterpret_cast<yolo::YOLOInference *>(handle);
    auto *lb    = reinterpret_cast<yolo::Letterbox *>(letterboxHandle);

    if (!infer || !lb) {
        throwYoloException(env, -3);
        return nullptr;
    }

    int code;
    auto results = infer->track(*lb, &code,
                                 static_cast<float>(scoreThresh),
                                 static_cast<float>(iouThresh));

    if (code != 0) {
        throwYoloException(env, code);
        return nullptr;
    }

    return buildResultArray(env, results);
}

/**
 * 销毁 C++ YOLOInference 对象（含后端引擎上下文 + ByteTrack 跟踪器）.
 */
JNIEXPORT void JNICALL
Java_com_fastyolo_YoloInference_nativeDestroy(JNIEnv * /*env*/, jclass /*clazz*/,
                                               jlong handle) {
    delete reinterpret_cast<yolo::YOLOInference *>(handle);
}

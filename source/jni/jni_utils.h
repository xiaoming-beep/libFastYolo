/*
 * JNI 辅助工具 — 缓存类引用、抛出异常、错误码映射.
 *
 * 所有 JNI native 方法共用这些辅助函数，
 * 避免每次调用都 FindClass / GetMethodID.
 *
 * 前置依赖（调用方 .cpp 必须先 include）:
 *   #include <jni.h>                 JNI 类型定义 (JNIEnv, jclass, jobject...)
 *   #include <fast_yolo/yolo.h>      yolo 命名空间 (YOLOResult, STATUS_* 宏)
 */
#ifndef LIBFASTYOLO_JNI_UTILS_H
#define LIBFASTYOLO_JNI_UTILS_H
// 不在此处 include <jni.h> 和 <fast_yolo/yolo.h>，
// 因为 Windows 上 <jni.h> 间接引入的 <winnt.h> 会定义 STATUS_SUCCESS 等
// NTSTATUS 宏，与 yolo.h 的 #define 冲突。
// 由调用方 .cpp 在正确的顺序下（先 jni → undef Windows 宏 → 再 yolo）引入.

// ==================== Java 异常抛出 ====================

/**
 * 将 C++ 侧状态码转为 Java YoloException 并抛出.
 * 调用后 native 方法必须立即返回——后续 JNI 调用行为未定义.
 *
 * @param env JNI 环境指针
 * @param code C++ 侧状态码（STATUS_SUCCESS 时不抛异常）
 */
static inline void throwYoloException(JNIEnv *env, int code) {
    // code == 0 即 STATUS_SUCCESS，不抛异常
    if (code == 0) return;

    // 查找 YoloException 类和构造器
    jclass exClass = env->FindClass("com/fastyolo/YoloException");
    if (!exClass || env->ExceptionCheck()) return;

    //  构造器签名: (ILjava/lang/String;)V
    jmethodID ctor = env->GetMethodID(exClass, "<init>", "(ILjava/lang/String;)V");
    if (!ctor || env->ExceptionCheck()) {
        env->DeleteLocalRef(exClass);
        return;
    }

    // 不使用 yolo::STATUS_* 宏名（会与 MinGW ntstatus.h 冲突），直接用字面量
    const char *msg;
    switch (code) {
        case -1:  msg = "模型未找到"; break;       // STATUS_MODEL_NOT_FOUND
        case -2:  msg = "模型初始化失败"; break;     // STATUS_MODEL_INIT_FAILED
        case -3:  msg = "引擎尚未初始化"; break;     // STATUS_NOT_INITIALIZED
        case -4:  msg = "推理失败"; break;           // STATUS_INFERENCE_FAILED
        case -5:  msg = "任务不支持该操作"; break;   // STATUS_TASK_UNSUPPORTED
        default:  msg = "未知 native 错误"; break;
    }

    jstring jmsg = env->NewStringUTF(msg);
    jobject ex = env->NewObject(exClass, ctor, code, jmsg);
    env->Throw(static_cast<jthrowable>(ex));

    // 清理本地引用
    env->DeleteLocalRef(jmsg);
    env->DeleteLocalRef(ex);
    env->DeleteLocalRef(exClass);
}

/**
 * 在 C++ 侧检查 JNI 是否有待处理异常.
 * 如果有，返回 true（调用方应尽快返回）.
 */
static inline bool jniHasException(JNIEnv *env) {
    return env->ExceptionCheck() == JNI_TRUE;
}

// ==================== YoloResult 对象构造 ====================

/**
 * 缓存 YoloResult / Keypoint 类的构造器引用，避免每次推理都 FindClass.
 */
struct YoloResultCache {
    jclass clazz;        // com/fastyolo/YoloResult 全局引用
    jmethodID ctor10;    // (IIIIIFI[Lcom/fastyolo/Keypoint;[BF)V — 10 参数（含任务附带数据）
    jmethodID ctor7;     // (IIIIIFI)V — 7 参数构造器（含 trackId，无附带数据）
    jmethodID ctor6;     // (IIIIIF)V  — 6 参数构造器（trackId=-1）
    jclass keypointClazz; // com/fastyolo/Keypoint 全局引用
    jmethodID keypointCtor; // (FFF)V
    bool initialized;
};

inline YoloResultCache &getYoloResultCache() {
    static YoloResultCache cache{};
    return cache;
}

/**
 * 初始化 YoloResult 类缓存（在 JNI 函数中首次调用时惰性初始化）.
 * 返回 true 表示成功.
 */
static inline bool initYoloResultCache(JNIEnv *env) {
    YoloResultCache &c = getYoloResultCache();
    if (c.initialized) return true;

    jclass local = env->FindClass("com/fastyolo/YoloResult");
    if (!local || env->ExceptionCheck()) return false;

    c.clazz = static_cast<jclass>(env->NewGlobalRef(local));
    env->DeleteLocalRef(local);

    c.ctor10 = env->GetMethodID(c.clazz, "<init>", "(IIIIIFI[Lcom/fastyolo/Keypoint;[BF)V");
    c.ctor7 = env->GetMethodID(c.clazz, "<init>", "(IIIIIFI)V");
    c.ctor6 = env->GetMethodID(c.clazz, "<init>", "(IIIIIF)V");

    jclass kpLocal = env->FindClass("com/fastyolo/Keypoint");
    if (!kpLocal || env->ExceptionCheck()) return false;
    c.keypointClazz = static_cast<jclass>(env->NewGlobalRef(kpLocal));
    env->DeleteLocalRef(kpLocal);
    c.keypointCtor = env->GetMethodID(c.keypointClazz, "<init>", "(FFF)V");

    if (!c.ctor10 || !c.ctor7 || !c.ctor6 || !c.keypointCtor) return false;

    c.initialized = true;
    return true;
}

/**
 * 从 C++ YOLOResult 构造 Java YoloResult 对象（含 POSE/SEGMENT/OBB 附带数据）.
 * 必须在 initYoloResultCache() 成功后调用.
 */
static inline jobject buildYoloResult(JNIEnv *env, const yolo::YOLOResult &r) {
    YoloResultCache &c = getYoloResultCache();

    // POSE：关键点数组
    jobject keypointArray = nullptr;
    if (!r.keypoints.empty()) {
        keypointArray = env->NewObjectArray(static_cast<jsize>(r.keypoints.size()),
                                            c.keypointClazz, nullptr);
        if (keypointArray) {
            for (size_t i = 0; i < r.keypoints.size(); ++i) {
                const auto &k = r.keypoints[i];
                jobject kp = env->NewObject(c.keypointClazz, c.keypointCtor, k.x, k.y, k.score);
                env->SetObjectArrayElement(static_cast<jobjectArray>(keypointArray),
                                           static_cast<jsize>(i), kp);
                env->DeleteLocalRef(kp);
            }
        }
    }

    // SEGMENT：原图分辨率二值掩码
    jbyteArray maskArray = nullptr;
    if (!r.mask.empty()) {
        maskArray = env->NewByteArray(static_cast<jsize>(r.mask.size()));
        if (maskArray) {
            env->SetByteArrayRegion(maskArray, 0, static_cast<jsize>(r.mask.size()),
                                    reinterpret_cast<const jbyte *>(r.mask.data()));
        }
    }

    // OBB：旋转角（弧度）
    const bool has_payload = keypointArray != nullptr || maskArray != nullptr;
    jobject obj;
    if (has_payload) {
        obj = env->NewObject(c.clazz, c.ctor10,
                             r.x, r.y, r.width, r.height, r.clsId, r.score, r.track_id,
                             keypointArray, maskArray, r.angle);
    } else {
        obj = env->NewObject(c.clazz, c.ctor7,
                             r.x, r.y, r.width, r.height, r.clsId, r.score, r.track_id);
    }
    if (keypointArray) env->DeleteLocalRef(keypointArray);
    if (maskArray) env->DeleteLocalRef(maskArray);
    return obj;
}

// ==================== ClassifyResult / SemSegResult / DepthResult ====================

/**
 * 缓存稠密任务结果类（ClassifyResult / SemSegResult / DepthResult）.
 */
struct DenseResultCache {
    jclass classifyClazz;    // com/fastyolo/ClassifyResult
    jmethodID classifyCtor;  // (IFI[I[F[F)V
    jclass semsegClazz;      // com/fastyolo/SemSegResult
    jmethodID semsegCtor;    // (II[I)V
    jclass depthClazz;       // com/fastyolo/DepthResult
    jmethodID depthCtor;     // (II[FFF)V
    bool initialized;
};

inline DenseResultCache &getDenseResultCache() {
    static DenseResultCache cache{};
    return cache;
}

static inline bool initDenseResultCache(JNIEnv *env) {
    DenseResultCache &c = getDenseResultCache();
    if (c.initialized) return true;

    jclass local = env->FindClass("com/fastyolo/ClassifyResult");
    if (!local || env->ExceptionCheck()) return false;
    c.classifyClazz = static_cast<jclass>(env->NewGlobalRef(local));
    env->DeleteLocalRef(local);
    c.classifyCtor = env->GetMethodID(c.classifyClazz, "<init>", "(IFI[I[F[F)V");

    local = env->FindClass("com/fastyolo/SemSegResult");
    if (!local || env->ExceptionCheck()) return false;
    c.semsegClazz = static_cast<jclass>(env->NewGlobalRef(local));
    env->DeleteLocalRef(local);
    c.semsegCtor = env->GetMethodID(c.semsegClazz, "<init>", "(II[I)V");

    local = env->FindClass("com/fastyolo/DepthResult");
    if (!local || env->ExceptionCheck()) return false;
    c.depthClazz = static_cast<jclass>(env->NewGlobalRef(local));
    env->DeleteLocalRef(local);
    c.depthCtor = env->GetMethodID(c.depthClazz, "<init>", "(II[FFF)V");

    if (!c.classifyCtor || !c.semsegCtor || !c.depthCtor) return false;
    c.initialized = true;
    return true;
}

/** int 向量 → jintArray（失败返回 nullptr） */
static inline jintArray toJIntArray(JNIEnv *env, const std::vector<int> &v) {
    jintArray arr = env->NewIntArray(static_cast<jsize>(v.size()));
    if (arr && !v.empty()) {
        env->SetIntArrayRegion(arr, 0, static_cast<jsize>(v.size()),
                               reinterpret_cast<const jint *>(v.data()));
    }
    return arr;
}

/** float 向量 → jfloatArray */
static inline jfloatArray toJFloatArray(JNIEnv *env, const std::vector<float> &v) {
    jfloatArray arr = env->NewFloatArray(static_cast<jsize>(v.size()));
    if (arr && !v.empty()) {
        env->SetFloatArrayRegion(arr, 0, static_cast<jsize>(v.size()),
                                 reinterpret_cast<const jfloat *>(v.data()));
    }
    return arr;
}

/** uint8 向量 → jintArray（语义分割类别图，uint8 转 int） */
static inline jintArray u8ToJIntArray(JNIEnv *env, const std::vector<uint8_t> &v) {
    jintArray arr = env->NewIntArray(static_cast<jsize>(v.size()));
    if (arr && !v.empty()) {
        std::vector<jint> tmp(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            tmp[i] = v[i];
        }
        env->SetIntArrayRegion(arr, 0, static_cast<jsize>(v.size()), tmp.data());
    }
    return arr;
}

/**
 * 从 C++ ClassifyResult 构造 Java ClassifyResult.
 * 必须在 initDenseResultCache() 成功后调用.
 */
static inline jobject buildClassifyResult(JNIEnv *env, const yolo::ClassifyResult &r) {
    DenseResultCache &c = getDenseResultCache();
    jintArray topIds = toJIntArray(env, r.top_ids);
    jfloatArray topScores = toJFloatArray(env, r.top_scores);
    jfloatArray scores = toJFloatArray(env, r.scores);
    return env->NewObject(c.classifyClazz, c.classifyCtor,
                          r.top_class, r.top_score, r.top_k,
                          topIds, topScores, scores);
}

/**
 * 从 C++ DenseResult 构造 Java SemSegResult.
 */
static inline jobject buildSemSegResult(JNIEnv *env, const yolo::DenseResult &r) {
    DenseResultCache &c = getDenseResultCache();
    jintArray classMap = u8ToJIntArray(env, r.class_map);
    return env->NewObject(c.semsegClazz, c.semsegCtor, r.width, r.height, classMap);
}

/**
 * 从 C++ DenseResult 构造 Java DepthResult.
 */
static inline jobject buildDepthResult(JNIEnv *env, const yolo::DenseResult &r) {
    DenseResultCache &c = getDenseResultCache();
    jfloatArray depthMap = toJFloatArray(env, r.depth_map);
    return env->NewObject(c.depthClazz, c.depthCtor, r.width, r.height, depthMap,
                          r.depth_min, r.depth_max);
}

#endif  // LIBFASTYOLO_JNI_UTILS_H

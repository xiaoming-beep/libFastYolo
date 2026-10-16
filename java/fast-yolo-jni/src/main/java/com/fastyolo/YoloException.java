package com.fastyolo;

import lombok.Getter;

/**
 * YOLO 异常，封装 native 侧返回的错误码.
 * 继承 RuntimeException，调用方可选 catch.
 *
 * <p>错误码常量与 C++ yolo.h 中的宏定义保持一致:
 * <pre>
 *   STATUS_SUCCESS          =  0  正常
 *   STATUS_MODEL_NOT_FOUND  = -1  模型未找到
 *   STATUS_MODEL_INIT_FAILED= -2  模型初始化失败
 *   STATUS_NOT_INITIALIZED  = -3  尚未初始化
 *   STATUS_INFERENCE_FAILED = -4  推理失败
 *   STATUS_TASK_UNSUPPORTED = -5  任务不支持该操作（如对非 DETECT 调用 track()）
 * </pre>
 */
@Getter
public class YoloException extends RuntimeException {

    /** 操作成功 */
    public static final int STATUS_SUCCESS = 0;
    /** 模型文件不存在或无法读取 */
    public static final int STATUS_MODEL_NOT_FOUND = -1;
    /** 后端引擎初始化失败（模型格式/版本不兼容等） */
    public static final int STATUS_MODEL_INIT_FAILED = -2;
    /** 尚未调用 init()，或已调用 close() */
    public static final int STATUS_NOT_INITIALIZED = -3;
    /** 推理过程出错 */
    public static final int STATUS_INFERENCE_FAILED = -4;
    /** 任务不支持该操作（如对非 DETECT 任务调用 track()） */
    public static final int STATUS_TASK_UNSUPPORTED = -5;

    /** native 侧返回的原始状态码 */
    private final int nativeCode;

    public YoloException(int nativeCode, String message) {
        super(message + " [code=" + nativeCode + "]");
        this.nativeCode = nativeCode;
    }

    public static YoloException modelNotFound() {
        return new YoloException(STATUS_MODEL_NOT_FOUND, "模型未找到");
    }

    public static YoloException initFailed() {
        return new YoloException(STATUS_MODEL_INIT_FAILED, "模型初始化失败");
    }

    public static YoloException notInitialized() {
        return new YoloException(STATUS_NOT_INITIALIZED, "引擎尚未初始化");
    }

    public static YoloException inferenceFailed() {
        return new YoloException(STATUS_INFERENCE_FAILED, "推理失败");
    }

    public static YoloException taskUnsupported() {
        return new YoloException(STATUS_TASK_UNSUPPORTED, "任务不支持该操作");
    }

    /**
     * 根据 native 侧返回的 int 状态码构造对应的异常.
     *
     * @param code native 侧状态码
     * @return 对应的 YoloException，状态码为 0 时不抛异常，调用方自行判断
     */
    public static YoloException fromCode(int code) {
        switch (code) {
            case STATUS_MODEL_NOT_FOUND:
                return modelNotFound();
            case STATUS_MODEL_INIT_FAILED:
                return initFailed();
            case STATUS_NOT_INITIALIZED:
                return notInitialized();
            case STATUS_INFERENCE_FAILED:
                return inferenceFailed();
            case STATUS_TASK_UNSUPPORTED:
                return taskUnsupported();
            default:
                return new YoloException(code, "未知 native 错误");
        }
    }
}

/*
 * libFastYolo JNI 接口头文件（聚合）
 *
 * 由 Java 类 com.fastyolo.Letterbox / com.fastyolo.YoloInference
 * 中的 native 方法声明经 `javac -h` 生成（见 jni编译.txt），
 * 本文件仅聚合各 per-class 头文件。
 *
 * 重新生成: cd java/fast-yolo-jni && mvn compile && \
 *   javac -h ../../source/include/fast_yolo -d /tmp/jni_out \
 *     -cp target/classes:$LOMBOK $(find src/main/java -name '*.java')
 */
#ifndef LIBFASTYOLO_JNI_H
#define LIBFASTYOLO_JNI_H

#include "com_fastyolo_Letterbox.h"
#include "com_fastyolo_YoloInference.h"

#endif /* LIBFASTYOLO_JNI_H */

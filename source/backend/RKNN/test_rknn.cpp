// RKNN 后端自测工具：单张图片走完整 predict/track 链路 + 批量计时
// 用法: test_rknn <model.rknn> <image.jpg> [帧数=300] [mode=predict|track]
// 说明: 首帧打印全部检测结果；track 模式连跑 10 帧演示 ByteTrack track_id；
//       计时循环不含首帧（NPU 首次运行有额外开销），平均耗时/FPS 打印到标准输出

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <fast_yolo/yolo.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// 最近邻缩放（fill_bgr 契约要求先缩放到 dst 尺寸；自测工具用最近邻足够）
static void resize_u8(const unsigned char *src, int sw, int sh, unsigned char *dst, int dw, int dh) {
    for (int y = 0; y < dh; ++y) {
        const int sy = std::min(y * sh / dh, sh - 1);
        const unsigned char *srow = src + static_cast<size_t>(sy) * sw * 3;
        unsigned char *drow = dst + static_cast<size_t>(y) * dw * 3;
        for (int x = 0; x < dw; ++x) {
            const int sx = std::min(x * sw / dw, sw - 1);
            drow[x * 3 + 0] = srow[sx * 3 + 0];
            drow[x * 3 + 1] = srow[sx * 3 + 1];
            drow[x * 3 + 2] = srow[sx * 3 + 2];
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("用法: test_rknn <model.rknn> <image.jpg> [帧数=300] [mode=predict|track]\n");
        return 0;
    }
    const char *model_path = argv[1];
    const char *image_path = argv[2];
    const int frames = (argc > 3) ? atoi(argv[3]) : 300;
    const std::string mode = (argc > 4) ? argv[4] : "predict";

    // 1. 读模型（核心层约定：模型以内存字节传入）
    FILE *f = fopen(model_path, "rb");
    if (!f) {
        printf("错误：模型文件打不开 %s\n", model_path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> model_data(static_cast<size_t>(size));
    if (fread(model_data.data(), 1, model_data.size(), f) != model_data.size()) {
        printf("错误：模型文件读取不完整 %s\n", model_path);
        fclose(f);
        return 1;
    }
    fclose(f);

    // 2. 初始化推理（NPU 设备 + 任务类型 AUTO 按输出形状自动识别）
    auto config = std::make_shared<yolo::YOLOConfig>(model_data, 0.25f, 0.45f);
    config->device = yolo::InferDevice::NPU;
    yolo::YOLOInference inference(config);
    int code = inference.init();
    if (code != STATUS_SUCCESS) {
        printf("错误：init 失败 code=%d\n", code);
        return 1;
    }
    printf("任务识别: %d  输出组数: %zu\n", static_cast<int>(inference.resolve_task()),
           config->output_shapes.size());

    // 3. 读图（stb 出 RGB，Letterbox::fill_bgr 吃 BGR，原地交换）
    int w = 0, h = 0, comp = 0;
    unsigned char *img = stbi_load(image_path, &w, &h, &comp, 3);
    if (!img) {
        printf("错误：图片读取失败 %s\n", image_path);
        return 1;
    }
    for (int i = 0; i < w * h; ++i) {
        std::swap(img[i * 3 + 0], img[i * 3 + 2]);
    }

    // 4. letterbox 填充（640 输入，114 灰边；先缩放到 dst 尺寸再填充）
    yolo::Letterbox letterbox(640, 114);
    letterbox.update(w, h);
    std::vector<unsigned char> scaled(static_cast<size_t>(letterbox.dst_width) *
                                      letterbox.dst_height * 3);
    resize_u8(img, w, h, scaled.data(), letterbox.dst_width, letterbox.dst_height);
    letterbox.fill_bgr(scaled.data());

    // 5. 首帧 predict：打印全部结果
    std::vector<yolo::YOLOResult> results = inference.predict(letterbox, &code);
    if (code != STATUS_SUCCESS) {
        printf("错误：predict 失败 code=%d\n", code);
        stbi_image_free(img);
        return 1;
    }
    printf("首帧检测 %zu 个目标:\n", results.size());
    for (const auto &r: results) {
        printf("  类别%d @ (%d %d %d %d) %.3f\n", r.clsId, r.x, r.y, r.width, r.height, r.score);
    }

    // 6. track 模式：连跑 10 帧演示 ByteTrack
    if (mode == "track") {
        printf("---- track 模式（同图 10 帧）----\n");
        for (int i = 0; i < 10; ++i) {
            std::vector<yolo::YOLOResult> tracked = inference.track(letterbox, &code);
            if (code != STATUS_SUCCESS) {
                printf("错误：track 失败 code=%d\n", code);
                break;
            }
            printf("第%d帧: %zu 个目标, track_id =", i + 1, tracked.size());
            for (const auto &r: tracked) {
                printf(" %d", r.track_id);
            }
            printf("\n");
        }
    }

    // 7. 计时循环（含 predict 全链路：letterbox 像素已在上面填好，循环内只做推理+解码）
    const auto t0 = std::chrono::steady_clock::now();
    int ok_frames = 0;
    for (int i = 0; i < frames; ++i) {
        std::vector<yolo::YOLOResult> out = inference.predict(letterbox, &code);
        if (code == STATUS_SUCCESS) {
            ++ok_frames;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double avg_ms = total_ms / (ok_frames > 0 ? ok_frames : 1);
    printf("---- 计时 %d 帧（成功 %d）----\n", frames, ok_frames);
    printf("平均 %.2f ms/帧, %.1f FPS\n", avg_ms, ok_frames > 0 ? ok_frames * 1000.0 / total_ms : 0.0);

    stbi_image_free(img);
    return 0;
}

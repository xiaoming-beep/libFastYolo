// test_onnxcl.cpp —— onnxcl 后端集成验证
//
// 目的：验证 onnxcl 接进 libFastYolo 之后，整条数据通路是对的：
//   模型字节(内存) -> 后端 create/init -> Letterbox 前处理 -> predict -> 原始输出张量
//
// 做法：直接用后端 ABI + libFastYolo 的公共 Letterbox 类，把
//   (1) Letterbox 产生的像素
//   (2) 后端吐出的原始输出张量
// 都 dump 成文件，交给 tools/verify_backend.py 和 onnxruntime 对拍。
// 这样引擎的数值正确性由 Python 侧独立验证，不依赖本程序的任何判断逻辑。
//
// 用法: test_onnxcl <model.onnx> <bgr.raw> <宽> <高> <输出前缀>
#include <fast_yolo/yolo.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::vector<char> readAll(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return {};
    }
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool writeRaw(const std::string &path, const void *data, const size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    f.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(f);
}

// 打印输出张量的概览（范围和均值，用于肉眼判断有无明显异常）
void printStat(const char *tag, const float *p, const size_t n) {
    float lo = p[0], hi = p[0];
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        if (p[i] < lo) lo = p[i];
        if (p[i] > hi) hi = p[i];
        sum += p[i];
    }
    std::printf("  %-12s 元素=%zu  范围=[%.4f, %.4f]  均值=%.4f\n", tag, n, lo, hi, sum / n);
}

}  // namespace

int main(int argc, char **argv) {
    // 关掉 stdout 缓冲：这样即使程序在后续步骤崩了，已经打印的日志也能看到
    // （块缓冲下崩溃会丢掉整个缓冲区，本次排查就吃过这个亏）
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    if (argc < 6) {
        std::printf("用法: %s <model.onnx> <bgr.raw> <宽> <高> <输出前缀>\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string bgr_path = argv[2];
    const int img_w = std::atoi(argv[3]);
    const int img_h = std::atoi(argv[4]);
    const std::string prefix = argv[5];

    // ---- 1. 读模型字节（libFastYolo 的接口就是传字节，不是路径）----
    const std::vector<char> model_bytes = readAll(model_path.c_str());
    if (model_bytes.empty()) {
        std::printf("读不到模型: %s\n", model_path.c_str());
        return 1;
    }
    std::printf("模型: %s (%.2f MB)\n", model_path.c_str(),
                static_cast<double>(model_bytes.size()) / 1048576.0);

    yolo::YOLOConfig config(model_bytes, 0.25f, 0.45f);
    config.device = yolo::InferDevice::OPENCL;

    // ---- 2. 走后端 ABI 建实例 ----
    void *ctx = yolo::yolo_infer_create(&config);
    if (ctx == nullptr) {
        std::printf("后端 create 失败\n");
        return 1;
    }
    int code = 0;
    yolo::yolo_infer_init(ctx, &config, &code);
    if (code != STATUS_SUCCESS) {
        std::printf("后端 init 失败, code=%d\n", code);
        yolo::yolo_infer_destroy(ctx);
        return 1;
    }

    std::printf("输入形状: [");
    for (size_t i = 0; i < config.input_shape.size(); ++i) {
        std::printf("%s%lld", i ? "," : "", static_cast<long long>(config.input_shape[i]));
    }
    std::printf("]\n");
    std::printf("输出个数: %zu\n", config.output_shapes.size());
    for (size_t i = 0; i < config.output_shapes.size(); ++i) {
        std::printf("  输出[%zu] dtype=%d shape=[", i, config.output_dtypes[i]);
        for (size_t d = 0; d < config.output_shapes[i].size(); ++d) {
            std::printf("%s%lld", d ? "," : "", static_cast<long long>(config.output_shapes[i][d]));
        }
        std::printf("]\n");
    }

    // ---- 3. 前处理：用 libFastYolo 自己的 Letterbox ----
    const std::vector<char> inData = readAll(bgr_path.c_str());
    if (inData.empty()) {
        std::printf("读不到输入文件: %s\n", bgr_path.c_str());
        yolo::yolo_infer_destroy(ctx);
        return 1;
    }

    size_t inputElems = 1;
    for (int64_t d : config.input_shape) inputElems *= static_cast<size_t>(d);
    // 两种输入模式：
    //   · 文件字节数 == 模型输入元素数 x 4 -> 直接当 float32 张量用
    //     （合成用例测试：模型输入形状是任意的，不走 Letterbox）
    //   · 否则当 BGR 图像走 Letterbox（集成测试：前处理由 libFastYolo 负责）
    const bool rawInput = (inData.size() == inputElems * sizeof(float));

    const int box = config.input_shape.size() >= 3 ? static_cast<int>(config.input_shape[2]) : 8;
    yolo::Letterbox letterbox(box);

    if (rawInput) {
        // pixels 是 public 成员，直接灌进去；update(box,box) 给出恒等映射
        letterbox.update(box, box);
        letterbox.pixels.assign(reinterpret_cast<const float *>(inData.data()),
                                reinterpret_cast<const float *>(inData.data()) + inputElems);
        std::printf("输入模式: 原始 float32 张量 (%zu 元素)\n", inputElems);
    } else {
        if (config.input_shape.size() != 4) {
            std::printf("BGR 模式要求 4 维输入\n");
            yolo::yolo_infer_destroy(ctx);
            return 1;
        }
        if (inData.size() != static_cast<size_t>(img_w) * img_h * 3) {
            std::printf("BGR 文件大小不符: 实际 %zu, 期望 %d\n", inData.size(), img_w * img_h * 3);
            yolo::yolo_infer_destroy(ctx);
            return 1;
        }
        letterbox.update(img_w, img_h);
        letterbox.fill_bgr(reinterpret_cast<const uint8_t *>(inData.data()));
        std::printf("输入模式: BGR -> Letterbox box=%d 原图=%dx%d scale=%.6f 偏移=(%.2f, %.2f) 像素数=%zu\n",
                    box, img_w, img_h, letterbox.get_scale(), letterbox.get_scaled_x(),
                    letterbox.get_scaled_y(), letterbox.pixels.size());
    }

    // dump 前处理结果，交给 Python 侧独立喂给 onnxruntime 对拍
    if (!writeRaw(prefix + "_pixels.bin", letterbox.pixels.data(),
                  letterbox.pixels.size() * sizeof(float))) {
        std::printf("写像素文件失败\n");
        yolo::yolo_infer_destroy(ctx);
        return 1;
    }

    // ---- 4. 推理 ----
    //
    // 【ONNXCL_REPEAT】重复推理 N 次，用途有二：
    //   1) 抓「第二次推理才崩」这类生命周期 bug —— 缓冲池/句柄复用的问题只有跑
    //      第二遍才会暴露（第一遍池子里还是空的，不会命中脏句柄）。
    //   2) 微基准：核显和 CPU 共享显存，单次测量噪声可达 ±15%，必须跑多次取最小值。
    int repeat = 1;
    if (const char *rp = std::getenv("ONNXCL_REPEAT")) {
        const int v = std::atoi(rp);
        if (v > 0) repeat = v;
    }

    yolo::YoloInferOutput outs[16];
    int n = 0;
    for (int rep = 0; rep < repeat; ++rep) {
        n = yolo::yolo_infer_predict(ctx, &letterbox, outs, 16, &code);
        if (code != STATUS_SUCCESS || n == 0) {
            std::printf("推理失败(第 %d/%d 次), code=%d, n=%d\n", rep + 1, repeat, code, n);
            yolo::yolo_infer_destroy(ctx);
            return 1;
        }
    }
    std::printf("推理成功(%d 次), 输出 %d 个\n", repeat, n);

    for (int i = 0; i < n; ++i) {
        const size_t cnt = static_cast<size_t>(outs[i].size);
        // 【必须按 elem_type 算元素宽度】以前这里写死了 sizeof(float)，
        // 遇到 uint8 输出（如语义分割的类别图）会把 1MB 的缓冲当 4MB 读，直接越界崩掉。
        size_t elemBytes = 4;
        switch (outs[i].elem_type) {
            case 2:  // uint8
            case 3:  // int8
                elemBytes = 1;
                break;
            case 7:  // int64
                elemBytes = 8;
                break;
            default:  // float32 / int32 / float16 等
                elemBytes = 4;
                break;
        }
        if (outs[i].elem_type == 1) {
            printStat("raw output(float)", static_cast<const float *>(outs[i].data), cnt);
        } else {
            std::printf("  raw output(elem_type=%d) 元素=%zu  字节=%zu\n", outs[i].elem_type, cnt,
                        cnt * elemBytes);
        }
        char path[512];
        std::snprintf(path, sizeof(path), "%s_out%d.bin", prefix.c_str(), i);
        if (!writeRaw(path, outs[i].data, cnt * elemBytes)) {
            std::printf("写输出文件失败: %s\n", path);
            yolo::yolo_infer_destroy(ctx);
            return 1;
        }
        std::printf("  已写出 %s\n", path);
    }

    yolo::yolo_infer_destroy(ctx);
    std::printf("完成\n");
    return 0;
}

#ifndef LIBFASTYOLO_IMG_UTIL_H
#define LIBFASTYOLO_IMG_UTIL_H
#include <vector>
#include <fstream>
#include <cstdint>

namespace yolo {
    //宏定义
    constexpr float INV_255 = 1.0f / 255.0f;

    // 读取整个模型文件到内存
    std::vector<char> read_bin_file(const std::string &path);

    /**
     * 优化上下填充_bgr
     */
    void vertical_fill_bgr(int box_size, int scaled_width, int scaled_height, const uint8_t *img_data,
                           float *pixels);

    /**
    * 优化左右填充bgr
    */
    void horizontal_fill_bgr(int box_size, int scaled_width, int scaled_height, const uint8_t *img_data,
                             float *pixels);

    /**
     * 中心裁剪填充（分类任务）：按最短边缩放后的 scaled_width×scaled_height 图像
     * （其中一边等于 box_size，另一边 ≥ box_size），取中心 box_size×box_size 区域
     * 写入 planar RRR..GGG..BBB float（BGR 输入，/255 归一化）。
     */
    void center_crop_fill_bgr(int box_size, int scaled_width, int scaled_height,
                              const uint8_t *img_data, float *pixels);
}
#endif

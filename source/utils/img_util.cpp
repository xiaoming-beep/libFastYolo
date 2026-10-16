#include <fast_yolo/img_util.h>

//读取文件
std::vector<char> yolo::read_bin_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open model file: " + path);
    }
    file.seekg(0, std::ios::end);
    const size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

/**
* 上下填充_bgr
*/
void yolo::vertical_fill_bgr(const int box_size, const int scaled_width, const int scaled_height,
                             const uint8_t *img_data,
                             float *pixels) {
    const int total_pixels = box_size * box_size;
    const int pad_top = (box_size - scaled_height) / 2;
    int dst_r = pad_top * box_size; // r 平面起始像素索引
    int dst_g = total_pixels + dst_r; // g 平面
    int dst_b = total_pixels * 2 + dst_r; // b 平面

    int src_idx = 0;
    const int row_bytes = scaled_width * 3;
    //填充
    for (int y = 0; y < scaled_height; y++) {
        const int row_end = src_idx + row_bytes;
        while (src_idx < row_end) {
            // 一次读取三个字节并处理
            const int b = img_data[src_idx++] & 0xff;
            const int g = img_data[src_idx++] & 0xff;
            const int r = img_data[src_idx++] & 0xff;

            pixels[dst_r++] = r * INV_255;
            pixels[dst_g++] = g * INV_255;
            pixels[dst_b++] = b * INV_255;
        }
        // 下一行在目标数组中的起始位置
        dst_r += (box_size - scaled_width);
        dst_g += (box_size - scaled_width);
        dst_b += (box_size - scaled_width);
    }
}

//左右填充bgr
void yolo::horizontal_fill_bgr(int box_size, int scaled_width, int scaled_height, const uint8_t *img_data,
                               float *pixels) {
    const int total_pixels = box_size * box_size;
    const int pad_left = (box_size - scaled_width) / 2;

    int src_idx = 0;

    for (int y = 0; y < scaled_height; y++) {
        // 计算当前行在三个平面中的起始写入位置
        const int base_pixel = y * box_size + pad_left;

        int r_offset = base_pixel;
        int g_offset = total_pixels + base_pixel;
        int b_offset = total_pixels * 2 + base_pixel;

        const int row_end = src_idx + scaled_width * 3;
        while (src_idx < row_end) {
            const int b = img_data[src_idx++] & 0xff;
            const int g = img_data[src_idx++] & 0xff;
            const int r = img_data[src_idx++] & 0xff;

            pixels[r_offset++] = r * INV_255;
            pixels[g_offset++] = g * INV_255;
            pixels[b_offset++] = b * INV_255;
        }
    }
}

//中心裁剪填充（分类任务）：取中心 box_size×box_size 区域，不填充
void yolo::center_crop_fill_bgr(const int box_size, const int scaled_width, const int scaled_height,
                                const uint8_t *img_data,
                                float *pixels) {
    // 裁剪偏移（scaled 尺寸中有一边 == box_size，另一边 ≥ box_size）
    const int crop_x = (scaled_width - box_size) / 2;
    const int crop_y = (scaled_height - box_size) / 2;
    const int total_pixels = box_size * box_size;

    for (int y = 0; y < box_size; y++) {
        const int src_row_start = (crop_y + y) * scaled_width + crop_x;
        const int base_pixel = y * box_size;
        int src_idx = src_row_start * 3;
        int r_offset = base_pixel;
        int g_offset = total_pixels + base_pixel;
        int b_offset = total_pixels * 2 + base_pixel;
        for (int x = 0; x < box_size; x++) {
            const int b = img_data[src_idx++] & 0xff;
            const int g = img_data[src_idx++] & 0xff;
            const int r = img_data[src_idx++] & 0xff;
            pixels[r_offset++] = r * INV_255;
            pixels[g_offset++] = g * INV_255;
            pixels[b_offset++] = b * INV_255;
        }
    }
}

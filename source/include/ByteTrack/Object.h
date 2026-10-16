#pragma once

#include "ByteTrack/Rect.h"

namespace byte_track
{
struct Object
{
    Rect<float> rect;
    int label;
    float prob;
    // 输入检测在原始结果 vector 中的下标（-1 表示无），用于跟踪后回填掩码/关键点等载荷
    int index = -1;

    Object(const Rect<float> &_rect,
           const int &_label,
           const float &_prob,
           const int &_index = -1);
};
}
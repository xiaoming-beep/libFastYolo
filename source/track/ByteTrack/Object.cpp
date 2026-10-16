#include "ByteTrack/Object.h"

byte_track::Object::Object(const Rect<float> &_rect,
                           const int &_label,
                           const float &_prob,
                           const int &_index) : rect(_rect), label(_label), prob(_prob), index(_index)
{
}
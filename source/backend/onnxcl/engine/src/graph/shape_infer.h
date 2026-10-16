#pragma once

#include <string>

#include "onnxcl/graph.h"

namespace onnxcl {

// 为单个节点推导输出张量的 shape / dtype。
// 要求输入张量的 shape 已经推导完成（graph_build 按拓扑序调用）。
bool inferNodeShape(const Node& node, Graph& g, std::string* err);

}  // namespace onnxcl

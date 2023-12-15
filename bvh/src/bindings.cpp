#include <pybind11/pybind11.h>

#include <torch/extension.h>
#include "bvh.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("create_bvh", &create_bvh);
    m.def("trace_bvh", &trace_bvh);
    m.def("trace_bvh_opacity", &trace_bvh_opacity);
}
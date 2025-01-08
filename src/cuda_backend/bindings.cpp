#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern "C" void add(int* c, const int* a, const int* b, int size);

// Wrapper for the CUDA function to work with NumPy arrays
pybind11::array_t<int> add_py(pybind11::array_t<int> a, pybind11::array_t<int> b) {
    auto buf_a = a.request(), buf_b = b.request();

    if (buf_a.size != buf_b.size) {
        throw std::runtime_error("Input arrays must have the same size");
    }

    int size = buf_a.size;
    auto result = pybind11::array_t<int>(size);
    auto buf_c = result.request();

    int* host_a = static_cast<int*>(buf_a.ptr);
    int* host_b = static_cast<int*>(buf_b.ptr);
    int* host_c = static_cast<int*>(buf_c.ptr);

    add(host_c, host_a, host_b, size);

    return result;
}

PYBIND11_MODULE(cuda_backend, m) {
    m.doc() = "CUDA addition module using Pybind11";
    m.def("add", &add_py, "Add two arrays using CUDA");
}

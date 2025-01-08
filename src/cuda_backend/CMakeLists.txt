cmake_minimum_required(VERSION 3.18)
project(cuda_backend LANGUAGES CUDA CXX)

# Find Python first
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Add this line to properly find pybind11
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# Use CUDAToolkit instead of CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures for RTX 3080 Ti (86) and RTX 40 series (89)
set(CMAKE_CUDA_ARCHITECTURES 86 89)

# Create CUDA library instead of executable
add_library(cuda_backend SHARED
    test.cu
    bindings.cpp
)

# Enable C++17 and CUDA support
set_target_properties(cuda_backend PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Link with CUDA runtime and pybind11
target_link_libraries(cuda_backend PRIVATE
    CUDA::cudart
    pybind11::module
)

# If you want Python to be able to import this module, set the output name
set_target_properties(cuda_backend PROPERTIES PREFIX "")
if(WIN32)
    set_target_properties(cuda_backend PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(cuda_backend PROPERTIES SUFFIX ".so")
endif()

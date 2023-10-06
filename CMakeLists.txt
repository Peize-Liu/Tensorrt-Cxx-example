cmake_minimum_required(VERSION 3.15)
project(tensorrt_test)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
add_compile_options(-Wno-deprecated-declarations -Wno-reorder  -Wno-format -Wno-sign-compare)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)



include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${CUDA_LIBRARIES} ${OpenCV_LIBS} nvinfer nvinfer_plugin nvonnxparser)
add_executable(tensorrt_test tensorrt_test.cpp)



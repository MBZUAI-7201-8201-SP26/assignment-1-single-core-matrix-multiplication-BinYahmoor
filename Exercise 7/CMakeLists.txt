cmake_minimum_required(VERSION 3.22...4.2)
project(ttnn_example_lab7_matmul)

add_executable(ttnn_example_lab7_matmul)
target_sources(ttnn_example_lab7_matmul PRIVATE lab7_matmul.cpp)

find_package(TT-Metalium REQUIRED)
find_package(TT-NN REQUIRED)

target_link_libraries(
    ttnn_example_lab7_matmul
    PUBLIC
        TT::Metalium
        TTNN::TTNN
)

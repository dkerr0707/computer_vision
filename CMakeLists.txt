cmake_minimum_required(VERSION 3.18)
project(train_mnist)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

add_executable(train_mnist train_mnist.cpp)
target_link_libraries(train_mnist ${TORCH_LIBRARIES})
target_link_directories(train_mnist PRIVATE /home/dk/dev/bling/.venv/lib/python3.12/site-packages/nvidia/cu13/lib)

add_executable(infer infer.cpp)
target_link_libraries(infer ${TORCH_LIBRARIES})
target_link_directories(infer PRIVATE /home/dk/dev/bling/.venv/lib/python3.12/site-packages/nvidia/cu13/lib)

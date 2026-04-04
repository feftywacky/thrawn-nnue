cmake_minimum_required(VERSION 3.20)
project(thrawn_nnue_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(thrawn_nnue_native SHARED
  src/binpack_bridge.cpp
)

target_include_directories(thrawn_nnue_native PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nnue_pytorch
)

if (APPLE)
  target_compile_options(thrawn_nnue_native PRIVATE -O3)
else()
  target_compile_options(thrawn_nnue_native PRIVATE -O3 -fPIC)
endif()

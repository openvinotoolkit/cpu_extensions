# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.13)

project(cpu_extensions_ext LANGUAGES CXX)

# ref:https://stackoverflow.com/questions/68401650/how-can-i-make-a-pytorch-extension-with-cmake
execute_process(COMMAND python3 -c "import torch;print(torch.utils.cmake_prefix_path+'/Torch/',end='')" OUTPUT_VARIABLE TORCH_CMAKE_PATH)
set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

add_library(cpu_extensions_ext SHARED
  mha_gpt.cpp
  module.cpp
  ../../src/test_common.cpp
)
set_target_properties(cpu_extensions_ext PROPERTIES
    OUTPUT_NAME "cpu_extensions_ext"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../)
install(TARGETS cpu_extensions_ext DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
target_compile_features(cpu_extensions_ext PRIVATE cxx_std_14)
target_include_directories(cpu_extensions_ext PRIVATE ../../src)
target_link_libraries(cpu_extensions_ext PRIVATE ${TORCH_LIBRARIES} Python::Python cpu_extensions stdc++ OpenMP::OpenMP_CXX)

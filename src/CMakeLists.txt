# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)
project(cpu_extensions)

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(cpu_extensions STATIC ${SOURCE_FILES})
target_include_directories(cpu_extensions PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                          PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
set_target_properties(cpu_extensions PROPERTIES
                      POSITION_INDEPENDENT_CODE ON)

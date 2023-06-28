# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)
project(cpu_extensions_tests)

enable_testing()

if (NOT TARGET gtest_main)
  include(FetchContent)
  FetchContent_Declare(
          googletest
          GIT_REPOSITORY https://github.com/google/googletest.git
          GIT_TAG release-1.11.0
          GIT_SHALLOW TRUE
          GIT_PROGRESS TRUE)
  FetchContent_MakeAvailable(googletest)
endif()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

include_directories(${CPU_EXTENSIONS_HEADERS_DIR})

file(GLOB_RECURSE TEST_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

find_package(OpenMP REQUIRED)

add_executable(cpu_extensions_tests ${TEST_SOURCE_FILES})
target_link_libraries(cpu_extensions_tests cpu_extensions gtest_main stdc++ OpenMP::OpenMP_CXX)

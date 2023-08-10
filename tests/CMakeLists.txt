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
  FetchContent_GetProperties(googletest)
  if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

file(GLOB_RECURSE TEST_SOURCE_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/test_*.cpp)

find_package(OpenMP REQUIRED)

add_executable(cpu_extensions_tests ${TEST_SOURCE_FILES})
target_include_directories(cpu_extensions_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(cpu_extensions_tests cpu_extensions gtest_main stdc++ OpenMP::OpenMP_CXX)
target_compile_options(cpu_extensions_tests PRIVATE ${EXTRA_CXX_FLAGS})


add_executable(benchfc ${CMAKE_CURRENT_SOURCE_DIR}/src/benchfc.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/test_common.cpp)
target_include_directories(benchfc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(benchfc cpu_extensions stdc++ OpenMP::OpenMP_CXX)

# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

project(root)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(CPU_EXTENSIONS_BUILD_TESTS "Build with tests" ON)
option(CPU_EXTENSIONS_ENABLE_LOG "Enable log" OFF)

message(INFO "--------------------------------")
message(STATUS "Build with tests: ${CPU_EXTENSIONS_BUILD_TESTS}")
message(INFO "--------------------------------")

if(MSVC)
  # TODO: validate
  if(MSVC_VERSION VERSION_LESS 1928)
    message(FATAL_ERROR "Insufficient msvc compiler version, current ${MSVC_VERSION}, minimum 1928.")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.2")
    message(FATAL_ERROR "Insufficient gcc compiler version, current ${CMAKE_CXX_COMPILER_VERSION}, minimum 11.2.")
  endif()
  set(EXTRA_CXX_FLAGS -march=sapphirerapids -flax-vector-conversions)
elseif(OV_COMPILER_IS_CLANG)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "12")
    message(FATAL_ERROR "Insufficient clang compiler version, current ${CMAKE_CXX_COMPILER_VERSION}, minimum 12.")
  endif()
  set(EXTRA_CXX_FLAGS -march=sapphirerapids -flax-vector-conversions)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "2023.0")
    message(FATAL_ERROR "Insufficient intel compiler version, current ${CMAKE_CXX_COMPILER_VERSION}, minimum 2023.0.")
  endif()
  set(EXTRA_CXX_FLAGS -march=sapphirerapids)
endif()

if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
add_subdirectory(src)
if (CPU_EXTENSIONS_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Get the latest commit hash
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_VARIABLE GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
file(WRITE ${CMAKE_BINARY_DIR}/git-state.txt ${GIT_HASH})
install(FILES
        ${CMAKE_BINARY_DIR}/git-state.txt
        DESTINATION ${CMAKE_INSTALL_PREFIX})

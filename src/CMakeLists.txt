# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)
project(cpu_extensions)

file(GLOB_RECURSE ${PROJECT_NAME}_SOURCE_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES
                      POSITION_INDEPENDENT_CODE ON)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
                                                  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_options(${PROJECT_NAME} PRIVATE ${EXTRA_CXX_FLAGS})

set(CMAKE_DST lib/cmake/${PROJECT_NAME})
# header files
include(GNUInstallDirs)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# library file
install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Targets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

# config file
install(EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Config.cmake
        DESTINATION ${CMAKE_DST})

# version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        VERSION 1.0.0
        COMPATIBILITY AnyNewerVersion)
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${CMAKE_DST})

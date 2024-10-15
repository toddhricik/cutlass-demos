cmake_minimum_required(VERSION 3.29)
#cmake_policy(SET CMP0112 NEW)
project(cutlass-ampere-demos LANGUAGES CXX CUDA)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)
include(GNUInstallDirs)

include(CMakePackageConfigHelpers)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake")
        file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake")
endif()
# Append path to "cmake" directory in root of prjoject directory
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

add_subdirectory(${CMAKE_SOURCE_DIR}/src outgoing)

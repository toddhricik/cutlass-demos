cmake_minimum_required(VERSION 3.29)
# There may be policy issues
#cmake_policy(SET CMP0112 NEW)
project(cutlass-ampere-demos LANGUAGES CXX CUDA)
# We are going to use a CUDA enabled Nvidia GPU
enable_language(CUDA)
# Print some helpful info
message("CUDACXX = ${CUDA_INSTALL_PATH}/bin/nvcc")

# CUTLASS requires c++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)
# This always comes in handy on UNIX for installation and more
include(GNUInstallDirs)
# We will use CMakePackageConfigHelpers when we install our dev/test/deploy product
include(CMakePackageConfigHelpers)
# Create the directory "cmake" at the root of the source tree
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake")
        file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake")
endif()
# Append path to "cmake" directory in root of prjoject directory
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
# Add the include directory. Create the include directory if needed
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake")
	file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/include")
endif()
add_subdirectory(${CMAKE_SOURCE_DIR}/include include)
# Add the src subdirectory which contains source files
add_subdirectory(${CMAKE_SOURCE_DIR}/src src)


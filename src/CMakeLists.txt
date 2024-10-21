
cmake_minimum_required(VERSION 3.29)

set(CMAKE_CXX_STANDARD 17 REQUIRED FATAL)

#set(CMAKE_CXX_STANDARD_REQUIRED YES)
include(GNUInstallDirs)

add_executable(my_cute_demo my_cute_demo.cu)
#set_target_properties(my_cute_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(my_cute_demo INTERFACE ${CMAKE_SOURCE_DIR}/include)

target_link_directories(my_cute_demo INTERFACE cutlassTH)


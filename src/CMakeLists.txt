
cmake_minimum_required(VERSION 3.29)

set(CMAKE_CXX_STANDARD 17 REQUIRED FATAL)

include(GNUInstallDirs)

add_executable(layoutExercises layoutExercises.cu)

target_include_directories(layoutExercises PRIVATE ${CMAKE_SOURCE_DIR}/include)



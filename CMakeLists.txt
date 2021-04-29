cmake_minimum_required(VERSION 3.16)
project(E351EqCalor)
set(CMAKE_CXX_STANDARD 14)
add_executable(E351EqCalor E351EqCalor.cpp)

# Require MPI for this project:
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpic++)
set(SOURCE_FILES E351EqCalor.cpp)
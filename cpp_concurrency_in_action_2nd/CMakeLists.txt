cmake_minimum_required(VERSION 3.12)

project(cpp_concurrency_in_action)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSION OFF)

set(ROOT_PATH ${CMAKE_SOURCE_DIR})

# set target output directory
file(MAKE_DIRECTORY out)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out/)

# if compiler is Clang, link libc++, otherwise, link default libstdc++
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
endif()

add_subdirectory(ch02_managing_threads)
add_subdirectory(ch03_sharing_data_between_threads)
#add_subdirectory(ch04_synchronizing_concurrent_operations)
#add_subdirectory(ch05_memory_order_and_atomic_types)

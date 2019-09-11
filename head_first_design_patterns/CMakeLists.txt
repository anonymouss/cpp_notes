cmake_minimum_required(VERSION 3.12)

project(leraning_head_first_design_patterns)

# use standard C++ 17, disallow extensions, need compiler support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSION OFF)

# if compiler is Clang, link libc++, otherwise, link default libstdc++
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
endif()

set(ROOT_PATH ${CMAKE_SOURCE_DIR})

# set target output directory
file(MAKE_DIRECTORY out)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out/)

add_subdirectory(01_Strategy)
add_subdirectory(02_Observer)
add_subdirectory(03_Decorator)
add_subdirectory(04_Factory)
add_subdirectory(05_Singleton)
add_subdirectory(06_Command)
add_subdirectory(07_Adapter_and_Facade)
add_subdirectory(08_Template_Method)
add_subdirectory(09_Iterator_and_Composite)
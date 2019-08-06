cmake_minimum_required(VERSION 3.12)

project(learning_effective_modern_cpp)

# use standard C++ 14, disallow extensions, need compiler support
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSION OFF)

set(ROOT_PATH ${CMAKE_SOURCE_DIR})

# set target output directory
file(MAKE_DIRECTORY out)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out/)

set(Boost_NO_BOOST_CMAKE ON)

# boost library
# find_package(Boost)
# if(Boost_FOUND)
#     message("Found boost library")
#     ADD_DEFINITIONS(-DHAS_BOOST)
#     include_directories(${Boost_INCLUDE_DIRS})
# else()
#     message(WARNING "boost not found, use workaround")
# endif()

# FIXME: somthing wrong with my boost installation that cmake can't detect boost.
# hardcode is just a workaround
ADD_DEFINITIONS(-DHAS_BOOST)
include_directories(/usr/local/include/boost)

add_subdirectory(source)

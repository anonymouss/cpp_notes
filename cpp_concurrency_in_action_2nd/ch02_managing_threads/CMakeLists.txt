set(TARGET_NAME "02_managing_threads")

add_executable(${TARGET_NAME} main.cpp)
target_link_libraries(${TARGET_NAME} pthread)
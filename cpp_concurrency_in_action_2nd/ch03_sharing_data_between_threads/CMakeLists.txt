set(TARGET_NAME "03_sharing_data_between_threads")

add_executable(${TARGET_NAME} main.cpp)
target_link_libraries(${TARGET_NAME} pthread)
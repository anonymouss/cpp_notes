set(TARGET_NAME "04_synchronizing_concurrent_operations")

add_executable(${TARGET_NAME} main.cpp)
target_link_libraries(${TARGET_NAME} pthread)

add_executable(${TARGET_NAME}_atm ATM.cpp)
target_link_libraries(${TARGET_NAME}_atm pthread)
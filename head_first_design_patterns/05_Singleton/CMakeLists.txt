include_directories (inc)
aux_source_directory (src SRC_DIR)

get_filename_component(TARGET_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_executable (${TARGET_NAME} main.cpp ${SRC_DIR})

if (UNIX)
    target_link_libraries(${TARGET_NAME} pthread)
endif ()

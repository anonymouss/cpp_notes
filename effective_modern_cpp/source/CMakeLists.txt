# foreach source files and compile
file (GLOB SOURCE_FILES ./*.cpp)
foreach (FILE ${SOURCE_FILES})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" FNAME ${FILE})
    add_executable (${FNAME} ${FILE})
    if (UNIX)
        target_link_libraries(${FNAME} pthread)
    endif ()
endforeach ()
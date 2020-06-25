add_executable( beams ${CMAKE_CURRENT_SOURCE_DIR}/beams.cpp )
target_link_libraries( beams admmelastic ${RENDER_LIBS} )

add_executable( beams_rubber ${CMAKE_CURRENT_SOURCE_DIR}/beams_rubber.cpp )
target_link_libraries( beams_rubber admmelastic ${RENDER_LIBS} )

cmake_minimum_required (VERSION 3.6)

set(SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/GLTF_PBR_Renderer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/QxGLTF_PBR_Render.cpp"
)

set(INCLUDE
    "${CMAKE_CURRENT_SOURCE_DIR}/interface/GLTF_PBR_Renderer.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/interface/QxGLTF_PBR_Render.hpp"
)

target_sources(DiligentFX PRIVATE ${SOURCE} ${INCLUDE})

target_include_directories(DiligentFX
PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/interface"
)

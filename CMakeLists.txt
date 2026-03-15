cmake_minimum_required(VERSION 3.10)

project(lajolla)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

if(MSVC)
 # add_compile_options(/Wall)
else()
  add_compile_options(-Wall)
endif()

find_package(Embree REQUIRED)
find_package(Vulkan REQUIRED)
find_program(SLANGC_EXECUTABLE slangc REQUIRED)

set(SHADER_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/shaders)
set(SHADER_OUTPUT ${CMAKE_BINARY_DIR}/basic.spv)
file(GLOB SHADER_SOURCE "${SHADER_SOURCE_DIR}/*.slang")

macro(addShaderCompile arg1 arg2)
	add_custom_command(
		OUTPUT ${arg1}.spv
		COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCE_DIR}/${arg2} -O3 -target spirv -entry ${arg1} -o ${CMAKE_BINARY_DIR}/${arg1}.spv
		DEPENDS ${SHADER_SOURCE}
		COMMENT ${arg1}.spv
	)
endmacro()

addShaderCompile(rgen_path_tracing PathTracing.slang)
addShaderCompile(miss Intersection.slang)
addShaderCompile(miss_shadow Intersection.slang)
addShaderCompile(chit_tri Intersection.slang)
addShaderCompile(ahit Intersection.slang)
addShaderCompile(tonemap Tonemap.slang)

add_custom_target(CompileShaders ALL DEPENDS
			${CMAKE_BINARY_DIR}/rgen_path_tracing.spv
			${CMAKE_BINARY_DIR}/miss.spv
			${CMAKE_BINARY_DIR}/miss_shadow.spv
			${CMAKE_BINARY_DIR}/chit_tri.spv
			${CMAKE_BINARY_DIR}/ahit.spv
			${CMAKE_BINARY_DIR}/tonemap.spv)

include_directories(${EMBREE_INCLUDE_PATH})
link_libraries(${EMBREE_LIBRARY} Vulkan::Vulkan)

include_directories(${CMAKE_SOURCE_DIR}/src)

set(SRCS src/3rdparty/miniz.h
         src/3rdparty/miniz.c
         src/3rdparty/pugiconfig.hpp
         src/3rdparty/pugixml.hpp
         src/3rdparty/pugixml.cpp
         src/3rdparty/stb_image.h
         src/3rdparty/tinyexr.h
         src/3rdparty/tinyply.h
         src/3rdparty/RGFW.h
         src/filters/box.inl
         src/filters/gaussian.inl
         src/filters/tent.inl
         src/lights/diffuse_area_light.inl
         src/lights/envmap.inl
         src/materials/lambertian.inl
         src/materials/roughdielectric.inl
         src/materials/roughplastic.inl
         src/materials/disney_diffuse.inl
         src/materials/disney_metal.inl
         src/materials/disney_glass.inl
         src/materials/disney_clearcoat.inl
         src/materials/disney_sheen.inl
         src/materials/disney_bsdf.inl
         src/media/homogeneous.inl
         src/media/heterogeneous.inl
         src/parsers/load_serialized.h
         src/parsers/parse_obj.h
         src/parsers/parse_ply.h
         src/parsers/parse_scene.h
         src/phase_functions/isotropic.inl
         src/phase_functions/henyeygreenstein.inl
         src/shapes/sphere.inl
         src/shapes/triangle_mesh.inl
         src/camera.h
         src/filter.h
         src/flexception.h
         src/frame.h
         src/image.h
         src/intersection.h
         src/lajolla.h
         src/gpu_device.h
         src/gpu_device.cpp
         src/light.h
         src/material.h
         src/matrix.h
         src/medium.h
         src/microfacet.h
         src/mipmap.h
         src/parallel.h
         src/path_tracing.h
         src/phase_function.h
         src/vol_path_tracing.h
         src/volume.h
         src/point_and_normal.h
         src/ray.h
         src/render.h
         src/scene.h
         src/shape.h
         src/spectrum.h
         src/table_dist.h
         src/texture.h
         src/transform.h
         src/vector.h
         src/parsers/load_serialized.cpp
         src/parsers/parse_obj.cpp
         src/parsers/parse_ply.cpp
         src/parsers/parse_scene.cpp
         src/camera.cpp
         src/filter.cpp
         src/image.cpp
         src/intersection.cpp
         src/light.cpp
         src/material.cpp
         src/medium.cpp
         src/parallel.cpp
         src/phase_function.cpp
         src/render.cpp
         src/scene.cpp
         src/shape.cpp
         src/table_dist.cpp
         src/transform.cpp
         src/volume.cpp)

add_library(lajolla_lib STATIC ${SRCS})
target_compile_definitions(lajolla_lib PRIVATE "__DEBUG__=$<IF:$<CONFIG:Debug>,1,0>")
add_library(RGFW SHARED "src/RGFW.cpp")
if(APPLE)
    target_link_libraries(RGFW PRIVATE "$<LINK_LIBRARY:FRAMEWORK,Cocoa>")
    target_link_libraries(RGFW PRIVATE "$<LINK_LIBRARY:FRAMEWORK,IOKit>")
endif()
add_executable(lajolla src/main.cpp)
target_link_libraries(lajolla lajolla_lib RGFW)
if(MSVC)
  add_custom_command(TARGET lajolla POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${PROJECT_SOURCE_DIR}/embree/bin"
      $<TARGET_FILE_DIR:lajolla>)
endif()

# Copy compiled SPIR-V next to executable for runtime loading
add_custom_command(TARGET lajolla POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy ${SHADER_OUTPUT} $<TARGET_FILE_DIR:lajolla>
)
add_dependencies(lajolla CompileShaders)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(lajolla Threads::Threads)

enable_testing()

add_executable(test_filter src/tests/filter.cpp)
target_link_libraries(test_filter lajolla_lib)
add_test(filter test_filter)
set_tests_properties(filter PROPERTIES PASS_REGULAR_EXPRESSION "SUCCESS")

add_executable(test_frame src/tests/frame.cpp)
target_link_libraries(test_frame lajolla_lib)
add_test(frame test_frame)
set_tests_properties(frame PROPERTIES PASS_REGULAR_EXPRESSION "SUCCESS")

add_executable(test_image src/tests/image.cpp)
target_link_libraries(test_image lajolla_lib)
add_test(image test_image)
set_tests_properties(image PROPERTIES PASS_REGULAR_EXPRESSION "SUCCESS")

add_executable(test_intersection src/tests/intersection.cpp)
target_link_libraries(test_intersection lajolla_lib)
add_test(intersection test_intersection)
set_tests_properties(intersection PROPERTIES PASS_REGULAR_EXPRESSION "SUCCESS")

add_executable(test_materials src/tests/materials.cpp)
target_link_libraries(test_materials lajolla_lib)
add_test(materials test_materials)
set_tests_properties(materials PROPERTIES PASS_REGULAR_EXPRESSION "SUCCESS")

add_executable(test_matrix src/tests/matrix.cpp)
target_link_libraries(test_matrix lajolla_lib)
add_test(matrix test_matrix)
set_tests_properties(matrix PROPERTIES PASS_REGULAR_EXPRESSION "SUCCESS")

add_executable(test_mipmap src/tests/mipmap.cpp)
target_link_libraries(test_mipmap lajolla_lib)
add_test(mipmap test_mipmap)
set_tests_properties(mipmap PROPERTIES PASS_REGULAR_EXPRESSION "SUCCESS")

cmake_minimum_required(VERSION 3.13)

project(pynanomito VERSION 0.1 LANGUAGES CUDA CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PythonExtensions REQUIRED)
find_package(CUDA 7.0 REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}"
  -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(${CMAKE_PROJECT_NAME} MODULE
	PyNanoMito.cpp
	../NanoMito3D/kernels/GaussianFilterGPU.cu
	../NanoMito3D/kernels/LocalThresholdGPU.cu
	../NanoMito3D/kernels/RenderingGPU.cu
	../NanoMito3D/kernels/DensityFilterGPU.cu
	../NanoMito3D/kernels/Device.cpp
	../NanoMito3D/Volume.h
	../NanoMito3D/Volume.cpp
	../NanoMito3D/AnalyzeSkeleton.h
	../NanoMito3D/AnalyzeSkeleton.cpp
	../NanoMito3D/SkeletonGraph.h
	../NanoMito3D/SkeletonGraph.cpp
	../NanoMito3D/Skeletonize3D.h
	../NanoMito3D/Skeletonize3D.cpp
)
	
python_extension_module(${CMAKE_PROJECT_NAME})

include_directories(${CUDA_INCLUDE_DIRS} "../NanoMito3D/" "../Common/")

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "50")

add_subdirectory("../NanoMito3D/3dparty/cuNSearch" "cuNSearch")
add_subdirectory("../common" "common")

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC
                           ${NumPy_INCLUDE_DIRS}
                           ${PYTHON_INCLUDE_DIRS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${CUDA_LIBRARIES} cuNSearch Common)

install(TARGETS ${CMAKE_PROJECT_NAME} LIBRARY DESTINATION PyNanoMito3D)

add_custom_command(TARGET ${CMAKE_PROJECT_NAME} POST_BUILD        # Adds a post-build event to MyTest
	COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:${CMAKE_PROJECT_NAME}> "${CMAKE_CURRENT_SOURCE_DIR}"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:protobuf::libprotobuf-lite> "${CMAKE_CURRENT_SOURCE_DIR}"
)

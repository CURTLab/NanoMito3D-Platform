cmake_minimum_required(VERSION 3.12)
#Requires cmake 3.12 for first class cuda support with visual studio

project(cuNSearch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set (INCLUDE_HEADERS
    include/PointSet.h
    include/ActivationTable.h
    include/Common.h
    include/cuNSearch.h
)

set (HEADER_FILES
    src/Types.h
    src/cuNSearchDeviceData.h
    src/GridInfo.h
    src/NotImplementedException.h
    src/PointSetImplementation.h
    src/cuNSearchKernels.cuh
    src/helper_linearIndex.h
    src/helper_mortonCode.h
    Utils/cuda_helper.h
)

set (SOURCE_FILES
    src/PointSet.cu
    src/PointSetImplementation.cu
    src/cuNSearch.cu
    src/cuNSearchDeviceData.cu
    src/cuNSearchKernels.cu
    Utils/cuda_helper.cu
)

add_library(cuNSearch STATIC ${INCLUDE_HEADERS} ${HEADER_FILES} ${SOURCE_FILES})

set_target_properties(cuNSearch PROPERTIES CUDA_ARCHITECTURES "50")

target_include_directories(cuNSearch PUBLIC
    "include"
    "Utils"
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/cuNSearch
)
target_link_libraries(cuNSearch PUBLIC ${CUDA_LIBRARIES})
target_compile_definitions(cuNSearch PUBLIC $<$<CONFIG:DEBUG>:DEBUG>)

unset(USE_DOUBLE_PRECISION CACHE)

cmake_minimum_required(VERSION 3.12)
#Requires cmake 3.12 for first class cuda support with visual studio

project(CompactNSearch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set (INCLUDE_HEADERS
    CompactNSearch.h
    Config.h
    DataStructures.h
    PointSet.h
)

set (SOURCE_FILES
    CompactNSearch.cpp
)

add_library(CompactNSearch STATIC ${INCLUDE_HEADERS} ${HEADER_FILES} ${SOURCE_FILES})

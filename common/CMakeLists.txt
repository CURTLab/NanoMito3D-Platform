cmake_minimum_required(VERSION 3.12)
#Requires cmake 3.12 for first class cuda support with visual studio

project(Common LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

set (PROJECT_SOURCES
    Bounds.h
    Localizations.cpp
    Localizations.h
    Octree.h
    Types.h
)

# Git Version
IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../.git)
  FIND_PACKAGE(Git)
  IF(GIT_FOUND)
    EXECUTE_PROCESS(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE "BUILD_VERSION"
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    EXECUTE_PROCESS(
        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE "BUILD_BRANCH"
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    MESSAGE( STATUS "Git version: ${BUILD_VERSION}, branch: ${BUILD_BRANCH}" )
  ELSE(GIT_FOUND)
    SET(BUILD_VERSION 0)
    SET(BUILD_BRANCH "main")
  ENDIF(GIT_FOUND)
ENDIF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../.git)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/Version.h @ONLY)
list(APPEND PROJECT_SOURCES Version.h)

# Protobuf stuff
include_directories(${Protobuf_INCLUDE_DIRS})

message(STATUS "Using Protocol Buffers ${Protobuf_VERSION}")

set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)

file(GLOB PROTO_FILES "${PROTO_DIR}/*.proto")
foreach(PROTO_FILE ${PROTO_FILES})
    get_filename_component (FILENAME ${PROTO_FILE} NAME_WLE)
    set(PROTO_SRC "${OUTPUT_DIR}/${FILENAME}.pb.cc")
    set(PROTO_HDR "${OUTPUT_DIR}/${FILENAME}.pb.h")

    add_custom_command(
        OUTPUT "${PROTO_SRC}" "${PROTO_HDR}"
        COMMAND protobuf::protoc
        ARGS --cpp_out "${OUTPUT_DIR}" -I"${PROTO_DIR}" "${PROTO_FILE}"
        COMMENT "Generating ${PROTO_SRC} from ${PROTO_FILE}"
        DEPENDS ${PROTO_FILE}
    )
    list(APPEND PROJECT_SOURCES ${PROTO_SRC})
    list(APPEND PROJECT_SOURCES ${PROTO_HDR})
endforeach()

add_library(Common STATIC ${PROJECT_SOURCES})

target_link_libraries(Common
    PRIVATE
        protobuf::libprotobuf
)

target_compile_definitions(Common PUBLIC $<$<CONFIG:DEBUG>:DEBUG>)

cmake_minimum_required(VERSION 3.12)
#Requires cmake 3.12 for first class cuda support with visual studio

project(Common LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf CONFIG REQUIRED)

set (PROJECT_SOURCES
	Bounds.h
	Localizations.cpp
	Localizations.h
	Octree.h
	Types.h
	proto/TSFProto.proto
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

add_library(Common STATIC ${PROJECT_SOURCES})

# Protobuf stuff
message(STATUS "Using Protocol Buffers ${Protobuf_VERSION}")

set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

target_include_directories(Common PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")

protobuf_generate(
	TARGET Common
	IMPORT_DIRS "${CMAKE_CURRENT_LIST_DIR}/proto"
	PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")

target_link_libraries(Common
	PRIVATE
	protobuf::libprotobuf-lite
)

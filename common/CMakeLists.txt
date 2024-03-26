cmake_minimum_required(VERSION 3.12)
#Requires cmake 3.12 for first class cuda support with visual studio

project(Common LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf CONFIG REQUIRED)

set(RELEASE_VERSION OFF CACHE BOOL "Build as release version (No test paths)")

set (PROJECT_SOURCES
	Bounds.h
	Localizations.cpp
	Localizations.h
	Octree.h
	Types.h
	proto/TSFProto.proto
)

if(RELEASE_VERSION)
	set(BUILD_VERSION 0x00000100)
	set(BUILD_BRANCH "release")
	set(BUILD_APP_VER "v0.1")
else(RELEASE_VERSION)
	# Git Version
	if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../.git)
		find_package(Git)
		if(GIT_FOUND)
			execute_process(
				COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
				WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
				OUTPUT_VARIABLE "BUILD_VERSION"
				ERROR_QUIET
				OUTPUT_STRIP_TRAILING_WHITESPACE)
			execute_process(
				COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
				WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
				OUTPUT_VARIABLE "BUILD_BRANCH"
				ERROR_QUIET
				OUTPUT_STRIP_TRAILING_WHITESPACE)
			message( STATUS "Git version: ${BUILD_VERSION}, branch: ${BUILD_BRANCH}" )
		else(GIT_FOUND)
			set(BUILD_VERSION 0)
			set(BUILD_BRANCH "main")
		endif(GIT_FOUND)
	endif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../.git)
	math(EXPR BUILD_VERSION_DEC "0x${BUILD_VERSION}" OUTPUT_FORMAT DECIMAL)
	set(BUILD_APP_VER "r${BUILD_VERSION_DEC}")
endif(RELEASE_VERSION)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/Version.h @ONLY)
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

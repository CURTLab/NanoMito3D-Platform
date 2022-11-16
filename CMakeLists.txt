cmake_minimum_required(VERSION 3.13)

project(Paper_NanoMito_VTK VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(VTK 8.0 REQUIRED COMPONENTS GUISupportQt)
find_package(OpenCV REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(ZLIB REQUIRED)

set(PROJECT_SOURCES
    main.cpp
    MainWindow.cpp MainWindow.h MainWindow.ui
    VolumeWidget.h VolumeWidget.cpp
)

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


if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(Paper_NanoMito_VTK
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET Paper_NanoMito_VTK APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(Paper_NanoMito_VTK SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(Paper_NanoMito_VTK
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_link_libraries(Paper_NanoMito_VTK
    PRIVATE
        Qt${QT_VERSION_MAJOR}::Widgets
        VTK::GUISupportQt
        ${OpenCV_LIBS}
        protobuf::libprotobuf
)

set_target_properties(Paper_NanoMito_VTK PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

install(TARGETS Paper_NanoMito_VTK
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(Paper_NanoMito_VTK)
endif()

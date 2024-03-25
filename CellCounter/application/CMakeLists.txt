cmake_minimum_required(VERSION 3.13)

project(CellCounterApp VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../widgets ${CMAKE_CURRENT_BINARY_DIR}/widgets)

include_directories("../../widgets")

set(RELEASE_VERSION OFF CACHE BOOL "Build as release version (No test paths)")

set(PROJECT_SOURCES
    main.cpp
    MainWindow.cpp MainWindow.h MainWindow.ui
    CellCounter.h CellCounter.cpp
    LocalMaximumSearch.h LocalMaximumSearch.cpp
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(${CMAKE_PROJECT_NAME}
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
else()
    add_executable(${CMAKE_PROJECT_NAME}
        ${PROJECT_SOURCES}
    )
endif()

target_link_libraries(${CMAKE_PROJECT_NAME}
    PRIVATE
        Qt${QT_VERSION_MAJOR}::Widgets
        ${OpenCV_LIBS}
        Widgets
)

if (RELEASE_VERSION)
	target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE -DRELEASE_VERSION)
else()
	target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE "DEV_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/../\"")
endif()

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

add_custom_command(TARGET CellCounterApp POST_BUILD        # Adds a post-build event to MyTest
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
        "${CMAKE_CURRENT_SOURCE_DIR}/../../examples/CellCounterModel.onnx"      # <--this is in-file
        $<TARGET_FILE_DIR:CellCounterApp>)                 # <--this is out-file path

install(TARGETS ${CMAKE_PROJECT_NAME}
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(${CMAKE_PROJECT_NAME})
endif()

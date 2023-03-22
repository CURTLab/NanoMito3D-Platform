cmake_minimum_required(VERSION 3.12)
#Requires cmake 3.12 for first class cuda support with visual studio

project(Widgets LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake ${CMAKE_MODULE_PATH})

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED)
find_package(Qwt REQUIRED)

set (PROJECT_SOURCES
    ImagePlotWidget.h ImagePlotWidget.cpp
    MatPlotWidget.h MatPlotWidget.cpp
    DrawingImagePlotWidget.h DrawingImagePlotWidget.cpp
)

add_library(Widgets STATIC ${PROJECT_SOURCES})

target_link_libraries(Widgets
    PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    Qwt::Qwt
    ${OpenCV_LIBS}
)

target_compile_definitions(Widgets PUBLIC $<$<CONFIG:DEBUG>:DEBUG>)

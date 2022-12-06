cmake_minimum_required(VERSION 3.13)

project(NanoMito3D VERSION 0.1 LANGUAGES CUDA CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(VTK 8.0 REQUIRED COMPONENTS GUISupportQt RenderingVolumeOpenGL2 CommonComputationalGeometry RenderingAnnotation)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(CUDA 7.0 REQUIRED)

add_subdirectory(3dparty/cuNSearch)

set(PROJECT_SOURCES
    kernels/DensityFilterGPU.cu
    kernels/DensityFilterCPU.cpp
    kernels/Device.cu
    kernels/GaussianFilterGPU.cu
    kernels/RenderingCPU.cpp
    kernels/RenderingGPU.cu
    kernels/LocalThresholdGPU.cu
    kernels/LocalThresholdCPU.cpp
    main.cpp
    MainWindow.cpp MainWindow.h MainWindow.ui
    VolumeWidget.h VolumeWidget.cpp
    Volume.h Volume.cpp
    Device.h
    DensityFilter.h
    GaussianFilter.h
    Rendering.h
    LocalThreshold.h
    Skeletonize3D.h Skeletonize3D.cpp
    AnalyzeSkeleton.h AnalyzeSkeleton.cpp
    SkeletonGraph.h SkeletonGraph.cpp
    Segments.h Segments.cpp
    AnalyzeMitochindra.h AnalyzeMitochindra.cpp
)

include_directories(${CUDA_INCLUDE_DIRS} "../common")

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
        ${VTK_LIBRARIES}
        ${OpenCV_LIBS}
        cuNSearch
        Common
)

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "50")

# VTK autoinit
vtk_module_autoinit(
  TARGETS ${CMAKE_PROJECT_NAME}
  MODULES ${VTK_LIBRARIES}
)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE "DEV_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/..\"")

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

install(TARGETS ${CMAKE_PROJECT_NAME}
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(Paper_NanoMito_VTK)
endif()

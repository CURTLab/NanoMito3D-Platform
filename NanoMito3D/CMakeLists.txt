cmake_minimum_required(VERSION 3.13)

project(NanoMito3D VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_SUPPORT ON CACHE BOOL "Use CUDA support")

if (CUDA_SUPPORT)
	enable_language(CUDA)

	set(CMAKE_CUDA_STANDARD 14)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)

	add_subdirectory(3dparty/cuNSearch)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake ${CMAKE_MODULE_PATH})

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(VTK 8.0 REQUIRED COMPONENTS
	GUISupportQt RenderingVolumeOpenGL2 CommonComputationalGeometry RenderingAnnotation IOExport IOGeometry
)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)
find_package(HDF5)

add_subdirectory(3dparty/CompactNSearch)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../common ${CMAKE_CURRENT_BINARY_DIR}/common)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../widgets ${CMAKE_CURRENT_BINARY_DIR}/widgets)

set(PROJECT_SOURCES
	kernels/DensityFilterCPU.cpp
	kernels/GaussianFilterCPU.cpp
	kernels/RenderingCPU.cpp
	kernels/LocalThresholdCPU.cpp
	kernels/LocalThreshold_impl.h
	kernels/Device.cpp
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
	Segments.h
	AnalyzeMitochondria.h AnalyzeMitochondria.cpp
)

if (CUDA_SUPPORT)
	list(APPEND PROJECT_SOURCES
		kernels/DensityFilterGPU.cu
		kernels/GaussianFilterGPU.cu
		kernels/RenderingGPU.cu
		kernels/LocalThresholdGPU.cu
	)
endif()

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
	CompactNSearch
	Common
	Widgets
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE "../common" "../widgets")

if (HDF5_FOUND)
	message(STATUS "Support hdf5 features")
	target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE HDF5::HDF5)
	target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE "DEV_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/../\"")
endif()

if (CUDA_SUPPORT)
	set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "50")
	target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE cuNSearch)
	target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE -DCUDA_SUPPORT)
	target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# VTK autoinit
vtk_module_autoinit(
	TARGETS ${CMAKE_PROJECT_NAME}
	MODULES ${VTK_LIBRARIES}
)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE "DEV_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/../\"" -D_USE_MATH_DEFINES -DNOMINMAX)

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
	qt_finalize_executable(${CMAKE_PROJECT_NAME})
endif()

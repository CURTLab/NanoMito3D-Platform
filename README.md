# NanoMito3D-Platform

NanoMito3D-Platform contains software applications for:
 * CellCounterApp: Automated cell counting from phase constrast image
 * BleedThroughCorrection: Fluorescent bleed-through correction from two-color 3D single molecule localization microscopy files
 * NanoMito3D: 3D mitochondrial network analysis with GPU acceleration from (multicolor) 3D single molecule localization microscopy files
 
 ## NanoMito3D Application
![Thumbnail](https://raw.githubusercontent.com/CURTLab/NanoMito3D-Platform/main/thumbnail_nanomito3D.png)

### NanoMito3D is based on
* [Skeletonize3D](https://imagej.net/plugins/skeletonize3d) (ImageJ Plugin)
* [AnalyzeSkeleton](https://imagej.net/plugins/analyze-skeleton) (ImageJ Plugin)
* [cuNSearch](https://github.com/InteractiveComputerGraphics/cuNSearch) (compute neighborhood information on GPU)

## CellCounter Application
![Thumbnail](https://raw.githubusercontent.com/CURTLab/NanoMito3D-Platform/main/thumbnail_cellcounter.png)

# Tested prerequisites for compilation
* Windows 10
* Visual Studio 2019
* QT 5.15.2
* OpenCV 4.5.5
* VTK 9.0
* Qwt 6.1.6
* protobuf 3.21.9
* CUDA 7.0
* CMake 3.18.1

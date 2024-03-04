# NanoMito3D-Platform

NanoMito3D-Platform contains software applications for:
 * CellCounterApp: Deep Learning assisted cell counting from phase constrast images
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

## BleedThroughCorrection Application
![Thumbnail](https://raw.githubusercontent.com/CURTLab/NanoMito3D-Platform/main/thumbnail_bleedthroughcorr.PNG)

# 3rd-party libaries:
* [cuNSearch](https://github.com/InteractiveComputerGraphics/cuNSearch)
* [CompactNSearch](https://github.com/InteractiveComputerGraphics/CompactNSearch)

# Tested prerequisites for compilation
* Windows 10
* Visual Studio 2019
* QT 5.15.2/6.4.2
* OpenCV 4.5.5
* VTK 9.0/9.2.6
* Qwt 6.1.6/6.2.0
* protobuf 21.12/3.21.9
* CUDA 7.0
* CMake 3.18.1

[Protobuf version 21.12](https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protobuf-all-21.12.zip) is recommended, since starting with version 22 protobuf is using Abseil (which gave me problems).
Furthermore, [`TSFProto.proto`](https://github.com/nicost/TSFProto/blob/master/src/TSFProto.proto) by Nico Stuurman is modified to support protobuf-lite and the syntax is set to `proto2`.

/****************************************************************************
 *
 * Copyright (C) 2022 Fabian Hauser
 *
 * Author: Fabian Hauser <fabian.hauser@fh-linz.at>
 * University of Applied Sciences Upper Austria - Linz - Austra
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************/

#include "VolumeWidget.h"

#include <QGridLayout>

#include <QVTKOpenGLNativeWidget.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

#include <vtkCamera.h>
#include <vtkDataSetMapper.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkProperty.h>
#include <vtkRenderWindowInteractor.h>

#include <vtkVolume.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkNamedColors.h>
#include <vtkStructuredPoints.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkVolumeProperty.h>
#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>
#include <vtkSphereSource.h>
#include <vtkGlyph3D.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>

#include <vtkParametricSpline.h>
#include <vtkParametricFunctionSource.h>
#include <vtkTubeFilter.h>

#include <vtkCubeAxesActor.h>
#include <vtkTextProperty.h>

class VolumeWidgetPrivate : public QVTKOpenGLNativeWidget
{
public:
	inline VolumeWidgetPrivate(QWidget *parent) : QVTKOpenGLNativeWidget(parent) {}


	vtkSmartPointer<vtkRenderer> renderer;
};

VolumeWidget::VolumeWidget(QWidget *parent)
	: m_d(new VolumeWidgetPrivate(parent))
{
	Q_D(VolumeWidget);

	QGridLayout *layout = new QGridLayout;
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(d);
	setLayout(layout);

	vtkNew<vtkGenericOpenGLRenderWindow> window;
	d->setRenderWindow(window.Get());

	// Camera
	vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
	camera->SetViewUp(0, 1, 0);
	camera->SetPosition(0, 0, 10);
	camera->SetFocalPoint(0, 0, 0);

	// Renderer
	d->renderer = vtkSmartPointer<vtkRenderer>::New();
	d->renderer->SetActiveCamera(camera);
	d->renderer->SetBackground(0.5, 0.5, 0.5);
	d->renderWindow()->AddRenderer(d->renderer);
}

VolumeWidget::~VolumeWidget()
{
	delete m_d;
}

void VolumeWidget::clear()
{
	Q_D(VolumeWidget);
	d->renderer->RemoveAllViewProps();
	d->renderer->ResetCamera();
	d->renderWindow()->Render();
}

void VolumeWidget::setVolume(Volume volume, std::array<double, 4> color, bool copyData)
{
	Q_D(VolumeWidget);

	const size_t numPts = volume.voxels();

	vtkNew<vtkUnsignedCharArray> scalars;
	if (copyData)
		std::copy_n(volume.data(), numPts, scalars->WritePointer(0, numPts));
	else
		scalars->SetArray(volume.data(), numPts, 1);

	vtkNew<vtkStructuredPoints> imageData;
	imageData->GetPointData()->SetScalars(scalars);
	imageData->SetDimensions(volume.width(), volume.height(), volume.depth());
	imageData->SetOrigin(volume.origin()[0] * 1E-3, volume.origin()[1] * 1E-3, volume.origin()[2] * 1E-3);
	imageData->SetSpacing(volume.voxelSize()[0] * 1E-3, volume.voxelSize()[1] * 1E-3, volume.voxelSize()[2] * 1E-3);

	vtkNew<vtkSmartVolumeMapper> volumeMapper;
	//volumeMapper->SetBlendModeToIsoSurface();
	//volumeMapper->SetBlendModeToComposite(); // composite first
	volumeMapper->SetBlendModeToMaximumIntensity();
	volumeMapper->SetInputData(imageData);

	vtkNew<vtkVolumeProperty> volumeProperty;
	volumeProperty->ShadeOff();
	volumeProperty->SetInterpolationType(VTK_LINEAR_INTERPOLATION);

	vtkNew<vtkPiecewiseFunction> compositeOpacity;
	compositeOpacity->AddPoint(0.0, 0.0);
	compositeOpacity->AddPoint(255.0, color[3]);
	volumeProperty->SetScalarOpacity(compositeOpacity); // composite first.

	vtkNew<vtkColorTransferFunction> colortf;
	colortf->AddRGBPoint(0.0, color[0], color[1], color[2]);
	volumeProperty->SetColor(colortf);

	vtkNew<vtkVolume> vol;
	vol->SetMapper(volumeMapper);
	vol->SetProperty(volumeProperty);

	vtkNew<vtkCubeAxesActor> cubeAxesActor;

	cubeAxesActor->SetXLabelFormat("%-#3.1f");
	cubeAxesActor->SetYLabelFormat("%-#3.1f");
	cubeAxesActor->SetZLabelFormat("%-#3.1f");

	cubeAxesActor->SetUseTextActor3D(1);
	cubeAxesActor->SetBounds(vol->GetBounds());
	cubeAxesActor->SetCamera(d->renderer->GetActiveCamera());
	cubeAxesActor->SetXTitle("X / µm");
	cubeAxesActor->SetYTitle("Y / µm");
	cubeAxesActor->SetZTitle("Z / µm");
	//cubeAxesActor->SetScreenSize(10);
	cubeAxesActor->SetEnableDistanceLOD(0);
	cubeAxesActor->SetEnableViewAngleLOD(0);

#if 1
	vtkNew<vtkStringArray> axisLabels;
	vtkStdString s;
	s.resize(64);
	snprintf(s.data(), 64, "%3.1f", vol->GetBounds()[4]);
	axisLabels->InsertNextValue(s);
	snprintf(s.data(), 64, "%3.1f", vol->GetBounds()[5]);
	axisLabels->InsertNextValue(s);
	cubeAxesActor->SetAxisLabels(2, axisLabels);
#endif

	cubeAxesActor->DrawXGridlinesOn();
	cubeAxesActor->DrawYGridlinesOn();
	cubeAxesActor->DrawZGridlinesOn();
#if VTK_MAJOR_VERSION == 6
	cubeAxesActor->SetGridLineLocation(VTK_GRID_LINES_FURTHEST);
#elif VTK_MAJOR_VERSION > 6
	cubeAxesActor->SetGridLineLocation(cubeAxesActor->VTK_GRID_LINES_FURTHEST);
#endif

	cubeAxesActor->XAxisMinorTickVisibilityOff();
	cubeAxesActor->YAxisMinorTickVisibilityOff();
	cubeAxesActor->ZAxisMinorTickVisibilityOff();

	cubeAxesActor->SetFlyModeToOuterEdges();
	d->renderer->AddActor(cubeAxesActor);

	//cubeAxesActor->PrintSelf(std::cout, vtkIndent(0));

	d->renderer->AddViewProp(vol);
	d->renderer->ResetCamera();
	d->renderWindow()->Render();
}

void VolumeWidget::addSpheres(const std::vector<std::array<float, 3> > &points, float r, std::array<double, 3> color)
{
	Q_D(VolumeWidget);

	vtkNew<vtkPoints> pts;
	for (size_t i = 0; i < points.size(); ++i)
		pts->InsertPoint(i, points[i][0] * 1E-3, points[i][1] * 1E-3, points[i][2] * 1E-3);

	vtkNew<vtkPolyData> profile;
	profile->SetPoints(pts);

	vtkNew<vtkSphereSource> sphereSource;
	sphereSource->SetRadius(r * 1E-3);
	sphereSource->Update();

	vtkNew<vtkGlyph3D> balls;
	balls->SetInputData(profile);
	balls->SetSourceConnection(sphereSource->GetOutputPort());
	balls->Update();

	vtkNew<vtkPolyDataMapper> mapBalls;
	mapBalls->SetInputConnection(balls->GetOutputPort());
	mapBalls->Update();

	vtkNew<vtkActor> ballActor;
	ballActor->SetMapper(mapBalls);
	ballActor->GetProperty()->SetColor(color[0], color[1], color[2]);

	d->renderer->AddActor(ballActor);
	d->renderer->ResetCamera();
	d->renderWindow()->Render();
}

void VolumeWidget::addGraph(std::shared_ptr<SkeletonGraph> graph, const Volume &volume, float r, std::array<double,3> color)
{
	Q_D(VolumeWidget);

	auto edges = graph->edgeList();
	if (edges.empty())
		return;

	for (const auto &edge : edges) {
		vtkNew<vtkPoints> points;
		if (edge->v1 >= 0) {
			const auto v1 = graph->vertex(edge->v1);
			auto pos = volume.mapVoxel(v1->firstPoint().x, v1->firstPoint().y, v1->firstPoint().z, true);
			points->InsertNextPoint(pos[0] * 1E-3, pos[1] * 1E-3, pos[2] * 1E-3);
		}

		for (int i = 0; i < edge->slab.size(); ++i) {
			auto pos = volume.mapVoxel(edge->slab[i].x, edge->slab[i].y, edge->slab[i].z, true);
			points->InsertNextPoint(pos[0] * 1E-3, pos[1] * 1E-3, pos[2] * 1E-3);
		}

		if (edge->v2 >= 0) {
			const auto v2 = graph->vertex(edge->v2);
			auto pos = volume.mapVoxel(v2->firstPoint().x, v2->firstPoint().y, v2->firstPoint().z, true);
			points->InsertNextPoint(pos[0] * 1E-3, pos[1] * 1E-3, pos[2] * 1E-3);
		}

		// Fit a spline to the points
		vtkNew<vtkParametricSpline> spline;
		spline->SetPoints(points);

		vtkNew<vtkParametricFunctionSource> functionSource;
		functionSource->SetParametricFunction(spline);
		//functionSource->SetUResolution(10 * points->GetNumberOfPoints());
		functionSource->Update();

		// Create the tubes
		vtkNew<vtkTubeFilter> tuber;
		tuber->SetInputData(functionSource->GetOutput());
		tuber->SetNumberOfSides(6);
		tuber->SetRadius(r * 1E-3); // in µm

		vtkNew<vtkPolyDataMapper> tubeMapper;
		tubeMapper->SetInputConnection(tuber->GetOutputPort());

		vtkNew<vtkActor> actor;
		actor->SetMapper(tubeMapper);
		actor->GetProperty()->SetColor(color[0], color[1], color[2]);

		d->renderer->AddActor(actor);
	}
	d->renderer->ResetCamera();
	d->renderWindow()->Render();
}

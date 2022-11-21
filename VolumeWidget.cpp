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

void VolumeWidget::setVolume(Volume volume)
{
	Q_D(VolumeWidget);

	const size_t numPts = volume.voxels();

	vtkNew<vtkUnsignedCharArray> scalars;
	scalars->SetArray(volume.data(), numPts, 1);
	//auto s = scalars->WritePointer(0, numPts);
	//std::copy_n(volume.data(), numPts, s);

	vtkNew<vtkStructuredPoints> imageData;
	imageData->GetPointData()->SetScalars(scalars);
	imageData->SetDimensions(volume.width(), volume.height(), volume.depth());
	imageData->SetOrigin(volume.origin()[0], volume.origin()[1], volume.origin()[2]);
	imageData->SetSpacing(volume.voxelSize()[0], volume.voxelSize()[1], volume.voxelSize()[2]);

	vtkNew<vtkSmartVolumeMapper> volumeMapper;
	//volumeMapper->SetBlendModeToIsoSurface();
	volumeMapper->SetBlendModeToComposite(); // composite first
	volumeMapper->SetInputData(imageData);

	vtkNew<vtkVolumeProperty> volumeProperty;
	volumeProperty->ShadeOff();
	volumeProperty->SetInterpolationType(VTK_LINEAR_INTERPOLATION);

	vtkNew<vtkPiecewiseFunction> compositeOpacity;
	compositeOpacity->AddPoint(0.0, 0.0);
	compositeOpacity->AddPoint(255.0, 1.0);
	volumeProperty->SetScalarOpacity(compositeOpacity); // composite first.

	vtkNew<vtkColorTransferFunction> color;
	color->AddRGBPoint(0.0, 0.0, 0.0, 1.0);
	color->AddRGBPoint(255.0, 0.0, 0.0, 1.0);
	volumeProperty->SetColor(color);

	vtkNew<vtkVolume> vol;
	vol->SetMapper(volumeMapper);
	vol->SetProperty(volumeProperty);

	d->renderer->AddViewProp(vol);
	d->renderer->ResetCamera();
	d->renderWindow()->Render();
}

void VolumeWidget::addSpheres(const std::vector<std::array<float, 3> > &points, float r, std::array<double, 3> color)
{
	Q_D(VolumeWidget);

	vtkNew<vtkPoints> pts;
	for (size_t i = 0; i < points.size(); ++i)
		pts->InsertPoint(i, points[i][0], points[i][1], points[i][2]);

	vtkNew<vtkPolyData> profile;
	profile->SetPoints(pts);

	vtkNew<vtkSphereSource> sphereSource;
	sphereSource->SetRadius(r);
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

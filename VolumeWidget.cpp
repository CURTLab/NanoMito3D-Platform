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


#include <vtkCamera.h>
#include <vtkDataSetMapper.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkProperty.h>
#include <vtkRenderWindowInteractor.h>

VolumeWidget::VolumeWidget(QWidget *parent)
	: QVTKOpenGLNativeWidget(parent)
{
	vtkNew<vtkGenericOpenGLRenderWindow> window;
	setRenderWindow(window.Get());

	// Camera
	vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
	camera->SetViewUp(0, 1, 0);
	camera->SetPosition(0, 0, 10);
	camera->SetFocalPoint(0, 0, 0);

	// Renderer
	m_renderer = vtkSmartPointer<vtkRenderer>::New();
	m_renderer->SetActiveCamera(camera);
	m_renderer->SetBackground(0.5, 0.5, 0.5);
	renderWindow()->AddRenderer(m_renderer);
}

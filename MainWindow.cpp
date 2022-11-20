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

#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "Localizations.h"
#include "DensityFilter.h"
#include "Device.h"
#include "GaussianFilter.h"
#include "Rendering.h"

#include <chrono>
#include <iostream>

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, m_ui(new Ui::MainWindow)
{
	m_ui->setupUi(this);

	GPU::initGPU();
}

MainWindow::~MainWindow()
{
}

void MainWindow::showEvent(QShowEvent *event)
{
	QMainWindow::showEvent(event);

//#ifdef QT_DEBUG
	try {
		auto start = std::chrono::steady_clock::now();

		Localizations locs;
		//locs.load(DEV_PATH "/examples/WOP_CD62p_AntiMito_C1000_dual_Cell4_dSTORM_red_blue_031_v3.tsf");
		locs.load(DEV_PATH "/examples/WOP_CD62p_AntiMito_C1000_dual_dSTORM_red_blue_034_v3.tsf");
		m_ui->statusbar->showMessage(tr("Loaded %1 localizations from file. %2 x %3 µm²").arg(locs.size()).arg(locs.width()*1E-3).arg(locs.height()*1E-3));

		const std::array<float,3> voxelSize{85.f, 85.f, 25.f}; // nm

		auto dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
		std::cout << "Locs: " << locs.size() << " in " << dur.count() << " s" << std::endl;

#if 1
		// filter localizations by channel and PA
		start = std::chrono::steady_clock::now();

		locs.erase(std::remove_if(locs.begin(), locs.end(), [](const Localization &l) {
			return (l.channel == 1 || l.PAx > 100 || l.PAy > 100 || l.PAz > 150);
		}), locs.end());

		dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
		std::cout << "Filtered: " << locs.size() << " in " << dur.count() << " s" << std::endl;
#endif

#if 0
		// filter by density CPU
		start = std::chrono::steady_clock::now();

		locs.erase(DensityFilter::remove_cpu(locs, 10, 250.f), locs.end());

		end = std::chrono::steady_clock::now();
		dur = std::chrono::duration<double>(end - start);

		std::cout << "Density filter (CPU): " << locs.size()  << " in " << dur.count() << " s" << std::endl;
#else
		// filter by density GPU
		start = std::chrono::steady_clock::now();
		locs.erase(DensityFilter::remove_gpu(locs, 10, 250.f), locs.end());

		dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

		std::cout << "Density filter (GPU): " << locs.size()  << " in " << dur.count() << " s" << std::endl;
#endif

		// 3D rendering
		start = std::chrono::steady_clock::now();

		//m_volume = Rendering::render_cpu(locs, voxelSize, 5);

		locs.copyTo(DeviceType::Device);
		m_volume = Rendering::render_gpu(locs, voxelSize, 5);

		m_filteredVolume = Volume(m_volume.size(), voxelSize, {0.f, 0.f, locs.minZ()});

		dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

		std::cout << "Rendering (GPU): " << dur.count() << " s" << std::endl;

#if 1
		// gaussian filter 3D
		start = std::chrono::steady_clock::now();

		GaussianFilter::gaussianFilter_gpu(m_volume.constData(), m_filteredVolume.data(), m_volume.width(), m_volume.height(), m_volume.depth(), 7, 1.5f);

		dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

		std::cout << "Gaussian filter (GPU): " << dur.count() << " s" << std::endl;
#endif

		m_ui->widget->setVolume(m_filteredVolume);
	} catch(std::exception &e) {
		m_ui->statusbar->showMessage(tr("Error: ") + e.what());
		std::cerr << std::string("Error: ") + e.what() << std::endl;
	}
//#endif
}

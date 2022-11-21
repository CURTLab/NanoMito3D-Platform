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
#include "LocalThreshold.h"
#include "Skeletonize3D.h"
#include "AnalyzeSkeleton.h"

#include <chrono>
#include <iostream>

#include <QMessageBox>

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

	try {
#if 0
		Localizations locs;
		//locs.load(DEV_PATH "/examples/WOP_CD62p_AntiMito_C1000_dual_Cell4_dSTORM_red_blue_031_v3.tsf");
		locs.load(DEV_PATH "/examples/WOP_CD62p_AntiMito_C1000_dual_dSTORM_red_blue_034_v3.tsf");
		m_ui->statusbar->showMessage(tr("Loaded %1 localizations from file. %2 x %3 µm²").arg(locs.size()).arg(locs.width()*1E-3).arg(locs.height()*1E-3));

		const std::array<float,3> voxelSize{85.f, 85.f, 25.f}; // nm
		m_volume = render(locs, voxelSize, {100.f, 100.f, 150.f}, 2);
		analyse(m_volume, 1.5);
#else
		m_volume = Volume::loadTif(DEV_PATH "/examples/bat-cochlea-volume.tif");
		analyse(m_volume, 1.5f);
#endif
	} catch(std::exception &e) {
		QMessageBox::critical(this, "Error", e.what());
		std::cerr << std::string("Error: ") + e.what() << std::endl;
	}

	//analyse(m_volume);
}

Volume MainWindow::render(Localizations &locs, std::array<float, 3> voxelSize, std::array<float, 3> maxPA, int channel)
{
	Volume ret;
	try {
#if 1
		// filter localizations by channel and PA
		auto start = std::chrono::steady_clock::now();

		locs.erase(std::remove_if(locs.begin(), locs.end(), [&maxPA,channel](const Localization &l) {
			if (channel > 0 && l.channel != channel)
				return true;
			return (l.PAx > maxPA[0] || l.PAy > maxPA[1] || l.PAz > maxPA[2]);
		}), locs.end());

		auto dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
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

		locs.copyTo(DeviceType::Device);
		ret = Rendering::render_gpu(locs, voxelSize, 5);

		dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

		std::cout << "Rendering (GPU): " << dur.count() << " s" << std::endl;
	} catch(std::exception &e) {
		QMessageBox::critical(this, "Rendering error", e.what());
		std::cerr << std::string("Rendering Error: ") + e.what() << std::endl;
	}

	return ret;
}

void MainWindow::analyse(Volume &volume, float sigma)
{
	try {
		Volume filteredVolume(volume.size(), volume.voxelSize(), volume.origin());

		// gaussian filter 3D
		auto start = std::chrono::steady_clock::now();

		const int windowSize = (int)(sigma * 4) | 1;
		// default 7
		GaussianFilter::gaussianFilter_gpu(volume.constData(), filteredVolume.data(), volume.width(), volume.height(), volume.depth(), windowSize, sigma);

		auto dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

		std::cout << "Gaussian filter (GPU): " << dur.count() << " s" << std::endl;

		// local thresholding 3D
		start = std::chrono::steady_clock::now();

		LocalThreshold::localThrehsold_gpu(LocalThreshold::IsoData, filteredVolume, filteredVolume, 11);

		dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

		std::cout << "Local threshold filter (GPU): " << dur.count() << " s" << std::endl;

		// skeleton 3D
		start = std::chrono::steady_clock::now();

		Skeleton3D::skeletonize(filteredVolume, m_skeleton);

		dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

		std::cout << "Skeltonize (CPU): " << dur.count() << " s" << std::endl;

		// analyse skeleton 3D
		start = std::chrono::steady_clock::now();

		Skeleton3D::Analysis analysis;
		auto trees = analysis.calculate(m_skeleton, {}, Skeleton3D::Analysis::NoPruning, true, 0.0, true);

		dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

		std::cout << "Analyse skeleton (CPU): " << dur.count() << " s" << std::endl;

		std::vector<std::array<float,3>> endPoints;
		for (const auto &t : trees) {
			for (const auto &p : t.endPoints)
				endPoints.push_back(std::array<float,3>{p.x*volume.voxelSize()[0] + volume.origin()[0],
																	 p.y*volume.voxelSize()[1] + volume.origin()[1],
																	 p.z*volume.voxelSize()[2] + volume.origin()[2]});
		}
		m_ui->widget_2->addSpheres(endPoints, std::max({volume.voxelSize()[0], volume.voxelSize()[1], volume.voxelSize()[2]}), {1.f,0.f,0.f});

		m_ui->widget->setVolume(filteredVolume);
		m_ui->widget_2->setVolume(analysis.taggedImage());
	} catch(std::exception &e) {
		QMessageBox::critical(this, "Analyse error", e.what());
		std::cerr << std::string("Analyse error: ") + e.what() << std::endl;
	}
}

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
#include "Octree.h"

#include <chrono>
#include <iostream>

void drawPSF(uint8_t *imageData, const Localization &l, const int volumeDims[3], const std::array<float,3> &voxelSize, int windowSize)
{
	const int ix = qRound((l.x / voxelSize[0]));
	const int iy = qRound((l.y / voxelSize[1]));
	const int iz = qRound((l.z / voxelSize[2]));

	const int w = windowSize/2;
	for (int z = -w; z <= w; ++z) {
		for (int y = -w; y <= w; ++y) {
			for (int x = -w; x <= w; ++x) {
				if ((ix + x < 0) || (iy + y < 0) || (iz + z < 0) ||
					 (ix + x >= volumeDims[0]) || (iy + y >= volumeDims[1]) || (iz + z >= volumeDims[2]))
					continue;
				const double tx = ((ix + x) * voxelSize[0] - l.x) / l.PAx;
				const double ty = ((iy + y) * voxelSize[1] - l.y) / l.PAy;
				const double tz = ((iz + z) * voxelSize[2] - l.z) / l.PAz;
				const double e = (255.0/windowSize)*exp(-0.5 * tx * tx -0.5 * ty * ty -0.5 * tz * tz);
				auto &val = imageData[ix + x + volumeDims[0] * (iy + y) + volumeDims[0] * volumeDims[1] * (iz + z)];
				val = qBound(0.0, val + e, 255.0);
			}
		}
	}
}

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, m_ui(new Ui::MainWindow)
{
	m_ui->setupUi(this);
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
		locs.load(DEV_PATH "/examples/WOP_CD62p_AntiMito_C1000_dual_dSTORM_red_blue_034_v3.tsf");
		m_ui->statusbar->showMessage(tr("Loaded %1 localizations from file. %2 x %3 µm²").arg(locs.size()).arg(locs.width()*1E-3).arg(locs.height()*1E-3));

		const std::array<float,3> voxelSize{85.f, 85.f, 25.f}; // nm

		int dims[3];
		dims[0] = static_cast<int>(std::ceilf(locs.width()  / voxelSize[0]));
		dims[1] = static_cast<int>(std::ceilf(locs.height() / voxelSize[1]));
		dims[2] = static_cast<int>(std::ceilf(locs.depth()  / voxelSize[2]));

		auto end = std::chrono::steady_clock::now();
		auto dur = std::chrono::duration<double>(end - start);
		std::cout << "Locs: " << locs.size() << " in " << dur.count() << " s" << std::endl;

		// filter localizations by channel and PA
		start = std::chrono::steady_clock::now();

		locs.erase(std::remove_if(locs.begin(), locs.end(), [](const Localization &l) {
			return (l.channel == 1 || l.PAx > 100 || l.PAy > 100 || l.PAz > 150);
		}), locs.end());

		end = std::chrono::steady_clock::now();
		dur = std::chrono::duration<double>(end - start);

		std::cout << "Filtered: " << locs.size() << " in " << dur.count() << " s" << std::endl;

#if 1
		// filter by density
		start = std::chrono::steady_clock::now();
		Octree<uint32_t,float,50> tree(locs.bounds());
		for (uint32_t i = 0; i < locs.size(); ++i)
			tree.insert(locs[i].position(), i);
		std::cout << "Octree: " << tree.size() << std::endl;

		locs.erase(std::remove_if(locs.begin(), locs.end(), [&tree](const Localization &l) {
			const float radius = 250;
			int minPoints = 10;
			const auto pts = tree.countInSphere(l.position(), radius);
			return pts < minPoints;
		}), locs.end());

		end = std::chrono::steady_clock::now();
		dur = std::chrono::duration<double>(end - start);

		std::cout << "Filtered2: " << locs.size()  << " in " << dur.count() << " s" << std::endl;
#endif

		m_volume = Volume(dims, voxelSize, {0.f, 0.f, locs.minZ()});
		m_volume.fill(0);
		for (const auto &l : locs) {
			drawPSF(m_volume.data(), l, dims, voxelSize, 5);
		}

		m_ui->widget->setVolume(m_volume);
	} catch(std::exception &e) {
		m_ui->statusbar->showMessage(tr("Error: ") + e.what());
	}
//#endif
}

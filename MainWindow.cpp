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
#include "Segments.h"
#include "Octree.h"

#include <chrono>
#include <QDebug>

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
#if 1
		Localizations locs;
		//locs.load(DEV_PATH "/examples/WOP_CD62p_AntiMito_C1000_dual_Cell4_dSTORM_red_blue_031_v3.tsf");
		locs.load(DEV_PATH "/examples/WOP_CD62p_AntiMito_C1000_dual_dSTORM_red_blue_034_v3.tsf");
		m_ui->statusbar->showMessage(tr("Loaded %1 localizations from file. %2 x %3 µm²").arg(locs.size()).arg(locs.width()*1E-3).arg(locs.height()*1E-3));
		qDebug() << "Loaded" << locs.size() << "localizations from file";

		const std::array<float,3> voxelSize{85.f, 85.f, 25.f}; // nm
		m_volume = render(locs, voxelSize, {100.f, 100.f, 200.f}, 2);

		analyse(m_volume, locs, 1.5);
#else
		m_volume = Volume::loadTif(DEV_PATH "/examples/bat-cochlea-volume.tif");
		analyse(m_volume, 1.5f);
#endif
	} catch(std::exception &e) {
		QMessageBox::critical(this, "Error", e.what());
		qCritical().nospace() << tr("Error: ") + e.what();

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
		qDebug().nospace() << "Filtered: " << locs.size() << " in " << dur.count() << " s";
#endif

#if 0
		// filter by density CPU
		start = std::chrono::steady_clock::now();

		locs.erase(DensityFilter::remove_cpu(locs, 10, 250.f), locs.end());

		end = std::chrono::steady_clock::now();
		dur = std::chrono::duration<double>(end - start);

		qDebug().nospace() << "Density filter (CPU): " << locs.size()  << " in " << dur.count() << " s";
#else
		// filter by density GPU
		start = std::chrono::steady_clock::now();
		locs.erase(DensityFilter::remove_gpu(locs, 10, 250.f), locs.end());

		dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

		qDebug().nospace() << "Density filter (GPU): " << locs.size()  << " in " << dur.count() << " s";
#endif

		// 3D rendering
		start = std::chrono::steady_clock::now();

		locs.copyTo(DeviceType::Device);
		ret = Rendering::render_gpu(locs, voxelSize, 5);

		dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

		qDebug().nospace() << "Rendering (GPU): " << dur.count() << " s";
	} catch(std::exception &e) {
		QMessageBox::critical(this, "Rendering error", e.what());
		qCritical().nospace() << tr("Rendering Error: ") + e.what();
	}

	return ret;
}

void MainWindow::analyse(Volume &volume, Localizations &locs, float sigma)
{
	try {
		Volume filteredVolume(volume.size(), volume.voxelSize(), volume.origin());

		// gaussian filter 3D
		auto start = std::chrono::steady_clock::now();

		const int windowSize = (int)(sigma * 4) | 1;
		// default 7
		GaussianFilter::gaussianFilter_gpu(volume.constData(), filteredVolume.data(), volume.width(), volume.height(), volume.depth(), windowSize, sigma);

		auto dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

		qDebug().nospace() << "Gaussian filter (GPU): " << dur.count() << " s";

		// local thresholding 3D
		start = std::chrono::steady_clock::now();

		LocalThreshold::localThrehsold_gpu(LocalThreshold::IsoData, filteredVolume, filteredVolume, 11);

		dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

		qDebug().nospace() << "Local threshold filter (GPU): " << dur.count() << " s";

		// skeleton 3D
		start = std::chrono::steady_clock::now();

		Skeleton3D::skeletonize(filteredVolume, m_skeleton);

		dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

		qDebug().nospace() << "Skeltonize (CPU): " << dur.count() << " s";

		// analyse skeleton 3D
		start = std::chrono::steady_clock::now();

		Skeleton3D::Analysis analysis;
		auto trees = analysis.calculate(m_skeleton, {}, Skeleton3D::Analysis::NoPruning, true, 0.0, true);

		dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

		qDebug().nospace() << "Analyse skeleton (CPU): " << dur.count() << " s";

		Octree<uint32_t,float,50> tree(locs.bounds());
		for (uint32_t i = 0; i < locs.size(); ++i)
			tree.insert(locs[i].position(), i);

		// filter skeleton 3D
		start = std::chrono::steady_clock::now();
		Volume segmentedVolume(volume.size(), volume.voxelSize(), volume.origin());
		segmentedVolume.fill(0);

		const float r = std::max({volume.voxelSize()[0], volume.voxelSize()[1], volume.voxelSize()[2]});
		std::vector<std::array<float,3>> endPoints;
		std::vector<std::shared_ptr<SkeletonGraph>> graphs;

		Segments segments;
		Segment s;
		for (int i = 0; i < trees.size(); ++i) {
			const auto &t = trees[i];
			auto [segment,voxels,box] = trees.extractVolume(filteredVolume, 1, i);

			if ((box.width() <= 1) || (box.height() <= 1) || (box.depth() <= 1) || (voxels < 50)) {
				continue;
			}

			// draw segment to new volume and count SMLM signals
			uint32_t signalCount = 0;
			for (int z = box.minZ; z <= box.maxZ; ++z) {
				for (int y = box.minY; y <= box.maxY; ++y) {
					for (int x = box.minX; x <= box.maxX; ++x) {
						if (segment(x, y, z)) {
							segmentedVolume(x, y, z) = 255;
							signalCount += static_cast<uint32_t>(tree.countInBox(volume.mapVoxel(x, y, z), volume.voxelSize()));
						}
					}
				}
			}

			// fill segment
			s.numBranches = t.numberOfBranches;
			s.numEndPoints = t.numberOfEndPoints;
			s.numJunctionVoxels = t.numberOfJunctionVoxels;
			s.numJunctions = t.numberOfJunctions;
			s.numSlabs = t.numberOfSlabs;
			s.numTriples = t.numberOfTriplePoints;
			s.numQuadruples = t.numberOfQuadruplePoints;
			s.averageBranchLength = t.averageBranchLength;
			s.maximumBranchLength = t.maximumBranchLength;
			s.shortestPath = t.shortestPath;
			s.voxels = voxels;
			// add 1 since bounding box calculates (max-min)
			s.width = box.width() + 1;
			s.height = box.height() + 1;
			s.depth = box.depth() + 1;
			s.signalCount = signalCount;

			segments.push_back(s);

			for (const auto &p : t.endPoints)
				endPoints.push_back(m_skeleton.mapVoxel(p.x, p.y, p.z, true));
			graphs.push_back(t.graph);
		}
		segments.volume = segmentedVolume;

		qDebug().nospace() << "Filter skeleton (CPU): " << dur.count() << " s";

		m_ui->volumeView->setVolume(segmentedVolume, {0., 0., 1., 0.4});
		for (const auto &g : graphs)
			m_ui->volumeView->addGraph(g, segmentedVolume, 0.5f * r, {0.f, 1.f, 0.f});
		m_ui->volumeView->addSpheres(endPoints, 0.8f * r, {1.f,0.f,0.f});

	} catch(std::exception &e) {
		QMessageBox::critical(this, "Analyse error", e.what());
		qCritical().nospace() << tr("Analyse error: ") + e.what();
	}
}

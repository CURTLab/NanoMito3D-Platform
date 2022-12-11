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

#include "AnalyzeMitochondria.h"

#include <QDebug>
#include <QThreadPool>

#include "DensityFilter.h"
#include "Rendering.h"

#include "GaussianFilter.h"
#include "LocalThreshold.h"
#include "Skeletonize3D.h"
#include "AnalyzeSkeleton.h"
#include "Segments.h"
#include "Octree.h"

AnalyzeMitochondria::AnalyzeMitochondria(QObject *parent)
	: QObject{parent}
{

}

void AnalyzeMitochondria::load(const QString &fileName, bool threaded)
{
	auto func = [this,fileName]() {
		try {
			m_locs.load(fileName.toStdString());
			m_fileName = fileName;
			emit localizationsLoaded();
		} catch(std::exception &e) {
			qCritical().nospace() << tr("AnalyzeMitochondria::load Error: ") + e.what();
			emit error(tr("Load localizations error"), e.what());
		}
	};

	if (threaded)
		QThreadPool::globalInstance()->start(func);
	else
		func();
}

int AnalyzeMitochondria::availableChannels() const
{
	return m_locs.channels();
}

void AnalyzeMitochondria::render(std::array<float, 3> voxelSize, std::array<float, 3> maxPA, int windowSize, int channel, bool densityFilter, int minPts, float radius, bool useGPU, bool threaded)
{
	auto func = [this,voxelSize,maxPA,windowSize,channel,minPts,radius,useGPU,densityFilter]() {
		try {
			// filter localizations by channel and PA
			auto start = std::chrono::steady_clock::now();

			m_locs.erase(std::remove_if(m_locs.begin(), m_locs.end(), [&maxPA,channel](const Localization &l) {
				if (channel > 0 && l.channel != channel)
					return true;
				return (l.PAx > maxPA[0] || l.PAy > maxPA[1] || l.PAz > maxPA[2]);
			}), m_locs.end());

			auto dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
			qDebug().nospace() << "Filtered: " << m_locs.size() << " in " << dur.count() << " s";

			if (densityFilter) {
				// filter by density
				start = std::chrono::steady_clock::now();

				if (useGPU)
					m_locs.erase(DensityFilter::remove_gpu(m_locs, minPts, radius), m_locs.end());
				else
					m_locs.erase(DensityFilter::remove_cpu(m_locs, minPts, radius), m_locs.end());

				dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

				qDebug().nospace() << "Density filter " << (useGPU ? "GPU" : "CPU") << ": " << m_locs.size()  << " in " << dur.count() << " s";
			}

			// 3D rendering
			start = std::chrono::steady_clock::now();

			if (useGPU)
				m_volume = Rendering::render_gpu(m_locs, voxelSize, windowSize);
			else
				m_volume = Rendering::render_cpu(m_locs, voxelSize, windowSize);

			dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

			qDebug().nospace() << "Rendering " << (useGPU ? "GPU" : "CPU") << ": " << dur.count() << " s";

			emit volumeRendered();
		} catch(std::exception &e) {
			qCritical().nospace() << tr("AnalyzeMitochondria::render Error: ") + e.what();
			emit error(tr("Rendering error"), e.what());
		}
	};

	if (threaded)
		QThreadPool::globalInstance()->start(func);
	else
		func();
}

void AnalyzeMitochondria::analyze(float sigma, ThresholdMethods thresholdMethod, bool useGPU, bool threaded)
{
	auto func = [this,sigma,thresholdMethod,useGPU]() {
		try {
			Volume filteredVolume(m_volume.size(), m_volume.voxelSize(), m_volume.origin());

			// gaussian filter 3D
			auto start = std::chrono::steady_clock::now();

			const int windowSize = (int)(sigma * 4) | 1;
			// default 7
			GaussianFilter::gaussianFilter_gpu(m_volume.constData(), filteredVolume.data(), m_volume.width(), m_volume.height(), m_volume.depth(), windowSize, sigma);

			auto dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

			qDebug().nospace() << "Gaussian filter (GPU): " << dur.count() << " s";

			// local thresholding 3D
			start = std::chrono::steady_clock::now();

			if (useGPU) {
				switch(thresholdMethod) {
				case ThresholdMethods::LocalISOData: LocalThreshold::localThrehsold_gpu(LocalThreshold::IsoData, filteredVolume, filteredVolume, 11); break;
				case ThresholdMethods::LocalOtsu: LocalThreshold::localThrehsold_gpu(LocalThreshold::Otsu, filteredVolume, filteredVolume, 11); break;
				}
			} else {
				switch(thresholdMethod) {
				case ThresholdMethods::LocalISOData: LocalThreshold::localThrehsold_cpu(LocalThreshold::IsoData, filteredVolume, filteredVolume, 11); break;
				case ThresholdMethods::LocalOtsu: LocalThreshold::localThrehsold_cpu(LocalThreshold::Otsu, filteredVolume, filteredVolume, 11); break;
				}
			}

			dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

			qDebug().nospace() << "Threshold filter " << (useGPU ? "GPU" : "CPU") << ": " << dur.count() << " s";

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

			/*Octree<uint32_t,float,50> tree(m_locs.bounds());
			for (uint32_t i = 0; i < m_locs.size(); ++i)
				tree.insert(m_locs[i].position(), i);*/

			// filter skeleton 3D
			start = std::chrono::steady_clock::now();
			Volume segmentedVolume(m_volume.size(), m_volume.voxelSize(), m_volume.origin());
			segmentedVolume.fill(0);

			emit progressRangeChanged(0, static_cast<int>(trees.size())-1);

			m_segments.clear();

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
								//signalCount += static_cast<uint32_t>(tree.countInBox(m_volume.mapVoxel(x, y, z), m_volume.voxelSize()));
							}
						}
					}
				}

				Segment s;
				s.graph = t.graph;

				// fill segment
				s.data.numBranches = t.numberOfBranches;
				s.data.numEndPoints = t.numberOfEndPoints;
				s.data.numJunctionVoxels = t.numberOfJunctionVoxels;
				s.data.numJunctions = t.numberOfJunctions;
				s.data.numSlabs = t.numberOfSlabs;
				s.data.numTriples = t.numberOfTriplePoints;
				s.data.numQuadruples = t.numberOfQuadruplePoints;
				s.data.averageBranchLength = t.averageBranchLength;
				s.data.maximumBranchLength = t.maximumBranchLength;
				s.data.shortestPath = t.shortestPath;
				s.data.voxels = voxels;
				// add 1 since bounding box calculates (max-min)
				s.data.width = box.width() + 1;
				s.data.height = box.height() + 1;
				s.data.depth = box.depth() + 1;
				s.data.signalCount = signalCount;

				for (const auto &p : t.endPoints)
					s.endPoints.push_back(m_skeleton.mapVoxel(p.x, p.y, p.z, true));

				m_segments.push_back(s);

				emit progressChanged(i);
			}
			m_segments.volume = segmentedVolume;
			emit progressChanged(static_cast<int>(trees.size())-1);

			qDebug().nospace() << "Filter skeleton (CPU): " << dur.count() << " s";

			emit volumeAnalyzed();
		} catch(std::exception &e) {
			qCritical().nospace() << tr("AnalyzeMitochondria::render Error: ") + e.what();
			emit error(tr("Rendering error"), e.what());
		}
	};

	if (threaded)
		QThreadPool::globalInstance()->start(func);
	else
		func();
}

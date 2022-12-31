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
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

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
			uint32_t locs = 0;
			m_locs.load(fileName.toStdString(), [this,&locs](uint32_t i, uint32_t n,const Localization &l) {
				if (locs != n) {
					locs = n;
					emit progressRangeChanged(0, static_cast<int>(locs)-1);
				}
				if (i % 1000 == 1)
					emit progressChanged(i);
			});
			emit progressChanged(static_cast<int>(locs)-1);
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
			qDebug().nospace() << "Filtered channel/PA (CPU): " << m_locs.size() << " in " << dur.count() << " s";

			if (densityFilter) {
				// filter by density
				start = std::chrono::steady_clock::now();

				if (useGPU)
					m_locs.erase(DensityFilter::remove_gpu(m_locs, minPts, radius), m_locs.end());
				else
					m_locs.erase(DensityFilter::remove_cpu(m_locs, minPts, radius), m_locs.end());

				dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

				qDebug().nospace() << "Density filter (" << (useGPU ? "GPU" : "CPU") << "): " << m_locs.size()  << " in " << dur.count() << " s";
			}

			// 3D rendering
			start = std::chrono::steady_clock::now();

			if (useGPU)
				m_volume = Rendering::render_gpu(m_locs, voxelSize, windowSize);
			else
				m_volume = Rendering::render_cpu(m_locs, voxelSize, windowSize);

			dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

			qDebug().nospace() << "Rendering (" << (useGPU ? "GPU" : "CPU") << "): " << dur.count() << " s";

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
			if (useGPU)
				GaussianFilter::gaussianFilter_gpu(m_volume.constData(), filteredVolume.data(), m_volume.width(), m_volume.height(), m_volume.depth(), windowSize, sigma);
			else
				GaussianFilter::gaussianFilter_cpu(m_volume.constData(), filteredVolume.data(), m_volume.width(), m_volume.height(), m_volume.depth(), windowSize, sigma);

			auto dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

			qDebug().nospace() << "Gaussian filter (" << (useGPU ? "GPU" : "CPU") << "): " << dur.count() << " s";

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

			qDebug().nospace() << "Threshold filter (" << (useGPU ? "GPU" : "CPU") << "): " << dur.count() << " s";

			// skeleton 3D
			start = std::chrono::steady_clock::now();

			Skeleton3D::skeletonize(filteredVolume, m_skeleton);

			dur = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);

			qDebug().nospace() << "Skeltonize (CPU): " << dur.count() << " s";

			// analyse skeleton 3D
			start = std::chrono::steady_clock::now();

			Skeleton3D::Analysis analysis;
			auto trees = analysis.calculate(m_skeleton, {}, Skeleton3D::Analysis::NoPruning, true, 0.0, true);

			m_labeledVolume.alloc(m_volume.width(), m_volume.height(), m_volume.depth());
			std::fill_n(m_labeledVolume.data, m_volume.voxels(), 0);

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

			int id = 1;
			for (int i = 0; i < trees.size(); ++i) {
				const auto &t = trees[i];
				auto [segment,voxels,box] = trees.extractVolume(filteredVolume, 1, i);

				if ((box.width() <= 1) || (box.height() <= 1) || (box.depth() <= 1) || (voxels < 50)) {
					continue;
				}

				// draw segment to new volume and count SMLM signals
				uint32_t signalCount = 0;
				box.forEachVoxel([&](int x, int y, int z) {
					if (segment(x, y, z)) {
						segmentedVolume(x, y, z) = 255;
						m_labeledVolume(x, y, z) = id;
						//signalCount += static_cast<uint32_t>(tree.countInBox(m_volume.mapVoxel(x, y, z), m_volume.voxelSize()));
					}
				});

				Segment s;
				s.id = id++;
				s.boundingBox = box;
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

void AnalyzeMitochondria::classify(bool threaded)
{
	auto func = [this]() {
		try {
			if (m_dtree.empty()) {
				emit error(tr("Classification error"), tr("No model loaded!"));
				return;
			}

			cv::Mat result;

			m_classifiedVolume = Volume(m_volume.size(), m_volume.voxelSize(), m_volume.origin());
			m_classifiedVolume.fill(0);

			std::vector<uint32_t> hist(m_numClasses, 0);
			uint32_t numVoxels = 0;

			emit progressRangeChanged(0, static_cast<int>(m_segments.size())-1);
			for (size_t i = 0; i < m_segments.size(); ++i) {
				cv::Mat dataSet(1, 14, CV_32FC1, &m_segments[i].data);
				m_dtree->predict(dataSet, result, 0);

				const int prediction = qRound(result.at<float>(0, 0));

				m_segments[i].prediction = prediction;

				m_segments[i].boundingBox.forEachVoxel([&](int x, int y, int z) {
					if (m_labeledVolume(x, y, z) == m_segments[i].id) {
						m_classifiedVolume(x, y, z) = prediction;
						hist[prediction]++;
						++numVoxels;
					}
				});

				emit progressChanged(static_cast<int>(i));
			}

			m_classificationResult.resize(m_numClasses-1);
			for (int i = 1; i < m_numClasses; ++i)
				m_classificationResult[i-1] = (double)hist[i] / numVoxels;

			emit volumeClassified();
		} catch(std::exception &e) {
			qCritical().nospace() << tr("AnalyzeMitochondria::classify Error: ") + e.what();
			emit error(tr("Classification error"), e.what());
		}
	};

	if (threaded)
		QThreadPool::globalInstance()->start(func);
	else
		func();
}

void AnalyzeMitochondria::loadModel(const QString &fileName)
{
	const auto fi = QFileInfo(fileName);
	if (fi.completeSuffix().toLower() == "csv") {
		// load header names from csv
		QStringList headerNames;

		QFile file(fileName);
		if (file.open(QIODevice::ReadOnly)) {
			headerNames = QString(file.readLine()).trimmed().split(", ");
			file.close();
		} else {
			emit error(tr("Load model"), tr("Could not load dataset! (%1)").arg(fileName));
			return;
		}

		// generate training data from csv
		cv::Ptr<cv::ml::TrainData> tDataContainer = cv::ml::TrainData::loadFromCSV(fileName.toStdString(), 1);
		cv::Mat trainData = tDataContainer->getTrainSamples();
		cv::Mat targetData = tDataContainer->getTrainResponses();

		// find number of classes
		double min, max;
		cv::minMaxIdx(targetData, &min, &max);

		m_numClasses = qRound(max+1);

		// finally train random forest from csv data
		m_dtree = cv::ml::RTrees::create();
		m_dtree->setCalculateVarImportance(true);
		m_dtree->setMaxCategories(m_numClasses+1);
		if (!m_dtree->train(trainData, cv::ml::ROW_SAMPLE, targetData)) {
			emit error(tr("Load model"), tr("Could not train loaded dataset! (%1)").arg(fileName));
			return;
		}

		// print variable importance
		qDebug() << "VarImportance:";
		cv::Mat importance = m_dtree->getVarImportance();
		for (int i = 0; i < importance.rows; ++i)
			qDebug().nospace() << i << ": " << headerNames.at(i) << ": " << importance.at<float>(i, 0);
	} else if (fi.completeSuffix().toLower() == "json") {
		// read number of classes directly from json file
		QFile file(fileName);
		if (file.open(QIODevice::ReadOnly)) {
			QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
			file.close();

			const auto root = doc.object()["opencv_ml_rtrees"].toObject();
			const auto class_labels = root["class_labels"].toArray();
			m_numClasses = class_labels.size();
		} else {
			emit error(tr("Load model"), tr("Could not load dataset! (%1)").arg(fileName));
			return;
		}
		// load model
		m_dtree = cv::ml::StatModel::load<cv::ml::RTrees>(fileName.toStdString());
	}

	if (m_dtree.empty()) {
		emit error(tr("Load model"), tr("Could not load model %1").arg(fileName));
		return;
	}

	qDebug() << "Samples:" << m_dtree->getVarCount();
	qDebug() << "Num classes:" << m_numClasses;
}

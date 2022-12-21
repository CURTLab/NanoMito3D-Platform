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

#include "Correction.h"

#include <QThreadPool>
#include <QDebug>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#include <opencv2/imgcodecs.hpp>

Correction::Correction(QObject *parent)
	: QObject(parent)
{
}

void Correction::load(const QString &fileName, bool threaded)
{
	auto func = [this,fileName]() {
		try {
			uint32_t nLocs = 0;
			m_locs.load(fileName.toStdString(), [this,&nLocs](uint32_t i, uint32_t n, const Localization &) {
				if (nLocs != n) {
					nLocs = n;
					emit progressRangeChanged(0, static_cast<int>(n)-1);
				}
				if (i % 100 == 1)
					emit progressChanged(i);
			});
			emit progressChanged(static_cast<int>(nLocs)-1);
			m_fileName = fileName;
			emit localizationsLoaded();
		} catch(std::exception &e) {
			qCritical().nospace() << tr("Correction::load Error: ") + e.what();
			emit error(tr("Load localizations error"), e.what());
		}
	};

	if (threaded)
		QThreadPool::globalInstance()->start(func);
	else
		func();
}

void Correction::loadRawImageFromSingleStack(const QString &registrationPath, const QString &stackPath, bool threaded)
{
	auto func = [this,registrationPath,stackPath]() {
		QFile file(registrationPath);
		if (!file.open(QIODevice::ReadOnly)) {
			emit error(tr("Load registration error"), tr("Could not open registration file!"));
			return;
		}
		QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
		file.close();

		std::array<cv::Rect,CHANNELS> rois;

		for (const auto &roi : doc.object()["roi"].toArray()) {
			QJsonObject values = roi.toObject();
			if (!values.isEmpty()
				 && values.contains("offset")
				 && values.contains("size")
				 && values.contains("channel"))
			{
				const int channel = values["channel"].toInt();
				if (channel >= CHANNELS) {
					qWarning() << "Load registration: Only two color channels are supported!";
					continue;
				}

				QJsonObject offset = values.value("offset").toObject();
				QJsonObject size = values.value("size").toObject();
				if (!offset.isEmpty()
					 && offset.contains("x")
					 && offset.contains("y")
					 && !size.isEmpty()
					 && size.contains("width")
					 && size.contains("height"))
				{
					rois[channel].x = offset.value("x").toInt();
					rois[channel].y = offset.value("y").toInt();
					rois[channel].width = size.value("width").toInt();
					rois[channel].height = size.value("height").toInt();
				}
			}
		}

		// check ROIs
		for (int i = 0; i < CHANNELS; ++i) {
			if ((rois[i].width <= 0) || (rois[i].height <= 0)) {
				emit error(tr("Load registration error"), tr("ROI of channel %1 could not been loaded!").arg(i+1));
				return;
			}
			if ((i > 0) && ((rois[0].width != rois[i].width) || (rois[0].height != rois[i].height))) {
				emit error(tr("Load registration error"), tr("Loaded ROIs are not the same size!"));
				return;
			}
		}


		try {
			// load raw image stack
			const auto numFrames = cv::imcount(stackPath.toStdString(), cv::IMREAD_UNCHANGED);

			emit progressRangeChanged(0, static_cast<int>(numFrames));
			emit progressChanged(0);

			std::vector<cv::Mat> stack;
			cv::imreadmulti(stackPath.toStdString(), stack, 0, static_cast<int>(numFrames), cv::IMREAD_UNCHANGED);


			for (int i = 0; i < CHANNELS; ++i)
				m_rawImageStack[i].resize(numFrames);

			// copy loaded frames into stacks as float32 cv::Mat
			cv::Mat frame;
			for (size_t j = 0; j < numFrames; ++j) {
				//cv::transpose(stack[j], frame);
				frame = stack[j];
				for (int i = 0; i < CHANNELS; ++i) {
					auto tmp = frame(rois[i]);
					tmp.convertTo(m_rawImageStack[i][j], CV_32F);
				}
				progressChanged(static_cast<int>(j) + 1);
			}
		} catch(std::exception &e) {
			qCritical().nospace() << tr("Correction load stacks Error: ") + e.what();
			emit error(tr("Load stacks error"), e.what());
		}

		emit imageStacksLoaded();
	};

	if (threaded)
		QThreadPool::globalInstance()->start(func);
	else
		func();
}

void Correction::loadTwoRawImageStacks(const QString &channel1Path, const QString &channel2Path, bool threaded)
{
	auto func = [this,channel1Path,channel2Path]() {
		if (channel1Path.isEmpty()) {
			emit error(tr("Load two stacks error"), tr("Could not load channel 1 image!"));
			return;
		}

		if (channel2Path.isEmpty()) {
			emit error(tr("Load two stacks error"), tr("Could not load channel 2 image!"));
			return;
		}

		auto loadChannel = [this](int channel, const QString &fileName, int numFrames) {
			if (channel >= CHANNELS) {
				qWarning() << "Load stacks: Only two color channels are supported!";
				return;
			}

			std::vector<cv::Mat> stack;
			cv::imreadmulti(fileName.toStdString(), stack, 0, static_cast<int>(numFrames), cv::IMREAD_UNCHANGED);

			m_rawImageStack[channel].resize(numFrames);
			for (size_t i = 0; i < numFrames; ++i) {
				stack[i].convertTo(m_rawImageStack[channel][i], CV_32F);
				progressChanged(static_cast<int>(i) + channel * numFrames);
			}
		};

		try {
			// load raw image stack
			const auto numFrames1 = cv::imcount(channel1Path.toStdString(), cv::IMREAD_UNCHANGED);
			const auto numFrames2 = cv::imcount(channel2Path.toStdString(), cv::IMREAD_UNCHANGED);
			const int numFrames = std::min(numFrames1, numFrames2);

			if (numFrames <= 0) {
				emit error(tr("Load two stacks error"), tr("Images are empty!"));
				return;
			}

			emit progressRangeChanged(0, 2 * numFrames - 1);
			loadChannel(0, channel1Path, numFrames);
			loadChannel(1, channel2Path, numFrames);

			emit imageStacksLoaded();
		} catch(std::exception &e) {
			qCritical().nospace() << tr("Load 2ch stacks Error: ") + e.what();
			emit error(tr("Load stacks error"), e.what());
		}

	};

	if (threaded)
		QThreadPool::globalInstance()->start(func);
	else
		func();
}

void Correction::correct(QImage labeling, float renderSize, QVector<QColor> labelColors, int channel, bool threaded)
{
	auto func = [this,labeling,renderSize,labelColors,channel]() {
		m_corrected.clear();
		m_corrected.copyMetaDataFrom(m_locs);

		std::vector<Features> features;
		std::vector<int> labels;
		Features f;

		// check if featureSize is the same size as struct Features
		const int featureSize = 6;
		static_assert(sizeof(Features) == featureSize * sizeof(float));
		static_assert(sizeof(featureNames) == featureSize * sizeof(char*));

		int hist[2] = {0};
		for (const auto &l : m_locs) {
			if (l.channel != channel)
				continue;

			// extract color from labeling image
			QPoint pos(l.x/renderSize, l.y/renderSize);
			QColor c = labeling.pixelColor(pos);

			// seach for label color
			int target = -1;
			for (int i = 0; i < labelColors.size(); ++i) {
				if (c == labelColors[i]) {
					target = i;
					break;
				}
			}

			// skip unlabeled localizations
			if (target == -1)
				continue;

			hist[target]++;

			extractFeatures(l, f);

			features.push_back(f);
			labels.push_back(target);
		}

		cv::Ptr<cv::ml::RTrees> dtree;
		try {
			cv::Mat trainData(static_cast<int>(features.size()), featureSize, CV_32F, features.data());
			cv::Mat trainLabel(labels, false);

			dtree = cv::ml::RTrees::create();
			//dtree->setMaxDepth(18);
			dtree->setCalculateVarImportance(true);
			if (!dtree->train(trainData, cv::ml::ROW_SAMPLE, trainLabel)) {
				emit error(tr("Correction error"), "Training failed!");
				return;
			}

			cv::Mat importance = dtree->getVarImportance();
			for (int i = 0; i < importance.rows; ++i)
				qDebug().nospace() << i << ": " << featureNames[i] << ": " << importance.at<float>(i, 0);

			// test accurary
			cv::Mat test;
			dtree->predict(trainData, test);

			double hist1[2] = {0},  hist2[2] = {0};
			double sum = 0;
			for (int i = 0; i < trainLabel.rows; ++i) {
				const int pred = qRound(test.at<float>(i, 0));
				sum += (pred == labels[i]);
				hist1[labels[i]] += 1.0;
				hist2[pred] += 1.0;
			}
			sum /= trainLabel.rows;
			qDebug().nospace() << "Accuracy: " << sum << "%" << ", value accurary: " << (hist2[1] / hist1[1]) << "%";

			// filter image
			cv::Mat result;
			for (const auto &l : m_locs) {
				if (l.channel == channel) {
					extractFeatures(l, f);
					dtree->predict(cv::Mat(1, featureSize, CV_32F, &f), result, 0);

					if (qRound(result.at<float>(0, 0)) != 1)
						continue;
				}
				m_corrected.push_back(l);
			}
			qDebug() << m_locs.size() << m_corrected.size();
		} catch(std::exception &e) {
			qCritical().nospace() << tr("Correction::load Error: ") + e.what();
			emit error(tr("Correction error"), e.what());
			return;
		}

		emit correctionFinished();
	};

	if (threaded)
		QThreadPool::globalInstance()->start(func);
	else
		func();
}

int Correction::availableChannels() const
{
	return m_locs.channels();
}

void Correction::extractFeatures(const Localization &l, Features &f) const
{
	f.frame = l.frame;
	f.PAx = l.PAx;
	f.PAy = l.PAy;
	f.PAz = l.PAz;
	f.intensity = l.intensity;
	f.background = l.background;
}

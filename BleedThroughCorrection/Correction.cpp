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

Correction::Correction(QObject *parent)
	: QObject{parent}
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

void Correction::correct(QImage labeling, float renderSize, QVector<QColor> labelColors, int channel, bool threaded)
{
	auto func = [this,labeling,renderSize,labelColors,channel]() {
		m_corrected.clear();
		m_corrected.copyMetaDataFrom(m_locs);

		std::vector<Features> features;
		std::vector<int> trainLabel;
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
			trainLabel.push_back(target);
		}

		cv::Ptr<cv::ml::RTrees> dtree;
		try {
			cv::Mat trainData(static_cast<int>(features.size()), featureSize, CV_32F, features.data());

			dtree = cv::ml::RTrees::create();
			//dtree->setMaxDepth(18);
			dtree->setCalculateVarImportance(true);
			if (!dtree->train(trainData, cv::ml::ROW_SAMPLE, cv::Mat(trainLabel, false))) {
				emit error(tr("Correction error"), "Training failed!");
				return;
			}

			cv::Mat importance = dtree->getVarImportance();
			for (int i = 0; i < importance.rows; ++i)
				qDebug().nospace() << i << ": " << featureNames[i] << ": " << importance.at<float>(i, 0);

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

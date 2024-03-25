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

#ifndef CELLCOUNTER_H
#define CELLCOUNTER_H

#include <QObject>
#include <QString>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class CellCounter : public QObject
{
	Q_OBJECT
public:
	explicit CellCounter(QObject *parent = nullptr);

	bool loadModel(QString model);

	void predictAsync(const cv::Mat &input, int subDivisions, int batchSize, bool rotation = false);

	inline constexpr const cv::Mat &result() const { return m_result; }
	inline constexpr const QString &lastError() const { return m_lastError; }

signals:
	void progressValueChanged(int progress);
	void progressRangeChanged(int min, int max);
	void finished();

private:
	bool predict(const cv::Mat &input, cv::Mat &output);
	cv::Mat splineWindow(int windowSize) const;

	int m_windowSize = 0;
	cv::dnn::Net m_model;
	cv::Mat m_result;
	QVector<cv::Rect> m_regions;
	int m_batchSize;
	int m_subDivisions;
	cv::Rect m_paddedRoi;
	bool m_rotation;
	cv::Mat m_padded;
	QString m_lastError;

};

#endif // CELLCOUNTER_H

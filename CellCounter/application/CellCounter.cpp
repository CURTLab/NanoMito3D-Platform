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

#include "CellCounter.h"

#include <QThreadPool>
#include <QDebug>
#include <stdexcept>

CellCounter::CellCounter(QObject *parent)
	: QObject{parent}
{
}

bool CellCounter::loadModel(QString model)
{
	m_model = cv::dnn::readNet(model.toStdString());
	if (m_model.empty())
		return false;
	m_windowSize = 128;

	return true;
}

void CellCounter::predictAsync(const cv::Mat &input, int subDivisions, int batchSize, bool rotation)
{
	m_rotation = rotation;
	m_subDivisions = subDivisions;
	m_result = cv::Mat{};

	const int subBindowSize = m_windowSize / subDivisions;

	const int nx = (input.size[0] + 2*subBindowSize) / subBindowSize;
	const int ny = (input.size[1] + 2*subBindowSize) / subBindowSize;

	const int bx1 = (nx * subBindowSize - input.size[0]) / 2;
	const int bx2 = nx * subBindowSize - input.size[0] - bx1;

	const int by1 = (ny * subBindowSize - input.size[1]) / 2;
	const int by2 = ny * subBindowSize - input.size[1] - by1;

	m_paddedRoi = cv::Rect(by1, bx1, input.size[1], input.size[0]);
	cv::copyMakeBorder(input, m_padded, bx1, bx2, by1, by2, cv::BORDER_REFLECT101);

	//qDebug() << m_padded.size[0] << m_padded.size[1];

	const int rangeX = (m_padded.size[0] - m_windowSize)/subBindowSize + 1;
	const int rangeY = (m_padded.size[1] - m_windowSize)/subBindowSize + 1;
	m_regions.resize(rangeX * rangeY);
	std::vector<cv::Mat> stack(rangeX * rangeY);
	for (int i = 0; i < m_regions.size(); ++i) {
		const int dy = i / rangeX;
		const int dx = i - dy * rangeX;
		m_regions[i] = cv::Rect(dy * subBindowSize, dx * subBindowSize, m_windowSize, m_windowSize);
#if !defined(QT_NO_DEBUG) || defined(QT_FORCE_ASSERTS)
		const auto &roi = m_regions[i];
		Q_ASSERT(0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m_padded.cols);
		Q_ASSERT(0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m_padded.rows);
#endif
		stack[i] = m_padded(m_regions[i]);
	}

	const int n = (rotation ? m_regions.size() * 4 : m_regions.size());
	m_batchSize = std::min(n, batchSize);

	qDebug() << "Start async with" << n << "regions";
	QThreadPool::globalInstance()->start([this]() {
		const int n = m_rotation ? 4 * m_regions.size() : m_regions.size();

		emit progressRangeChanged(0, n / m_batchSize + 1);
		emit progressValueChanged(0);

		const int advance = m_rotation ? m_batchSize / 4 : m_batchSize;
		auto begin = m_regions.begin();
		auto end = m_regions.begin() + advance;

		cv::Mat tiles({m_batchSize, m_windowSize, m_windowSize}, CV_32F);
		cv::Mat output(m_padded.size[0], m_padded.size[1], CV_32F, 0.f);

		for (int i = 0; begin < m_regions.end(); ++i) {
			if (end > m_regions.end())
				end = m_regions.end();

			float *ptr = tiles.ptr<float>();
			for (auto it = begin; it != end; ++it) {
				cv::Mat tmp, dest;

				dest = cv::Mat(m_windowSize, m_windowSize, CV_32F, ptr);
				m_padded(*it).convertTo(dest, CV_32F, 1.f/255);
				ptr += dest.total();

				for (int j = 0; m_rotation && j < 2; ++j) {
					dest = cv::Mat(m_windowSize, m_windowSize, CV_32F, ptr);
					ptr += dest.total();
					Q_ASSERT((int)std::distance(tiles.ptr<float>(), ptr) < tiles.total());

					m_padded(*it).convertTo(dest, CV_32F, 1.f/255);

					cv::rotate(dest, dest, j); // cv::ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE
				}
			}

			cv::Mat data;
			if (!predict(tiles, data) || (data.dims != 3) || (data.size[1] != m_windowSize) || (data.size[2] != m_windowSize)) {
				throw std::runtime_error("Error during prediction!");
			}

			cv::Mat wind = splineWindow(m_windowSize);
			float *raw_data = data.ptr<float>();
			for (auto it = begin; it != end; ++it) {
				cv::Mat src(m_windowSize, m_windowSize, CV_32F, 0.f);
				cv::Mat rot, tmp;

				tmp = cv::Mat(m_windowSize, m_windowSize, CV_32F, (void*)raw_data);
				raw_data += tmp.total();
				src += tmp;

				for (int j = 2; m_rotation && j != 0; --j) {
					tmp = cv::Mat(m_windowSize, m_windowSize, CV_32F, (void*)raw_data);
					raw_data += tmp.total();
					cv::rotate(tmp, rot, j);
					src += rot;
				}
				if (m_rotation)
					src /= 4.f;

				output(*it) += m_subDivisions > 1 ? src.mul(wind) : src;
			}

			begin += advance;
			end += advance;

			emit progressValueChanged(i + 1);
		}

		m_padded = cv::Mat();
		m_result = output(m_paddedRoi) / std::pow(m_subDivisions, 2.0);

		emit progressValueChanged(n / m_batchSize);

		emit finished();
	});
}

bool CellCounter::predict(const cv::Mat &input, cv::Mat &output)
{
	const int n = input.size[0];

	cv::Mat inputBlob = cv::dnn::blobFromImages(input);
	if (inputBlob.size[0] != n)
		return false;

	m_model.setInput(inputBlob);

	output = cv::Mat({n, 1, m_windowSize, m_windowSize}, CV_32F);
	m_model.forward(output);

	output = output.reshape(0, {n, m_windowSize, m_windowSize});

	return true;
}

cv::Mat CellCounter::splineWindow(int windowSize) const
{
	// translated to c++ from https://github.com/Vooban/Smoothly-Blend-Image-Patche

	cv::Mat wind(windowSize, 1, CV_32F, 0.f);

	const int intersection = windowSize/4;
	for (int i = 0; i < intersection; ++i) {
		const double w1 = (2 * i - 1.0) / windowSize;
		const double w2 = 2.0 - 2.0 * (windowSize - i - 1) / windowSize;
		wind.at<float>(i) = std::pow(std::abs(2.0 * w1), 2.0) * 0.5;
		wind.at<float>(windowSize - i - 1) = std::pow(std::abs(2.0 * w2), 2.0) * 0.5;
	}
	for (int i = intersection-1; i < windowSize-intersection; ++i) {
		const double w = i < windowSize/2 ? (2 * i - 1.0) / windowSize : 2.0 - 2.0 * i / windowSize;
		wind.at<float>(i) = 1.0 - std::pow(std::abs(2.0*(w - 1)), 2.0) * 0.5;
	}

	std::vector<float> array(windowSize);
	for (int i = 0; i < windowSize; ++i)
		array[i] = wind.at<float>(i);

	const auto mean = cv::mean(wind);
	wind /= mean;

	return wind * wind.t();
}

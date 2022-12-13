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

#include "LocalMaximumSearch.h"

LocalMaximumSearch::LocalMaximumSearch(int border, int windowSize)
	: m_border(border)
	, m_windowSize(windowSize)
{
}

std::vector<LocalMaximum> LocalMaximumSearch::findAll(const cv::Mat &image) const
{
	if (image.type() != CV_32F) {
		throw std::runtime_error("Image type is not CV_32F!");
	}

	std::vector<LocalMaximum> features;
	const int r = m_windowSize / 2;
	const int b = m_border;

	const int w = image.rows - (b + 1);
	const int h = image.cols - (b + 1);

	float value = 0.f;
	// A. Neubeck et.al., 'Efficient Non-MaximumSuppression', 2006, (2n+1)×(2n+1)-Block Algorithm
	for (int i = b; i < w; i += (r + 1)) {
		for (int j = b; j < h; j += (r + 1)) {
			int mi = i;
			int mj = j;
			value = image.at<float>(mi, mj);
			for (int i2 = i; i2 <= i + r; ++i2) {
				for (int j2 = j; j2 <= j + r; ++j2) {
					if (i2 < 0 || j2 < 0 || i2 >= image.rows || j2 >= image.cols)
						continue;
					if (image.at<float>(i2, j2) > value) {
						value = image.at<float>(i2, j2);
						mi = i2;
						mj = j2;
					}
				}
			}

			value = image.at<float>(mi, mj);
			for (int i2 = mi - r; i2 <= mi + r; ++i2) {
				for (int j2 = mj - r; j2 <= mj + r; ++j2) {
					if (i2 < 0 || j2 < 0 || i2 >= image.rows || j2 >= image.cols)
						continue;
					if (image.at<float>(i2, j2) > value)
						goto failed;
				}
			}
			features.push_back(LocalMaximum{value, mi, mj});
failed:;
		}
	}
	return features;
}

int LocalMaximumSearch::border() const
{
	return m_border;
}

void LocalMaximumSearch::setBorder(int border)
{
	m_border = border;
}

int LocalMaximumSearch::windowSize() const
{
	return m_windowSize;
}

void LocalMaximumSearch::setWindowSize(int windowSize)
{
	m_windowSize = windowSize;
}

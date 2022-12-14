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
#include "Rendering.h"

void Rendering::histogram2D(const Localizations &locs, cv::Mat &output, float renderSize, std::function<bool (int, const Localization &)> filter)
{
	const int width = static_cast<int>(std::ceil(locs.width() / renderSize));
	const int height = static_cast<int>(std::ceil(locs.height() / renderSize));

	output = cv::Mat(width, height, CV_32F, 0.f);

	int idx = 0;
	for (const auto &l : locs) {
		if (filter(idx++, l))
			continue;

		const float x0 = (l.x/renderSize);
		const float y0 = (l.y/renderSize);

		const float sigma = 0.75f;
		const int half_local_size = 2;

		int top = std::max(0, (int)(y0) - half_local_size);
		int bottom = std::min(height - 1, (int)(y0) + half_local_size);
		int left = std::max(0, (int)(x0) - half_local_size);
		int right = std::min(width - 1, (int)(x0) + half_local_size);

		for(int y = top; y <= bottom; ++y) {
			for(int x = left; x <= right; x++) {
				const float dx = (x - x0);
				const float dy = (y - y0);
				output.at<float>(x, y) = std::max(output.at<float>(x, y), 10.0f * std::expf(-0.5f * (dx * dx + dy * dy) / (sigma*sigma)));
			}
		}
	}
}

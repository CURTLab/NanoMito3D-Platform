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

#include "LocalThreshold.h"

#include <algorithm>
#include <execution>
#include <QDebug>

int calcHist(const Volume &input, uint16_t hist[256], int x, int y, int z, int r)
{
	std::fill_n(hist, 256, static_cast<uint16_t>(0));

	const int x0 = std::max(x - r, 0);
	const int x1 = std::min(x + r, input.width() - 1);

	const int y0 = std::max(y - r, 0);
	const int y1 = std::min(y + r, input.height() - 1);

	const int z0 = std::max(z - r, 0);
	const int z1 = std::min(z + r, input.depth() - 1);

	int numVoxels = 0;
	for (int z = z0; z <= z1; ++z) {
		for (int y = y0; y <= y1; ++y) {
			for (int x = x0; x <= x1; ++x) {
				++hist[input(x, y, z)];
				++numVoxels;
			}
		}
	}
	return numVoxels;
}

void LocalThreshold::localThrehsold_cpu(Method method, const Volume &input, Volume &output, int windowSize, std::function<void (uint32_t, uint32_t)> cb)
{
	Volume result(input.size(), input.voxelSize(), input.origin());
	result.fill(0);

	const auto numThreads = std::thread::hardware_concurrency();
	const size_t pack = (input.voxels() + numThreads - 1) / numThreads;

	std::atomic_int counter(0);

	// multi-threaded implementation
	std::vector<std::thread> threads;
	for (size_t offset = 0; offset < input.voxels(); offset += pack) {
		threads.emplace_back(std::thread([offset,&pack,&method,&input,&result,windowSize,&cb,&counter]() {
			const uint32_t n = static_cast<uint32_t>(input.voxels());

			uint16_t hist[256];

			const size_t begin = offset;
			const size_t end = std::min(offset + pack, input.voxels());

			uint8_t *dst = result.data() + begin;
			const uint8_t *src = input.constData() + begin;
			for (size_t i = begin; i < end; ++i, ++dst, ++src) {
				const auto idx = input.mapIndex(i);
				// calc local histogram
				const int numVoxels = calcHist(input, hist, idx[0], idx[1], idx[2], windowSize/2);

				// skip if all zero
				if (hist[0] == numVoxels) {
					*dst = 0;
					continue;
				}

				// threshold
				uint8_t threshold = 0;
				if (method == LocalThreshold::Otsu)
					threshold = otsuThreshold(hist, numVoxels);
				else if (method == LocalThreshold::IsoData)
					threshold = isoDataThreshold(hist, numVoxels);
				*dst = *src >= threshold ? 255 : 0;

				// optional progress indicator
				if (cb)
					cb(static_cast<uint32_t>(counter.fetch_add(1) + 1), n);
			}

		}));
	}

	for (size_t j = 0; j < threads.size(); ++j)
		threads[j].join();
	output = result;
}

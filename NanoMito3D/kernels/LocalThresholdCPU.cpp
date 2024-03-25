/****************************************************************************
 *
 * Copyright (C) 2022-2024 Fabian Hauser
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
#include <cmath>

#ifndef CUDA_SUPPORT
#define max fmax
#include "LocalThreshold_impl.h"
#undef max
#endif // CUDA_SUPPORT

void LocalThreshold::localThrehsold_cpu(Method method, const Volume &input, Volume &output, int windowSize, std::function<void (uint32_t, uint32_t)> cb)
{
	Volume result(input.size(), input.voxelSize(), input.origin());
	result.fill(0);

	const int nFilter = windowSize * windowSize * windowSize;
	const int nVoxels = static_cast<int>(input.voxels());

	const auto numThreads = std::thread::hardware_concurrency();
	const int pack = (nVoxels + numThreads - 1) / numThreads;

	std::atomic_int counter(0);

	// calculate linear offsets within the window
	std::shared_ptr<int[]> filterOffsets(new int[nFilter]);
	int *idx = filterOffsets.get();
	const int r = windowSize/2;
	for (int k = -r; k <= r; ++k) {
		for (int j = -r; j <= r; ++j) {
			for (int i = -r; i <= r; ++i)
				*idx++ = i + j * input.width() + k * input.width() * input.height();
		}
	}

	// multi-threaded implementation
	std::vector<std::thread> threads;
	for (int offset = 0; offset < nVoxels; offset += pack) {
		threads.emplace_back(std::thread([offset,&pack,method,&input,&result,&cb,&counter,nFilter,nVoxels,filterOffsets]() {
			const uint32_t n = static_cast<uint32_t>(nVoxels);

			uint16_t hist[256];

			const int begin = offset;
			const int end = std::min(offset + pack, nVoxels);

			uint8_t *dst = result.data() + begin;
			const uint8_t *src = input.constData() + begin;
			for (int i = begin; i < end; ++i, ++dst, ++src) {
				std::fill_n(hist, 256, static_cast<uint16_t>(0));

				for (int j = 0; j < nFilter; ++j) {
					// index for 3D window
					const int idx2 = i + filterOffsets[j];
					const int histIdx = (idx2 >= 0 && idx2 < nVoxels) ? input.constData()[idx2] : 0;
					hist[histIdx]++;
				}

				if (hist[0] == nFilter)
					*dst = 0;
				else if (method == LocalThreshold::Otsu)
					*dst = (*src >= LocalThreshold::otsuThreshold(hist, nFilter) ? 255 : 0);
				else if (method == LocalThreshold::IsoData)
					*dst = (*src >= LocalThreshold::isoDataThreshold(hist, nFilter) ? 255 : 0);
				else
					*dst = 0;

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

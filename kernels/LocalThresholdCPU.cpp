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

void calcHist(const Volume &input, uint16_t hist[256], int &numVoxels, int x, int y, int z, int r)
{
	std::fill_n(hist, 256, (uint16_t)0);

	numVoxels = 0;
	for (int k = -r; k <= r; ++k) {
		for (int j = -r; j <= r; ++j) {
			for (int i = -r; i <= r; ++i) {
				if (input.contains(x + i, y + j, z + k)) {
					hist[input(x + i, y + j, z + k)]++;
					++numVoxels;
				}
			}
		}
	}
}

void LocalThreshold::localThrehsold_cpu(Method method, Volume input, Volume output, int windowSize)
{
	output = Volume(input.size(), input.voxelSize(), input.origin());

	uint8_t threshold = 0;
	uint16_t hist[256];

	int numVoxels = 0;

	uint8_t *val = output.data();
	for (int z = 0; z < input.depth(); ++z) {
		for (int y = 0; y < input.height(); ++y) {
			for (int x = 0; x < input.width(); ++x, ++val) {
				// calc local histogram
				calcHist(input, hist, numVoxels, x, y, z, windowSize);

				if (hist[0] == numVoxels) {
					*val = 0;
					continue;
				}

				if (method == LocalThreshold::Otsu)
					threshold = otsuThreshold(hist, numVoxels);
				else if (method == LocalThreshold::IsoData)
					threshold = isoDataThreshold(hist, numVoxels);
				*val = *val >= threshold ? 255 : 0;
			}
		}
	}
}

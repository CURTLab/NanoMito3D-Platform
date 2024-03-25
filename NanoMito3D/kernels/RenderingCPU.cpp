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

#include "Rendering.h"

#include <cmath>
#include <algorithm>

namespace Rendering
{

void drawPSF_cpu(uint8_t *volume, const Localization &l, const std::array<int,3> &volumeDims, const std::array<float,3> &voxelSize, const std::array<float,3> &origin, int windowSize)
{
	const float posx = l.x - origin[0];
	const float posy = l.y - origin[1];
	const float posz = l.z - origin[2];

	const int ix = static_cast<int>(std::round((posx / voxelSize[0])));
	const int iy = static_cast<int>(std::round((posy / voxelSize[1])));
	const int iz = static_cast<int>(std::round((posz / voxelSize[2])));

	const size_t strideY = volumeDims[0];
	const size_t strideZ = volumeDims[0] * volumeDims[1];

	const int w = windowSize/2;
	for (int z = -w; z <= w; ++z) {
		for (int y = -w; y <= w; ++y) {
			for (int x = -w; x <= w; ++x) {
				if ((ix + x < 0) || (iy + y < 0) || (iz + z < 0) ||
					 (ix + x >= volumeDims[0]) || (iy + y >= volumeDims[1]) || (iz + z >= volumeDims[2]))
					continue;

				const size_t addr = (ix + x) + strideY * (iy + y) + strideZ * (iz + z);
				uint8_t *dst = volume + addr;

				const float tx = ((ix + x) * voxelSize[0] - posx) / l.PAx;
				const float ty = ((iy + y) * voxelSize[1] - posy) / l.PAy;
				const float tz = ((iz + z) * voxelSize[2] - posz) / l.PAz;
				const float e = expf(-0.5f * tx * tx -0.5f * ty * ty -0.5f * tz * tz);
				const uint8_t val = static_cast<uint8_t>(fminf((255.f/windowSize) * e + *dst, 255.f));
				// write max to volume
				*dst = std::max(*dst, val);
			}
		}
	}
}

}

Volume Rendering::render_cpu(Localizations &locs, std::array<float,3> voxelSize, int windowSize)
{
	std::array<int,3> dims;
	dims[0] = static_cast<int>(std::ceilf(locs.width()  / voxelSize[0]));
	dims[1] = static_cast<int>(std::ceilf(locs.height() / voxelSize[1]));
	dims[2] = static_cast<int>(std::ceilf(locs.depth()  / voxelSize[2]));

	std::array<float,3> orgin{0.f, 0.f, locs.minZ()};

	Volume volume(dims, voxelSize, orgin);
	volume.fill(0);
	for (const auto &l : locs)
		drawPSF_cpu(volume.data(), l, dims, voxelSize, orgin, windowSize);

	return volume;
}

void Rendering::renderHistgram3D_cpu(const Localizations &locs, uint32_t *output, std::array<int,3> size, std::array<float,3> voxelSize, std::array<float,3> origin)
{
	const size_t voxels = 1ull * size[0] * size[1] * size[2];
	const size_t stride[3] = {1ull, 1ull * size[0], 1ull * size[0] * size[1]};

	std::fill_n(output, voxels, 0u);

	// count localizations in each voxel
	for (const auto &l : locs) {
		const int x = static_cast<int>(std::round(((l.x - origin[0]) / voxelSize[0])));
		const int y = static_cast<int>(std::round(((l.y - origin[1]) / voxelSize[1])));
		const int z = static_cast<int>(std::round(((l.z - origin[2]) / voxelSize[2])));

		if (x >= 0 && x < size[0] && y >= 0 && y < size[1] && z >= 0 && z < size[2])
			output[stride[0] * x + stride[1] * y + stride[2] * z]++;
	}
}

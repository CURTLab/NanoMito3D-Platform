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

#include <cmath>
#include <algorithm>

void drawPSF(uint8_t *volume, const Localization &l, const std::array<int,3> &volumeDims, const std::array<float,3> &voxelSize, const std::array<float,3> &origin, int windowSize)
{
	const float posx = l.x - origin[0];
	const float posy = l.y - origin[1];
	const float posz = l.z - origin[2];

	const int ix = static_cast<int>(std::round((posx / voxelSize[0])));
	const int iy = static_cast<int>(std::round((posy / voxelSize[1])));
	const int iz = static_cast<int>(std::round((posz / voxelSize[2])));

	const int w = windowSize/2;
	for (int z = -w; z <= w; ++z) {
		for (int y = -w; y <= w; ++y) {
			for (int x = -w; x <= w; ++x) {
				if ((ix + x < 0) || (iy + y < 0) || (iz + z < 0) ||
					 (ix + x >= volumeDims[0]) || (iy + y >= volumeDims[1]) || (iz + z >= volumeDims[2]))
					continue;
				const float tx = ((ix + x) * voxelSize[0] - posx) / l.PAx;
				const float ty = ((iy + y) * voxelSize[1] - posy) / l.PAy;
				const float tz = ((iz + z) * voxelSize[2] - posz) / l.PAz;
				const float e = (255.f/windowSize)*expf(-0.5f * tx * tx -0.5f * ty * ty -0.5f * tz * tz);
				auto &val = volume[ix + x + volumeDims[0] * (iy + y) + volumeDims[0] * volumeDims[1] * (iz + z)];
				val = static_cast<uint8_t>(std::clamp(val + e, 0.f, 255.f));
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
		drawPSF(volume.data(), l, dims, voxelSize, orgin, windowSize);

	return volume;
}

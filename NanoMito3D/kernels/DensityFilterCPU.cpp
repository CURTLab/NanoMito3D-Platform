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

#include "DensityFilter.h"

#include "3dparty/CompactNSearch/CompactNSearch.h"

using namespace CompactNSearch;

Localizations::const_iterator DensityFilter::remove_cpu(Localizations &locs, size_t minPoints, float radius)
{
	const size_t nLocs = locs.size();

	std::unique_ptr<float[]> pts(new float[nLocs * 3]);
	for (size_t i = 0; i < nLocs; ++i) {
		pts[3*i + 0] = locs[i].x;
		pts[3*i + 1] = locs[i].y;
		pts[3*i + 2] = locs[i].z;
	}

	NeighborhoodSearch nsearch(radius);

	auto pointSetIndex = nsearch.add_point_set(pts.get(), nLocs, false, true);
	nsearch.find_neighbors();

	auto &pointSet = nsearch.point_set(pointSetIndex);

	const auto ret = std::remove_if(locs.begin(), locs.end(), [&locs,&pointSet,minPoints](const Localization &l) -> bool {
		const uint32_t idx = static_cast<uint32_t>(&l - locs.data());
		return pointSet.n_neighbors(0, idx) < minPoints;
	});

	return ret;
}

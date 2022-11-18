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

#include "DensityFilter.h"

#include "Octree.h"

Localizations::const_iterator DensityFilter::remove_cpu(Localizations &locs, int minPoints, float radius)
{
	// filter by density
	Octree<uint32_t,float,50> tree(locs.bounds());
	for (uint32_t i = 0; i < locs.size(); ++i)
		tree.insert(locs[i].position(), i);

	return std::remove_if(locs.begin(), locs.end(), [&tree](const Localization &l) {
		const float radius = 250;
		int minPoints = 10;
		const auto pts = tree.countInSphere(l.position(), radius);
		return pts < minPoints;
	});
}

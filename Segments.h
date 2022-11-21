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
#ifndef SEGMENTS_H
#define SEGMENTS_H

#include <vector>
#include <string>

#include "Volume.h"

struct Segment
{
	uint32_t numBranches;
	uint32_t numEndPoints;
	uint32_t numJunctionVoxels;
	uint32_t numJunctions;
	uint32_t numSlabs;
	uint32_t numTriples;
	uint32_t numQuadruples;
	float averageBranchLength;
	float maximumBranchLength;
	float shortestPath;
	uint32_t voxels;
	uint32_t width;
	uint32_t height;
	uint32_t depth;
};

class Segments : public std::vector<Segment>
{
public:
	Segments();

	Volume volume;

};

#endif // SEGMENTS_H

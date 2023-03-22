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
#include <opencv2/opencv.hpp>

#include "Volume.h"
#include "SkeletonGraph.h"
#include "Bounds.h"

struct SegmentData
{
	float numBranches;
	float numEndPoints;
	float numJunctionVoxels;
	float numJunctions;
	float numSlabs;
	float numTriples;
	float numQuadruples;
	float averageBranchLength;
	float maximumBranchLength;
	float shortestPath;
	float voxels;
	float width;
	float height;
	float depth;
	float signalCount;
};

struct Segment
{
	std::shared_ptr<SkeletonGraph> graph;
	std::vector<std::array<float,3>> endPoints;
	SegmentData data;
	Bounds<int> boundingBox;
	Volume vol;
	int prediction = 0;
	int id = -1;
	std::string fileName;
};

class Segments : public std::vector<std::shared_ptr<Segment>>
{
public:
	Segments() = default;

	Volume volume;

};

#endif // SEGMENTS_H

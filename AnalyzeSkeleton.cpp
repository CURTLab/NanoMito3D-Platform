/****************************************************************************
 * AnalyzeSkeleton_ plugin for ImageJ.
 *
 * Copyright (C) 2008 - 2017 Ignacio Arganda-Carreras.
 * Copyright (C) 2022 Fabian Hauser, adatpted for C++
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/gpl-3.0.html>.
 *
 ****************************************************************************/

#include "AnalyzeSkeleton.h"

#include <stdexcept>
#include <string>
#include <stack>

std::tuple<Volume, int, Bounds<int>> Skeleton3D::Trees::extractVolume(const Volume &volume, int threshold, int tree) const
{
	Volume ret(volume.size(), volume.voxelSize(), volume.origin());
	Volume visited(volume.size());
	const auto &t = at(tree);
	Point p;
	if (t.graph->root() != nullptr)
		p = t.graph->root()->firstPoint();
	else
		p = t.spStartPosition;
	Bounds<int> cuboid(volume.width(), 0, volume.height(), 0, volume.depth(), 0);

	std::stack<Point> stack;

	auto condAdd = [&stack,&visited](int x, int y, int z)
	{
		if (visited.contains(x, y, z) && !visited(x, y, z))
			stack.push({x, y, z});
	};

	int voxels = 0;
	stack.push(p);
	while(!stack.empty()) {
		Point u = stack.top(); stack.pop();
		if (volume(u.x, u.y, u.z) > threshold) {
			ret.setValue(u.x, u.y, u.z, 255);
			visited(u.x, u.y, u.z) = 1;

			cuboid.minX = std::min(cuboid.minX, u.x);
			cuboid.minY = std::min(cuboid.minY, u.y);
			cuboid.minZ = std::min(cuboid.minZ, u.z);
			cuboid.maxX = std::max(cuboid.maxX, u.x);
			cuboid.maxY = std::max(cuboid.maxY, u.y);
			cuboid.maxZ = std::max(cuboid.maxZ, u.z);

			++voxels;

			condAdd(u.x-1, u.y-1, u.z-1);
			condAdd(u.x  , u.y-1, u.z-1);
			condAdd(u.x+1, u.y-1, u.z-1);
			condAdd(u.x-1, u.y,   u.z-1);
			condAdd(u.x,   u.y,   u.z-1);
			condAdd(u.x+1, u.y,   u.z-1);
			condAdd(u.x-1, u.y+1, u.z-1);
			condAdd(u.x,   u.y+1, u.z-1);
			condAdd(u.x+1, u.y+1, u.z-1);
			condAdd(u.x-1, u.y-1, u.z  );
			condAdd(u.x,   u.y-1, u.z  );
			condAdd(u.x+1, u.y-1, u.z  );
			condAdd(u.x-1, u.y,   u.z  );
			condAdd(u.x+1, u.y,   u.z  );
			condAdd(u.x-1, u.y+1, u.z  );
			condAdd(u.x,   u.y+1, u.z  );
			condAdd(u.x+1, u.y+1, u.z  );
			condAdd(u.x-1, u.y-1, u.z+1);
			condAdd(u.x,   u.y-1, u.z+1);
			condAdd(u.x+1, u.y-1, u.z+1);
			condAdd(u.x-1, u.y,   u.z+1);
			condAdd(u.x,   u.y,   u.z+1);
			condAdd(u.x+1, u.y,   u.z+1);
			condAdd(u.x-1, u.y+1, u.z+1);
			condAdd(u.x,   u.y+1, u.z+1);
			condAdd(u.x+1, u.y+1, u.z+1);
		}
	}

	return {ret, voxels, cuboid};
}

const Skeleton3D::Trees &Skeleton3D::Analysis::calculate(Volume input, Volume originalImage, PruningMode pruneMode, bool pruneEnds, double pruneThreshold, bool shortPath)
{
	m_visited = Volume(input.size());
	m_visited.fill(0);
	m_inputImage = input;

	bool doPruneCycles = false;
	switch (pruneMode) {
	case Skeleton3D::Analysis::ShortestBranch:
		doPruneCycles = true;
		break;
	case Skeleton3D::Analysis::LowestIntensityVoxel:
	case Skeleton3D::Analysis::LowestIntensityBranch:
		doPruneCycles = true;
		m_originalImage = originalImage;
		break;
	default: break;
	}

	// initialize visit flags
	resetVisited();

	// Tag skeleton, differentiate trees and visit them
	processSkeleton(input);

	// prune ends
	if (pruneEnds) {
		pruneEndBranches(m_inputImage, m_taggedImage, pruneThreshold);
	}

	// Prune cycles if necessary
	if (doPruneCycles) {
		if (pruneCycles(m_inputImage, m_originalImage, pruneMode)) {
			// initialize visit flags
			resetVisited();
			// Recalculate analysis over the new image
			doPruneCycles = false;
			processSkeleton(m_inputImage);
		}
	}

	// Calculate triple points (junctions with exactly 3 branches)
	calculateTripleAndQuadruplePoints();

	if (shortPath) {
		m_shortPathImage = Volume(input.size(), input.voxelSize(), input.origin());
		m_shortPathImage.fill(0);

		// Visit skeleton and measure distances.
		// and apply Warshall algorithm
		for (auto &t : m_trees) {
			t.shortestPath = warshallAlgorithm(t.graph, m_shortPathImage, t.shortestPathPoints);
			if (!t.shortestPathPoints.empty()) {
				t.spStartPosition = t.shortestPathPoints.front();
				t.spEndPosition = t.shortestPathPoints.back();
			}
		}
	}

	return m_trees;
}

void Skeleton3D::Analysis::resetVisited()
{
	m_visited.fill(0);
}

void Skeleton3D::Analysis::processSkeleton(Volume input)
{
	m_listOfEndVoxels.clear();
	m_listOfJunctionVoxels.clear();
	m_listOfSlabVoxels.clear();
	m_listOfStartingSlabVoxels.clear();

	m_totalNumberOfEndPoints = 0;
	m_totalNumberOfJunctionVoxels = 0;
	m_totalNumberOfSlabVoxels = 0;

	m_taggedImage = tagImage(input);

	// Mark trees
	markTrees(m_taggedImage, m_labeledSkeletons);

	if (m_numOfTrees == 0)
		return;

	// Ask memory for every tree
	initializeTrees();

	// Divide groups of end-points and junction voxels
	if (m_numOfTrees == 1) {
		m_trees[0].endPoints = m_listOfEndVoxels;
		m_trees[0].numberOfEndPoints = static_cast<uint32_t>(m_listOfEndVoxels.size());
		m_trees[0].junctionVoxels = m_listOfJunctionVoxels;
		m_trees[0].numberOfJunctionVoxels = static_cast<uint32_t>(m_listOfJunctionVoxels.size());
		m_trees[0].startingSlab = m_listOfStartingSlabVoxels;
	} else if (m_numOfTrees > 1) {
		divideVoxelsByTrees(m_labeledSkeletons);
	}

	// Calculate number of junctions (skipping neighbor junction voxels)
	groupJunctions();

	// Mark all unvisited
	resetVisited();

	// Visit skeleton and measure distances.
	for (int i = 0; i < m_numOfTrees; i++)
		visitSkeleton(i);
}

Volume Skeleton3D::Analysis::tagImage(Volume input)
{
	Volume outputImage(input.size(), input.voxelSize(), input.origin());
	outputImage.fill(0);

	const int width = input.width();
	const int height = input.height();
	const int depth = input.depth();

	for (int z = 0; z < depth; z++) {
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				if (input(x, y, z) == 0)
					continue;
				int neighbors = numberOfNeighbors(input, x, y, z);
				if (neighbors < 2) {
					outputImage.setValue(x, y, z, END_POINT);
					++m_totalNumberOfEndPoints;
					m_listOfEndVoxels.push_back({x, y, z});
				} else if (neighbors > 2) {
					outputImage.setValue(x, y, z, JUNCTION);
					++m_totalNumberOfJunctionVoxels;
					m_listOfJunctionVoxels.push_back({x, y, z});
				} else {
					outputImage.setValue(x, y, z, SLAB);
					++m_totalNumberOfSlabVoxels;
					m_listOfSlabVoxels.push_back({x, y, z});
				}
			}
		}
	}

	return outputImage;
}

bool Skeleton3D::Analysis::markTrees(const Volume &taggedImage, GenericVolume<int> &outputImage)
{
	outputImage.alloc(taggedImage.width(), taggedImage.height(), taggedImage.depth());
	std::fill_n(outputImage.data, static_cast<size_t>(outputImage.width) * outputImage.height * outputImage.depth, 0);
	m_numOfTrees = 0;

	int color = 0;

	// Visit trees starting at end points
	for (const Point &endPointCoord : m_listOfEndVoxels) {
		if (isVisited(endPointCoord))
			continue;
		++color;

		if (color == std::numeric_limits<int>::max()) {
			throw std::runtime_error("More than" + std::to_string(std::numeric_limits<int>::max()-1) + "skeletons in the image!");
			return false;
		}

		// Visit the entire tree.
		visitTree(endPointCoord, outputImage, color);
		++m_numOfTrees;
	}

	// Visit trees starting at junction points
	// (some circular trees do not have end points)
	// Visit trees starting at end points
	for (const Point &junctionCoord : m_listOfJunctionVoxels) {
		if (isVisited(junctionCoord))
			continue;
		++color;

		if (color == std::numeric_limits<int>::max()) {
			throw std::runtime_error("More than" + std::to_string(std::numeric_limits<int>::max()-1) + "skeletons in the image!");
			return false;
		}

		int length = visitTree(junctionCoord, outputImage, color);
		if (length == 0) {
			color--; // the color was not used
			continue;
		}

		// increase number of trees
		++m_numOfTrees;
	}

	// Check for unvisited slab voxels
	// (just in case there are circular trees without junctions)
	for (const Point &slab : m_listOfSlabVoxels) {
		if (isVisited(slab))
			continue;

		m_listOfStartingSlabVoxels.push_back(slab);
		++color;

		if (color == std::numeric_limits<int>::max()) {
			throw std::runtime_error("More than" + std::to_string(std::numeric_limits<int>::max()-1) + "skeletons in the image!");
			return false;
		}

		// else, visit branch until next junction or end point.
		int length = visitTree(slab, outputImage, color);
		if (length == 0) {
			--color; // the color was not used
			continue;
		}

		++m_numOfTrees;
	}

	// Reset visited variable
	resetVisited();

	return true;
}

int Skeleton3D::Analysis::visitTree(const Point &startingPoint, GenericVolume<int> &outputImage, int color)
{
	int numOfVoxels = 0;

	if (isVisited(startingPoint))
		return 0;

	outputImage(startingPoint.x, startingPoint.y, startingPoint.z) = color;
	setVisited(startingPoint, true);

	std::list<Point> toRevisit;
	// Add starting point to revisit list if it is a junction
	if (isJunction(startingPoint))
		toRevisit.push_back(startingPoint);

	auto nextPoint = nextUnvisitedVoxel(startingPoint);
	while (nextPoint || !toRevisit.empty()) {
		if (nextPoint) {
			++numOfVoxels;
			outputImage(nextPoint->x, nextPoint->y, nextPoint->z) = color;
			setVisited(*nextPoint, true);

			if (isJunction(*nextPoint))
				toRevisit.push_back(*nextPoint);

			// Calculate next point to visit
			nextPoint = nextUnvisitedVoxel(*nextPoint);
		} else {
			// Calculate next point to visit
			nextPoint = nextUnvisitedVoxel(toRevisit.front());
			if (!nextPoint)
				toRevisit.pop_front();
		}
	}

	return numOfVoxels;
}

void Skeleton3D::Analysis::initializeTrees()
{
	m_trees.resize(m_numOfTrees);
	for (auto &t : m_trees) {
		t.numberOfBranches = 0;
		t.numberOfEndPoints = 0;
		t.numberOfJunctionVoxels = 0;
		t.numberOfJunctions = 0;
		t.numberOfSlabs = 0;
		t.numberOfTriplePoints = 0;
		t.numberOfQuadruplePoints = 0;
		t.averageBranchLength = 0.0;
		t.maximumBranchLength = 0.0;
		t.shortestPath = 0.0;
		t.endPoints.clear();
		t.junctionVoxels.clear();
		t.startingSlab.clear();
		t.singleJunctions.clear();
		t.graph = std::make_shared<SkeletonGraph>();
	}
}

void Skeleton3D::Analysis::divideVoxelsByTrees(GenericVolume<int> &treeIS)
{
	// Add end points to the corresponding tree
	for (const auto &p : m_listOfEndVoxels)
		m_trees[treeIS(p.x, p.y, p.z) - 1].endPoints.push_back(p);

	// Add junction voxels to the corresponding tree
	for (const auto &p : m_listOfJunctionVoxels)
		m_trees[treeIS(p.x, p.y, p.z) - 1].junctionVoxels.push_back(p);

	// Add special slab voxels to the corresponding tree
	for (const auto &p : m_listOfStartingSlabVoxels)
		m_trees[treeIS(p.x, p.y, p.z) - 1].startingSlab.push_back(p);

	// Assign number of end points and junction voxels per tree
	for (int i = 0; i < m_numOfTrees; ++i) {
		m_trees[i].numberOfEndPoints = static_cast<uint32_t>(m_trees[i].endPoints.size());
		m_trees[i].numberOfJunctionVoxels = static_cast<uint32_t>(m_trees[i].junctionVoxels.size());
	}
}

void Skeleton3D::Analysis::groupJunctions()
{
	// Mark all unvisited
	resetVisited();

	for (int i = 0; i < m_numOfTrees; ++i) {
		for (const auto &p : m_trees[i].junctionVoxels) {
			if (!isVisited(p))
				fusionNeighborJunction(p, m_trees[i].singleJunctions);
		}
	}

	for (auto &tree : m_trees) {
		tree.numberOfJunctions = static_cast<uint32_t>(tree.singleJunctions.size());

		// Create array of junction vertices for the graph
		tree.junctionVertex.resize(tree.numberOfJunctions);
		for (size_t j = 0; j < tree.numberOfJunctions; ++j) {
			const auto &list = tree.singleJunctions[j];
			tree.junctionVertex[j] = std::make_shared<Vertex>();
			for (const auto &p : list)
				tree.junctionVertex[j]->addPoint(p);
		}
	}

	// Mark all unvisited
	resetVisited();
}

void Skeleton3D::Analysis::fusionNeighborJunction(const Point &startingPoint, std::vector<std::vector<Point> > &singleJunctions)
{
	std::vector<Point> newGroup;
	newGroup.push_back(startingPoint);

	// Mark the starting junction as visited
	setVisited(startingPoint, true);

	// Look for neighbor junctions and add them to the new group
	std::list<Point> toRevisit;
	toRevisit.push_back(startingPoint);

	auto nextPoint = nextUnvisitedJunctionVoxel(startingPoint);
	while (nextPoint || !toRevisit.empty()) {
		if (nextPoint && !isVisited(*nextPoint)) {
			// Add to the group
			newGroup.push_back(*nextPoint);
			// Mark as visited
			setVisited(*nextPoint, true);
			// add it to the revisit list
			toRevisit.push_back(*nextPoint);
			// Calculate next junction point to visit
			nextPoint = nextUnvisitedJunctionVoxel(*nextPoint);
		} else {
			nextPoint = toRevisit.front();
			// Calculate next point to visit
			nextPoint = nextUnvisitedJunctionVoxel(*nextPoint);
			// Maintain junction in the list until there is no more branches
			if (!nextPoint)
				toRevisit.pop_front();
		}
	}
	singleJunctions.push_back(newGroup);
}

void Skeleton3D::Analysis::visitSkeleton(int currentTree)
{
	// length of branches
	double branchLength = 0.0;

	Tree &tree = m_trees[currentTree];
	tree.maximumBranchLength = 0.0;
	tree.numberOfSlabs = 0;

	for (auto &vertex : tree.junctionVertex)
		tree.graph->addVertex(vertex);

	// Visit branches starting at end points
	for (size_t i = 0; i < tree.numberOfEndPoints; ++i) {
		const Point &endPointCoord = tree.endPoints[i];
		// Skip when visited
		if (isVisited(endPointCoord))
			continue;

		// Initial vertex
		VertexPtr v1 = std::make_shared<Vertex>();
		v1->addPoint(endPointCoord);
		tree.graph->addVertex(v1);
		if (i == 0)
			tree.graph->setRoot(v1);

		m_slabList.clear();

		// Otherwise, visit branch until next junction or end point.
		std::array<double,4> properties = visitBranch(endPointCoord, currentTree);
		double length = properties[0];
		//double color3rd = properties[1];
		//double color = properties[2];
		//double length_ra = properties[3];

		// If length is 0, it means the tree is formed by only one voxel.
		if (length == 0.0) {
			// If there is an adjacent visited junction, count it
			// as a single voxel branch
			auto aux = visitedJunctionNeighbor(endPointCoord, v1);
			if (aux) {
				m_auxFinalVertex = findPointVertex(tree.junctionVertex, *aux);
				length += calculateDistance(endPointCoord, *aux);

				// Add the length to the first point of the vertex (to prevent later from having
				// euclidean distances larger than the actual distance)
				length += calculateDistance(m_auxFinalVertex->firstPoint(), endPointCoord);

				tree.graph->addVertex(m_auxFinalVertex);
				tree.graph->addEdge(v1, m_auxFinalVertex, m_slabList, length, PointEdge);
				++tree.numberOfBranches;

				branchLength += length;
			} else {
				//qWarning() << "set initial point to final point";
			}
			continue;
		}

		// If the final point is a slab, then we add the path to the
		// neighbor junction voxel not belonging to the initial vertex
		// (unless it is a self loop)
		if (isSlab(*m_auxPoint)) {
			Point aux = *m_auxPoint;

			m_auxPoint = visitedJunctionNeighbor(aux, v1);
			if (m_auxPoint) {
				m_auxFinalVertex = findPointVertex(tree.junctionVertex, *m_auxPoint);
				length += calculateDistance(*m_auxPoint, aux);
			} else {
				m_auxFinalVertex = v1;
				m_auxPoint = aux;
			}

			// Add the length to the first point of the vertex (to prevent later from having
			// euclidean distances larger than the actual distance)
			length += calculateDistance(m_auxFinalVertex->firstPoint(), *m_auxPoint);
		}

		tree.graph->addVertex(m_auxFinalVertex);
		tree.graph->addEdge(v1, m_auxFinalVertex, m_slabList, length, EndpointEdge);
		++tree.numberOfBranches;
		branchLength += length;
		tree.maximumBranchLength = std::max(tree.maximumBranchLength, length);
	}

	// If there is no end points, set the first junction as root.
	if (tree.numberOfEndPoints == 0 && tree.junctionVoxels.size() > 0) {
		tree.graph->setRoot(tree.junctionVertex.front());
	}

	// Now visit branches starting at junctions
	// 08/26/2009 Changed the loop to visit first the junction voxels that are
	//            forming a single junction.

	for (const auto &item : tree.junctionVertex) {
		for (const Point &junctionCoord : item->points()) {
			// Mark junction as visited
			setVisited(junctionCoord, true);
			auto nextPoint = nextUnvisitedVoxel(junctionCoord);
			while (nextPoint) {
				// Do not count adjacent junctions
				if (!isJunction(*nextPoint)) {
					// Create graph edge
					m_slabList.clear();
					m_slabList.push_back(*nextPoint);
					tree.numberOfSlabs++;

					// Calculate distance from junction to that point
					double length = calculateDistance(junctionCoord, *nextPoint);

					// Visit branch
					m_auxPoint = {};

					auto properties = visitBranch(*nextPoint, currentTree);
					length += properties[0];

					// Increase number of branches
					if (length != 0) {
						if (!m_auxPoint)
							m_auxPoint = *nextPoint;

						tree.numberOfBranches++;

						VertexPtr initialVertex;
						for (const auto &item1 : tree.junctionVertex) {
							if (item1->isVertexPoint(junctionCoord)) {
								initialVertex = item1;
								break;
							}
						}

						// If the final point is a slab, then we add the path to the
						// neighbor junction voxel not belonging to the initial vertex
						// (unless it is a self loop)
						if (isSlab(*m_auxPoint)) {
							Point aux = *m_auxPoint;
							m_auxPoint = visitedJunctionNeighbor(aux, initialVertex);
							if (!m_auxPoint) {
								m_auxPoint = aux;
								m_auxFinalVertex = initialVertex;
							} else {
								m_auxFinalVertex = findPointVertex(tree.junctionVertex, *m_auxPoint);
								length += calculateDistance(*m_auxPoint, aux);
							}
						}

						// Add the distance between the main vertex of the junction
						// and the initial junction vertex of the branch (this prevents from
						// having branches in the graph larger than the calculated branch length)
						length += calculateDistance(initialVertex->firstPoint(), junctionCoord);
						tree.graph->addEdge(initialVertex, m_auxFinalVertex, m_slabList, length, JunctionEdge);

						// Increase total length of branches
						branchLength += length;

						// update maximum branch length
						tree.maximumBranchLength = std::max(tree.maximumBranchLength, length);
					}
				} else {
					// Is junction point
					setVisited(*nextPoint, true);
				}
				nextPoint = nextUnvisitedVoxel(junctionCoord);
			}
		}
	}

	// Finally visit branches starting at slabs (special case for circular trees)
	if (tree.startingSlab.size() == 1) {
		const Point &startCoord = tree.startingSlab.front();

		// Create circular graph (only one vertex)
		auto v1 = std::make_shared<Vertex>();
		v1->addPoint(startCoord);

		m_slabList.clear();
		m_slabList.push_back(startCoord);

		tree.numberOfSlabs++;

		// visit branch until finding visited voxel.
		auto properties = visitBranch(startCoord, currentTree);
		const double length = properties[0];

		if (length != 0) {
			tree.numberOfBranches++;
			branchLength += length;
			// update maximum branch length
			tree.maximumBranchLength = std::max(tree.maximumBranchLength, length);
		}

		// Create circular edge
		tree.graph->addEdge(v1, v1, m_slabList, length, CircularEdge);
	}

	if (tree.numberOfBranches == 0)
		return;

	tree.averageBranchLength = branchLength / tree.numberOfBranches;

	return;
}

std::array<double, 4> Skeleton3D::Analysis::visitBranch(const Point &startingPoint, int iTree)
{
	double length = 0;
	double intensity = 0.0;
	double intensity3rd = 0.0;
	double length_ra = 0.0;
	std::array<double, 4> ret{length, intensity, intensity3rd, length_ra};

	Tree &tree = m_trees[iTree];

	std::list<Point> pointHistory;
	pointHistory.push_back(startingPoint);

	// mark starting point as visited
	setVisited(startingPoint, true);

	// Get next unvisited voxel
	auto nextPoint = nextUnvisitedVoxel(startingPoint);
	if (!nextPoint)
		return ret;

	Point previousPoint = startingPoint;
	// We visit the branch until we find an end point or a junction
	while (nextPoint && isSlab(*nextPoint)) {
		++tree.numberOfSlabs;

		// Add slab voxel to the edge
		m_slabList.push_back(*nextPoint);

		// Add length
		length += calculateDistance(previousPoint, *nextPoint);
		pointHistory.push_front(*nextPoint);

		length_ra += calculateDistance(pointHistory);

		// Mark as visited
		setVisited(*nextPoint, true);

		// Move in the graph
		previousPoint = *nextPoint;
		nextPoint = nextUnvisitedVoxel(previousPoint);
	}

	// If we find an unvisited end-point or junction, we set it
	// as final vertex of the branch
	if (nextPoint) {
		// Add distance to last point
		length += calculateDistance(previousPoint, *nextPoint);
		pointHistory.push_front(*nextPoint);
		length_ra += calculateDistance(pointHistory);

		// Mark last point as visited
		setVisited(*nextPoint, true);

		// Mark final vertex
		if (isEndPoint(*nextPoint)) {
			m_auxFinalVertex = std::make_shared<Vertex>();
			m_auxFinalVertex->addPoint(*nextPoint);
		} else if (isJunction(*nextPoint)) {
			m_auxFinalVertex = findPointVertex(tree.junctionVertex, *nextPoint);
			if (m_auxFinalVertex) {
				// Add the length to the first point of the vertex (to prevent later from having
				// euclidean distances larger than the actual distance)
				length += calculateDistance(m_auxFinalVertex->firstPoint(), *nextPoint);
				length_ra += calculateDistance(m_auxFinalVertex->firstPoint(), *nextPoint);
			}
		}

		m_auxPoint = nextPoint;
	} else {
		m_auxPoint = previousPoint;
	}

	// calculate average intensity (thickness) value, but only take the inner third of a branch.
	// at both ends the intensity (thickness) is most likely affected by junctions.
	size_t size = pointHistory.size();
	size_t start = (size_t) (size / 3.0);
	size_t end = (size_t) (2 * size / 3.0);

	auto it = pointHistory.begin();
	for (int i = 0; i < size; ++i, ++it) {
		int value = m_inputImage.value(it->x, it->y, it->z);
		if (value < 0) {
			value += 256;
		}
		intensity += value;
		if (i >= start && i < end) {
			intensity3rd += value;
		}
	}

	intensity /= size;
	intensity3rd /= (end - start);

	ret[0] = length;
	ret[1] = intensity3rd;
	ret[2] = intensity;
	ret[3] = length_ra;

	return ret;
}

void Skeleton3D::Analysis::pruneEndBranches(Volume &stack, Volume &taggedImage, double length)
{
	for (auto &t : m_trees) {
		auto &graph = t.graph;

		auto it = graph->vertexBegin();
		while (it != graph->vertexEnd()) {
			const auto &v = it->second;
			if (v->branches().size() == 1 && v->branches().front()->length <= length) {
				// Remove end point voxels
				for (const auto &p : v->points()) {
					stack(p.x, p.y, p.z) = 0;
					taggedImage(p.x, p.y, p.z) = 0;
					t.numberOfEndPoints--;
					m_totalNumberOfEndPoints--;
					for (auto it = m_listOfEndVoxels.begin(); it != m_listOfEndVoxels.end(); ++it) {
						if (*it == p) {
							m_listOfEndVoxels.erase(it);
							break;
						}
					}
				}

				// Remove branch voxels
				EdgePtr branch = v->branches().front();
				for (const auto &p : branch->slab) {
					stack(p.x, p.y, p.z) = 0;
					taggedImage(p.x, p.y, p.z) = 0;
					t.numberOfSlabs--;
					m_totalNumberOfSlabVoxels--;
					for (auto it = m_listOfSlabVoxels.begin(); it != m_listOfSlabVoxels.end(); ++it) {
						if (*it == p) {
							m_listOfSlabVoxels.erase(it);
							break;
						}
					}
				}

				// remove the Edge from the Graph
				graph->removeEdge(branch);

				// remove the Vertex from the Graph
				it = graph->vertexErase(it);
			} else {
				++it;
			}
		}

		// reassign ids if any ends where prune
		graph->reassignIds();
		continue;
	}
}

bool Skeleton3D::Analysis::pruneCycles(Volume &inputImage, const Volume &originalImage, PruningMode pruningMode)
{
	bool pruned = false;

	for (const auto &t : m_trees) {
		if (t.startingSlab.size() == 1) {
			const auto &p = t.startingSlab.front();
			inputImage.setValue(p.x, p.y, p.z, 0);
			pruned = true;
		} else {
			auto backEdges = t.graph->detectCycles();

			// If DFS returned backEdges, we need to delete the loops
			for (const auto &e : backEdges) {
				std::vector<EdgePtr> loopEdges;
				loopEdges.push_back(e);

				EdgePtr minEdge = e;

				// backtracking (starting at the vertex with higher order index
				auto v1 = t.graph->vertex(e->v1);
				auto v2 = t.graph->vertex(e->v2);

				auto finalLoopVertex = v1->visitOrder() < v2->visitOrder() ? v1 : v2;
				auto backtrackVertex = v1->visitOrder() < v2->visitOrder() ? v2 : v1;

				// backtrack until reaching final loop vertex
				while (finalLoopVertex != backtrackVertex) {
					// Extract predecessor
					const auto pre = backtrackVertex->predecessor();
					// Update shortest loop edge if necessary
					if ((pruningMode == ShortestBranch) && (pre->slab.size() < minEdge->slab.size())) {
						minEdge = pre;
					}
					// Add to loop edge list
					loopEdges.push_back(pre);
					// Extract predecessor
					backtrackVertex = t.graph->vertex(pre->v1 == backtrackVertex->id() ? pre->v2 : pre->v1);
				}

				// Prune cycle
				switch (pruningMode) {
				case ShortestBranch:
					// Remove middle slab from the shortest loop edge
					Point p;
					if (minEdge->slab.size() > 0) {
						p = minEdge->slab[minEdge->slab.size() / 2];
					} else {
						p = t.graph->vertex(minEdge->v1)->firstPoint();
					}
					inputImage.setValue(p.x, p.y, p.z, 0);
					break;
				case LowestIntensityVoxel:
					removeLowestIntensityVoxel(loopEdges, inputImage, originalImage);
					break;
				case LowestIntensityBranch:
					cutLowestIntensityBranch(loopEdges, t.graph, inputImage, originalImage);
					break;
				default:
					break;
				}
			}// endfor backEdges

			pruned = true;
		}
	}
	return pruned;
}

void Skeleton3D::Analysis::removeLowestIntensityVoxel(const std::vector<EdgePtr> &loopEdges, Volume &inputImage, const Volume &originalGrayImage)
{
	std::optional<Point> lowestIntensityVoxel;
	double lowestIntensityValue = std::numeric_limits<double>::max();

	for (const auto &e : loopEdges) {
		for (const Point &p : e->slab) {
			const double avg = averageNeighborhoodValue(originalGrayImage, p);
			if (avg < lowestIntensityValue) {
				lowestIntensityValue = avg;
				lowestIntensityVoxel = p;
			}
		}
	}

	// Cut loop in the lowest intensity pixel value position
	if (lowestIntensityVoxel) {
		inputImage.setValue(lowestIntensityVoxel->x,
								  lowestIntensityVoxel->y,
								  lowestIntensityVoxel->z, 0);
	}
}

void Skeleton3D::Analysis::cutLowestIntensityBranch(const std::vector<EdgePtr> &loopEdges, const GraphPtr &graph, Volume &inputImage, const Volume &originalGrayImage)
{
	EdgePtr lowestIntensityEdge;
	double lowestIntensityValue = std::numeric_limits<double>::max();
	std::optional<Point> cutPoint;

	for (const auto &e : loopEdges) {
		// Calculate average intensity of the edge neighborhood
		double min_val = std::numeric_limits<double>::max();
		std::optional<Point> darkestPoint;

		double edgeIntensity = 0;
		double n_vox = 0;

		// Check slab points
		for (const Point &p : e->slab) {
			const double avg = averageNeighborhoodValue(originalGrayImage, p);
			// Keep track of the darkest slab point of the edge
			if (avg < min_val) {
				min_val = avg;
				darkestPoint = p;
			}
			edgeIntensity += avg;
			n_vox++;
		}
		// Check vertices
		for (const Point& p : graph->vertex(e->v1)->points()) {
			edgeIntensity += averageNeighborhoodValue(originalGrayImage, p);
			n_vox++;
		}
		for (const Point &p : graph->vertex(e->v2)->points()) {
			edgeIntensity += averageNeighborhoodValue(originalGrayImage, p);
			n_vox++;
		}

		if (n_vox != 0) {
			edgeIntensity /= n_vox;
		}

		// Keep track of the lowest intensity edge
		if (edgeIntensity < lowestIntensityValue) {
			lowestIntensityEdge = e;
			lowestIntensityValue = edgeIntensity;
			cutPoint = darkestPoint;
		}
	}

	// Cut loop in the lowest intensity branch medium position
	Point removeCoords;
	if (cutPoint && lowestIntensityEdge && !lowestIntensityEdge->slab.empty()) {
		removeCoords = *cutPoint;
	} else {
		throw std::runtime_error("Lowest intensity branch without slabs?!: vertex" + std::to_string(lowestIntensityEdge->v1));
		removeCoords = graph->vertex(lowestIntensityEdge->v1)->firstPoint();
	}

	inputImage.setValue(removeCoords.x,
							  removeCoords.y,
							  removeCoords.z, 0);
}

void Skeleton3D::Analysis::calculateTripleAndQuadruplePoints()
{
	uint8_t neighborhood[27];
	for (auto &tree : m_trees) {
		// Visit the groups of junction voxels
		for (const auto &groupOfJunctions : tree.singleJunctions) {
			// Count the number of slab and end-points neighbors of every voxel in the group
			int nBranch = 0;
			for (const auto &p : groupOfJunctions) {
				getNeighborhood(m_taggedImage, p.x, p.y, p.z, neighborhood);
				for (int k = 0; k < 27; k++) {
					if (neighborhood[k] == SLAB || neighborhood[k] == END_POINT)
						++nBranch;
				}
			}
			// If the junction has only 3 slab/end-point neighbors, then it is a triple point
			if (nBranch == 3)
				tree.numberOfTriplePoints++;
			else if (nBranch == 4) // quadruple point if 4
				tree.numberOfQuadruplePoints++;
		}
	}
}

double Skeleton3D::Analysis::calculateDistance(const Point &point1, const Point &point2)
{
	const double dx = (point1.x - point2.x);
	const double dy = (point1.y - point2.y);
	const double dz = (point1.z - point2.z);
	return sqrt(dx * dx + dy * dy + dz * dz);
}

double Skeleton3D::Analysis::calculateDistance(const std::list<Point> &points)
{
	const int indexOfLast = static_cast<int>(points.size()) - 1;

	//no Distance to be calculated here...
	if (indexOfLast < 1)
		return 0.0;

	// Point Of Interest
	const int poi = 5;

	//poi is indexOflast if List is shorter than 5
	std::list<Point>::const_iterator it;
	if (indexOfLast < poi) {
		it = (--points.end());
	} else {
		it = points.begin();
		std::advance(it, poi);
	}
	return calculateDistance(*it, points.front()) / poi;
}

int Skeleton3D::Analysis::numberOfNeighbors(const Volume &v, int x, int y, int z)
{
	int count = 0;
	count += (v.value(x-1, y-1, z-1) != 0);
	count += (v.value(x  , y-1, z-1) != 0);
	count += (v.value(x+1, y-1, z-1) != 0);
	count += (v.value(x-1, y,   z-1) != 0);
	count += (v.value(x,   y,   z-1) != 0);
	count += (v.value(x+1, y,   z-1) != 0);
	count += (v.value(x-1, y+1, z-1) != 0);
	count += (v.value(x,   y+1, z-1) != 0);
	count += (v.value(x+1, y+1, z-1) != 0);
	count += (v.value(x-1, y-1, z  ) != 0);
	count += (v.value(x,   y-1, z  ) != 0);
	count += (v.value(x+1, y-1, z  ) != 0);
	count += (v.value(x-1, y,   z  ) != 0);
	count += (v.value(x+1, y,   z  ) != 0);
	count += (v.value(x-1, y+1, z  ) != 0);
	count += (v.value(x,   y+1, z  ) != 0);
	count += (v.value(x+1, y+1, z  ) != 0);
	count += (v.value(x-1, y-1, z+1) != 0);
	count += (v.value(x,   y-1, z+1) != 0);
	count += (v.value(x+1, y-1, z+1) != 0);
	count += (v.value(x-1, y,   z+1) != 0);
	count += (v.value(x,   y,   z+1) != 0);
	count += (v.value(x+1, y,   z+1) != 0);
	count += (v.value(x-1, y+1, z+1) != 0);
	count += (v.value(x,   y+1, z+1) != 0);
	count += (v.value(x+1, y+1, z+1) != 0);
	return count;
}

void Skeleton3D::Analysis::getNeighborhood(const Volume &v, int x, int y, int z, uint8_t neighborhood[27])
{
	neighborhood[ 0] = v.value(x-1, y-1, z-1);
	neighborhood[ 1] = v.value(x  , y-1, z-1);
	neighborhood[ 2] = v.value(x+1, y-1, z-1);

	neighborhood[ 3] = v.value(x-1, y,   z-1);
	neighborhood[ 4] = v.value(x,   y,   z-1);
	neighborhood[ 5] = v.value(x+1, y,   z-1);

	neighborhood[ 6] = v.value(x-1, y+1, z-1);
	neighborhood[ 7] = v.value(x,   y+1, z-1);
	neighborhood[ 8] = v.value(x+1, y+1, z-1);

	neighborhood[ 9] = v.value(x-1, y-1, z  );
	neighborhood[10] = v.value(x,   y-1, z  );
	neighborhood[11] = v.value(x+1, y-1, z  );

	neighborhood[12] = v.value(x-1, y,   z  );
	neighborhood[13] = v.value(x,   y,   z  );
	neighborhood[14] = v.value(x+1, y,   z  );

	neighborhood[15] = v.value(x-1, y+1, z  );
	neighborhood[16] = v.value(x,   y+1, z  );
	neighborhood[17] = v.value(x+1, y+1, z  );

	neighborhood[18] = v.value(x-1, y-1, z+1);
	neighborhood[19] = v.value(x,   y-1, z+1);
	neighborhood[20] = v.value(x+1, y-1, z+1);

	neighborhood[21] = v.value(x-1, y,   z+1);
	neighborhood[22] = v.value(x,   y,   z+1);
	neighborhood[23] = v.value(x+1, y,   z+1);

	neighborhood[24] = v.value(x-1, y+1, z+1);
	neighborhood[25] = v.value(x,   y+1, z+1);
	neighborhood[26] = v.value(x+1, y+1, z+1);
}

std::optional<Point> Skeleton3D::Analysis::nextUnvisitedVoxel(const Volume &volume, const Volume &visited, const Point &v)
{
	std::optional<Point> unvisitedNeighbor;
	for (int x = -1; x < 2; x++) {
		for (int y = -1; y < 2; y++) {
			for (int z = -1; z < 2; z++) {
				if (x == 0 && y == 0 && z == 0)
					continue;

				const int dx = v.x + x, dy = v.y + y, dz = v.z + z;
				if (volume.value(dx, dy, dz) != 0
					 && (visited(dx, dy, dz) == 0))
				{
					unvisitedNeighbor = Point{dx, dy, dz};
					break;
				}
			}
		}
	}
	return unvisitedNeighbor;
}

double Skeleton3D::Analysis::averageNeighborhoodValue(const Volume &v, const Point &p)
{
	uint8_t neighborhood[27];
	getNeighborhood(v, p.x, p.y, p.z, neighborhood);

	double avg = 0.0;
	for (int i = 0; i < 27; ++i)
		avg += neighborhood[i];
	return avg / 27.0;
}

double Skeleton3D::Analysis::warshallAlgorithm(const GraphPtr &graph, Volume &shortPathImage, std::vector<Point> &shortestPathPoints)
{
	const auto &edgeList = graph->edgeList();
	const auto &vertexList = graph->vertices();

	const auto vn = vertexList.size();

	if (vertexList.size() == 0)
		return 0.0;

	// check for paths of only one vertex
	if (vertexList.size() == 1) {
		shortestPathPoints.push_back(vertexList.front()->firstPoint());
		return 0.0;
	}

	double maxPath = 0;

	std::vector<double> adjacencyMatrix(vn * vn, std::numeric_limits<double>::infinity());
	std::vector<int> predecessorMatrix(vn * vn, -1);

	// use lambda functions for matrix access
	auto set = [vn](auto &m, int x, int y, const auto &val) { m[x * vn + y] = val; };
	auto get = [vn](auto &m, int x, int y) { return m[x * vn + y]; };

	for (const auto &e : edgeList) {
		const int row = e->v1;
		const int column = e->v2;
		set(adjacencyMatrix, row, row, 0);
		set(adjacencyMatrix, column, column, 0);
		set(adjacencyMatrix, row, column, e->length);
		set(adjacencyMatrix, column, row, e->length);

		set(predecessorMatrix, row, row, -1);
		set(predecessorMatrix, column, column, -1);
		set(predecessorMatrix, row, column, row);
		set(predecessorMatrix, column, row, column);
	}

	// the warshall algorithm with k as candidate vertex and i and j walk through the adjacencyMatrix
	// the predecessor matrix is updated at the same time.
	for (int k = 0; k < vn; k++) {
		for (int i = 0; i < vn; i++) {
			for (int j = 0; j < vn; j++) {
				if (get(adjacencyMatrix,i,k) + get(adjacencyMatrix,k,j) < get(adjacencyMatrix,i,j)) {
					set(adjacencyMatrix,i,j,get(adjacencyMatrix,i,k) + get(adjacencyMatrix,k,j));
					set(predecessorMatrix,i,j,get(predecessorMatrix,k,j));
				}
			}
		}
	}

	int a = 0, b = 0;
	// find the maximum of all shortest paths
	for (int i = 0; i < vertexList.size(); i++) {
		for (int j = 0; j < vertexList.size(); j++) {
			// sometimes infinities still remain
			if (get(adjacencyMatrix,i,j) > maxPath && !isinf(get(adjacencyMatrix,i,j))) {
				maxPath = get(adjacencyMatrix,i,j);
				a = i;
				b = j;
			}
		}
	}

	// We know the first and last vertex of the longest shortest path, namely a and b
	// using the predecessor matrix we can now determine the path that is taken from a to b
	// remember a and b are indices and not the actual vertices.
	while (b != a) {
		const auto &predecessor = get(predecessorMatrix,a,b);
		const auto &endvertex = b;
		std::vector<EdgePtr> sp_edgeslist;
		double lengthtest = std::numeric_limits<double>::infinity();
		EdgePtr shortestedge;

		// search all edges for a combination of the two vertices
		for (const auto &e : edgeList) {
			if ((e->v1 == predecessor && e->v2 == endvertex) || (e->v1 == endvertex && e->v2 == predecessor)) {
				// sometimes there are multiple edges between two vertices so add them to a list
				// for a second test
				sp_edgeslist.push_back(e);
			}
		}
		// the second test
		// this test looks which edge has the shortest length in sp_edgeslist
		for (const auto &e : sp_edgeslist) {
			if (e->length < lengthtest) {
				shortestedge = e;
				lengthtest = e->length;
			}
		}
		// add vertex 1 points
		const auto &v1 = graph->vertex(shortestedge->v2 != predecessor
				? shortestedge->v2 : shortestedge->v1);
		for (Point p : v1->points()) {
			const auto it = std::find(shortestPathPoints.begin(), shortestPathPoints.end(), p);
			if (it == shortestPathPoints.end()) {
				shortestPathPoints.push_back(p);
				//setPixel(this.shortPathImage, p.x, p.y, p.z, SHORTEST_PATH);
			}
		}

		// add slab points of the shortest edge to the list of points
		std::vector<Point> slabs;
		if (shortestedge->v2 != predecessor) {
			slabs = std::vector<Point>(shortestedge->slab.rbegin(), shortestedge->slab.rend());
		} else {
			slabs = std::vector<Point>(shortestedge->slab.begin(), shortestedge->slab.end());
		}

		for (const Point &p : slabs) {
			shortestPathPoints.push_back(p);
			shortPathImage(p.x, p.y, p.z) = SHORTEST_PATH;
		}

		// add vertex 2 points too
		const auto &v2 = graph->vertex(shortestedge->v2 != predecessor
				? shortestedge->v1 : shortestedge->v2);
		for (Point p : v2->points()) {
			const auto it = std::find(shortestPathPoints.begin(), shortestPathPoints.end(), p);
			if (it == shortestPathPoints.end()) {
				shortestPathPoints.push_back(p);
				//setPixel(this.shortPathImage, p.x, p.y, p.z, SHORTEST_PATH);
			}
		}
		// now make the index of the endvertex the index of the predecessor so that the path now goes from
		// a to predecessor and repeat cycle
		b = get(predecessorMatrix,a,b);
	}

	return maxPath;
}

std::optional<Point> Skeleton3D::Analysis::nextUnvisitedVoxel(const Point &v) const
{
	std::optional<Point> unvisitedNeighbor;
	for (int x = -1; x < 2; x++) {
		for (int y = -1; y < 2; y++) {
			for (int z = -1; z < 2; z++) {
				if (x == 0 && y == 0 && z == 0)
					continue;

				const int dx = v.x + x, dy = v.y + y, dz = v.z + z;
				if (m_inputImage.value(dx, dy, dz) != 0
					 && !isVisited(dx, dy, dz))
				{
					unvisitedNeighbor = Point{dx, dy, dz};
					break;
				}
			}
		}
	}
	return unvisitedNeighbor;
}

std::optional<Point> Skeleton3D::Analysis::nextUnvisitedJunctionVoxel(const Point &v) const
{
	std::optional<Point> unvisitedNeighbor;
	for (int x = -1; x < 2; x++) {
		for (int y = -1; y < 2; y++) {
			for (int z = -1; z < 2; z++) {
				if (x == 0 && y == 0 && z == 0)
					continue;

				const int dx = v.x + x, dy = v.y + y, dz = v.z + z;
				if (m_inputImage.value(dx, dy, dz) != 0
					 && !isVisited(dx, dy, dz)
					 && isJunction(dx, dy, dz))
				{
					unvisitedNeighbor = Point{dx, dy, dz};
					break;
				}
			}
		}
	}
	return unvisitedNeighbor;
}

std::optional<Point> Skeleton3D::Analysis::visitedJunctionNeighbor(const Point &v, const VertexPtr &exclude) const
{
	std::optional<Point> unvisitedNeighbor;
	for (int x = -1; x < 2; x++) {
		for (int y = -1; y < 2; y++) {
			for (int z = -1; z < 2; z++) {
				if (x == 0 && y == 0 && z == 0)
					continue;

				const int dx = v.x + x, dy = v.y + y, dz = v.z + z;
				if (m_inputImage.value(dx, dy, dz) != 0
					 && isVisited(dx, dy, dz)
					 && isJunction(dx, dy, dz)
					 && !exclude->isVertexPoint(dx, dy, dz))
				{
					unvisitedNeighbor = Point{dx, dy, dz};
					break;
				}
			}
		}
	}
	return unvisitedNeighbor;
}

VertexPtr Skeleton3D::Analysis::findPointVertex(const std::vector<VertexPtr> &vertices, const Point &v) const
{
	for (auto &vertex : vertices) {
		if (vertex->isVertexPoint(v))
			return vertex;
	}
	return nullptr;
}

const Volume &Skeleton3D::Analysis::taggedImage() const
{
	return m_taggedImage;
}

const Volume &Skeleton3D::Analysis::shortPathImage() const
{
	return m_shortPathImage;
}

std::vector<Point> Skeleton3D::Analysis::traverse(const VertexPtr &v1, const VertexPtr &v2) const
{
	return traverse(v1->firstPoint(), v2);
}

std::vector<Point> Skeleton3D::Analysis::traverse(const Point &startingPoint, const VertexPtr &endVertex) const
{
	Volume visited(m_inputImage.size());
	visited.setValue(startingPoint.x, startingPoint.y, startingPoint.z, 1);

	std::vector<Point> ret;
	// Get next unvisited voxel
	while (true) {
		ret.clear();
		ret.push_back(startingPoint);

		auto nextPoint = nextUnvisitedVoxel(m_inputImage, visited, startingPoint);
		if (!nextPoint)
			return {};

		// We visit the branch until we find an end point or a junction
		while (nextPoint && isSlab(*nextPoint)) {
			ret.push_back(*nextPoint);

			// Mark as visited
			visited.setValue(nextPoint->x, nextPoint->y, nextPoint->z, 1);

			// Move in the graph
			nextPoint = nextUnvisitedVoxel(m_inputImage, visited, *nextPoint);
		}

		// If we find an unvisited end-point or junction, we set it
		// as final vertex of the branch
		if (nextPoint && endVertex->isVertexPoint(*nextPoint)) {
			ret.push_back(*nextPoint);
			return ret;
		} else if (nextPoint) {
			visited.setValue(nextPoint->x, nextPoint->y, nextPoint->z, 1);
		}
	}

	return {};
}

Volume Skeleton3D::Analysis::segmented(const Volume &thresholded, int voxelThreshold)
{
	Volume ret(thresholded.size(), thresholded.voxelSize(), thresholded.origin());
	Volume visited(thresholded.size());

	std::stack<Point> stack;
	std::vector<Point> stack2(thresholded.voxels()/4);

	auto condAdd = [&stack,&visited](int x, int y, int z)
	{
		if (visited.contains(x, y, z) && !visited(x, y, z))
			stack.push({x, y, z});
	};

	uint8_t color = 0;
	for (const auto &t : m_trees) {
		const auto &p = t.graph->root()->firstPoint();

		if (color == 255) {
			throw std::runtime_error("Skeleton3D:segmented: reached maximum of 255 segments!");
			break;
		}

		++color;

		stack.push(p);

		while(!stack.empty()) {
			Point u = stack.top(); stack.pop();
			if (thresholded(u.x, u.y, u.z) != 0) {
				ret(u.x, u.y, u.z) = color;
				visited(u.x, u.y, u.z) = 1;
				stack2.push_back({u.x, u.y, u.z});

				condAdd(u.x-1, u.y-1, u.z-1);
				condAdd(u.x  , u.y-1, u.z-1);
				condAdd(u.x+1, u.y-1, u.z-1);
				condAdd(u.x-1, u.y,   u.z-1);
				condAdd(u.x,   u.y,   u.z-1);
				condAdd(u.x+1, u.y,   u.z-1);
				condAdd(u.x-1, u.y+1, u.z-1);
				condAdd(u.x,   u.y+1, u.z-1);
				condAdd(u.x+1, u.y+1, u.z-1);
				condAdd(u.x-1, u.y-1, u.z  );
				condAdd(u.x,   u.y-1, u.z  );
				condAdd(u.x+1, u.y-1, u.z  );
				condAdd(u.x-1, u.y,   u.z  );
				condAdd(u.x+1, u.y,   u.z  );
				condAdd(u.x-1, u.y+1, u.z  );
				condAdd(u.x,   u.y+1, u.z  );
				condAdd(u.x+1, u.y+1, u.z  );
				condAdd(u.x-1, u.y-1, u.z+1);
				condAdd(u.x,   u.y-1, u.z+1);
				condAdd(u.x+1, u.y-1, u.z+1);
				condAdd(u.x-1, u.y,   u.z+1);
				condAdd(u.x,   u.y,   u.z+1);
				condAdd(u.x+1, u.y,   u.z+1);
				condAdd(u.x-1, u.y+1, u.z+1);
				condAdd(u.x,   u.y+1, u.z+1);
				condAdd(u.x+1, u.y+1, u.z+1);
			}
		}

		if (stack2.size() < voxelThreshold) {
			--color;
			for (const auto &u : stack2)
				ret(u.x, u.y, u.z) = 0;
		}
		stack2.clear();
	}

	return ret;
}

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

#ifndef ANALYZESKELETON_H
#define ANALYZESKELETON_H

#include "SkeletonGraph.h"
#include "Volume.h"
#include "Bounds.h"

namespace Skeleton3D
{

struct Tree
{
	uint32_t numberOfBranches;
	uint32_t numberOfEndPoints;
	uint32_t numberOfJunctionVoxels;
	uint32_t numberOfJunctions;
	uint32_t numberOfSlabs;
	uint32_t numberOfTriplePoints;
	uint32_t numberOfQuadruplePoints;
	double averageBranchLength;
	double maximumBranchLength;
	double shortestPath;
	std::vector<Point> endPoints;
	std::vector<Point> junctionVoxels;
	std::vector<Point> startingSlab;
	// list of groups of junction voxels that belong to the same tree junction
	std::vector<std::vector<Point>> singleJunctions;
	std::vector<VertexPtr> junctionVertex;
	std::vector<Point> shortestPathPoints;
	Point spStartPosition;
	Point spEndPosition;
	std::shared_ptr<SkeletonGraph> graph;
};

class Trees : public std::vector<Tree>
{
public:
	std::tuple<Volume, int, Bounds<int> > extractVolume(const Volume &volume, int threshold, int tree) const;

private:
	friend class Analysis;

	// disable inherited add functions
	using std::vector<Tree>::push_back;
	using std::vector<Tree>::insert;

};

template<class T>
struct GenericVolume
{
	inline GenericVolume()
		: width(0), height(0), depth(0)
		, data(nullptr)
	{}
	inline constexpr GenericVolume(int w, int h, int d)
		: width(w), height(h), depth(d)
		, data(new T[static_cast<size_t>(w) * h * d])
	{}
	inline ~GenericVolume() { delete [] data; }

	inline void alloc(int w, int h, int d) {
		delete [] data;
		width = w; height = h; depth = d;
		data = new T[static_cast<size_t>(w) * h * d];
	}

	inline constexpr const size_t idx(int x, int y, int z) {
		return x * 1ull + y * static_cast<size_t>(width) + z * static_cast<size_t>(width) * height;
	}

	inline constexpr const T &operator()(int x, int y, int z) const
	{ return data[idx(x, y, z)]; }

	inline constexpr T &operator()(int x, int y, int z)
	{ return data[idx(x, y, z)]; }

	GenericVolume(GenericVolume &) = delete;
	GenericVolume &operator =(GenericVolume &) = delete;

	int width;
	int height;
	int depth;
	T *data;
};

class Analysis
{
public:
	// end point flag
	static constexpr uint8_t END_POINT = 30;
	// junction flag
	static constexpr uint8_t JUNCTION = 70;
	// slab flag
	static constexpr uint8_t SLAB = 127;
	// shortest path flag
	static constexpr uint8_t SHORTEST_PATH = 96;

	enum PruningMode {
		// no pruning mode
		NoPruning,
		ShortestBranch,
		LowestIntensityVoxel,
		LowestIntensityBranch
	};

	// default constructor
	Analysis() = default;

	const Trees &calculate(Volume input,
								  Volume originalImage = {},
								  PruningMode pruneMode = NoPruning,
								  bool pruneEnds = false,
								  double pruneThreshold = 0.0,
								  bool shortPath = false);

	const Volume &taggedImage() const;
	const Volume &shortPathImage() const;

	inline const std::vector<Point> &listOfEndPoints(int tree) const
	{ return m_trees[tree].endPoints; }

	inline const std::shared_ptr<SkeletonGraph> &graph(int tree) const
	{ return m_trees[tree].graph; }

	std::vector<Point> traverse(const VertexPtr &v1, const VertexPtr &v2) const;
	std::vector<Point> traverse(const Point &startingPoint, const VertexPtr &endVertex) const;

	Volume segmented(const Volume &thresholded, int voxelThreshold = 0);

private:
	// Reset visit variable and set it to false
	void resetVisited();
	// Process skeleton: tag image, mark trees and visit
	void processSkeleton(Volume input);
	// Tag skeleton dividing the voxels between end points, junctions and slabs
	Volume tagImage(Volume input);
	// Color the different trees in the skeleton
	bool markTrees(const Volume &taggedImage, GenericVolume<int> &outputImage);
	// Visit tree marking the voxels with a reference tree color
	int visitTree(const Point &startingPoint, GenericVolume<int> &outputImage, int color);
	// Reset and allocate memory for the trees
	void initializeTrees();
	// Divide the end point, junction and special (starting) slab voxels in the corresponding tree lists
	void divideVoxelsByTrees(GenericVolume<int> &treeIS);
	// Calculate number of junction skipping neighbor junction voxels
	void groupJunctions();
	// Fusion neighbor junctions voxels into the same list
	void fusionNeighborJunction(const Point &startingPoint, std::vector<std::vector<Point>> &singleJunctions);
	// Visit skeleton starting at end-points, junctions and slab of circular skeletons, and record measurements
	void visitSkeleton(int currentTree);
	// Visit a branch and calculate length
	std::array<double,4> visitBranch(const Point &startingPoint, int iTree);
	// Prune end branches of a specific length
	void pruneEndBranches(Volume &stack, Volume &taggedImage, double length);
	// Prune cycles from tagged image and update it
	bool pruneCycles(Volume &inputImage, const Volume &originalImage, PruningMode pruningMode);
	// Cut the a list of edges in the lowest pixel intensity voxel (calculated from the original -grayscale- image)
	void removeLowestIntensityVoxel(const std::vector<EdgePtr> &loopEdges, Volume &inputImage, const Volume &originalGrayImage);
	// Cut the a list of edges in the lowest pixel intensity branch
	void cutLowestIntensityBranch(const std::vector<EdgePtr> &loopEdges, const GraphPtr &graph, Volume &inputImage, const Volume &originalGrayImage);
	// Calculate number of triple and quadruple points in the skeleton. Triple and quadruple points are junctions with exactly 3 and 4 branches respectively
	void calculateTripleAndQuadruplePoints();

	static double calculateDistance(const Point &point1, const Point &point2);
	static double calculateDistance(const std::list<Point> &points);
	static int numberOfNeighbors(const Volume &v, int x, int y, int z);
	static void getNeighborhood(const Volume &v, int x, int y, int z, uint8_t neighbors[27]);
	static std::optional<Point> nextUnvisitedVoxel(const Volume &volume, const Volume &visited, const Point &v);
	static double averageNeighborhoodValue(const Volume &v, const Point &p);
	static double warshallAlgorithm(const GraphPtr &graph, Volume &shortPathImage, std::vector<Point> &shortestPathPoints);
	std::optional<Point> nextUnvisitedVoxel(const Point &v) const;
	std::optional<Point> nextUnvisitedJunctionVoxel(const Point &v) const;
	std::optional<Point> visitedJunctionNeighbor(const Point &v, const VertexPtr &exclude) const;

	VertexPtr findPointVertex(const std::vector<VertexPtr > &vertices, const Point &v) const;

	inline bool isEndPoint(const Point &v) const { return m_taggedImage(v.x, v.y, v.z) == END_POINT; }
	inline bool isEndPoint(int x, int y, int z) const { return m_taggedImage(x, y, z) == END_POINT; }
	inline bool isJunction(const Point &v) const { return m_taggedImage(v.x, v.y, v.z) == JUNCTION; }
	inline bool isJunction(int x, int y, int z) const { return m_taggedImage(x, y, z) == JUNCTION; }
	inline bool isSlab(const Point &v) const { return m_taggedImage(v.x, v.y, v.z) == SLAB; }
	inline bool isSlab(int x, int y, int z) const { return m_taggedImage(x, y, z) == SLAB; }

	inline bool isVisited(const Point &v) const { return m_visited.value(v.x, v.y, v.z) != 0; }
	inline bool isVisited(int x, int y, int z) const { return m_visited.value(x, y, z) != 0; }

	inline void setVisited(const Point &v, bool b) { m_visited.setValue(v.x, v.y, v.z, b); }
	inline void setVisited(int x, int y, int z, bool b) { m_visited.setValue(x, y, z, b); }

	Volume m_inputImage;
	Volume m_taggedImage;
	Volume m_visited;
	Volume m_originalImage;
	Volume m_shortPathImage;
	GenericVolume<int> m_labeledSkeletons;

	Trees m_trees;

	std::vector<Point> m_listOfEndVoxels;
	std::vector<Point> m_listOfJunctionVoxels;
	std::vector<Point> m_listOfSlabVoxels;
	std::vector<Point> m_listOfStartingSlabVoxels;
	int m_totalNumberOfEndPoints;
	int m_totalNumberOfJunctionVoxels;
	int m_totalNumberOfSlabVoxels;

	std::vector<Point> m_slabList;

	int m_numOfTrees;

	// auxiliary temporary point+
	std::optional<Point> m_auxPoint;

	// auxiliary final vertex
	VertexPtr m_auxFinalVertex;

};

} // namespace Skeleton3D

#endif // ANALYZESKELETON_H

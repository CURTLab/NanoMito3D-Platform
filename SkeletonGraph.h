/****************************************************************************
 * Skeletonize3D plugin for ImageJ(C).
 * Copyright (C) 2008 Ignacio Arganda-Carreras
 * Copyright (C) 2022 Fabian Hauser, adatpted for C++
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation (http://www.gnu.org/licenses/gpl.txt )
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 ****************************************************************************/

#ifndef SKELETONGRAPH_H
#define SKELETONGRAPH_H

#include <vector>
#include <optional>
#include <memory>
#include <unordered_map>

struct Point
{
	int x, y, z;

	inline double distance(const Point &p) const {
		const double dx = (x - p.x), dy = (y - p.y), dz = (z - p.z);
		return sqrt(dx * dx + dy * dy + dz * dz);
	}
};

inline bool operator==(const Point &p1, const Point &p2)
{ return p1.x == p2.x && p1.y == p2.y && p1.z == p2.z; }

enum EdgeType {
	UndefinedEdge,
	TreeEdge,
	PointEdge,
	EndpointEdge,
	JunctionEdge,
	CircularEdge
};

struct SkeletonEdge
{
	inline SkeletonEdge(int v1, int v2, double len, const std::vector<Point> &slabs, EdgeType t)
		: v1(v1), v2(v2), length(len), slab(slabs), type(t) {}
	inline constexpr int oppositeVertex(int vid) const { return vid == v1 ? v2 : v1; }

	int v1;
	int v2;
	double length;
	std::vector<Point> slab;
	EdgeType type;
};
using EdgePtr = std::shared_ptr<SkeletonEdge>;

class Vertex
{
public:
	inline Vertex() : m_id(-1), m_visitOrder(-1) {}

	inline void addPoint(const Point &v) { m_points.push_back(v); }
	inline void clear() { m_points.clear(); }

	inline bool isVertexPoint(int x, int y, int z) const {
		for (const auto &p : m_points) {
			if (p.x == x && p.y == y && p.z == z)
				return true;
		}
		return false;
	}
	inline bool isVertexPoint(const Point &v) const { return isVertexPoint(v.x, v.y, v.z); }

	inline const Point &firstPoint() const { return m_points.front(); }
	inline constexpr const std::vector<Point> &points() const { return m_points; }

	inline constexpr int id() const { return m_id; }
	inline constexpr void setId(const int &id) { m_id = id; }

	inline EdgePtr predecessor() const { return m_predecessor; }
	inline void setPredecessor(EdgePtr e) { m_predecessor = e; }

	inline constexpr int visitOrder() const { return m_visitOrder; }
	inline constexpr void setVisitOrder(int order)  { m_visitOrder = order; }

	inline void setBranch(const EdgePtr &e) { m_branches.push_back(e); }
	inline constexpr const std::vector<EdgePtr> &branches() const { return m_branches; }
	inline void removeBranch(const EdgePtr &e) { m_branches.erase(std::find(m_branches.begin(), m_branches.end(), e)); }

private:
	std::vector<Point> m_points;
	std::vector<EdgePtr> m_branches;
	int m_id;
	EdgePtr m_predecessor;
	int m_visitOrder;

};
using VertexPtr = std::shared_ptr<Vertex>;

class SkeletonGraph
{
public:
	using VertexMap = std::unordered_map<int,VertexPtr>;

	SkeletonGraph();

	void addVertex(const VertexPtr &vertex);

	const VertexPtr &root() const;
	void setRoot(const VertexPtr &vertex);

	std::vector<VertexPtr> vertices() const;
	const std::vector<EdgePtr> &edgeList() const;

	// id based getters
	const VertexPtr &vertex(int id) const;
	std::vector<int> edges(int id) const;
	double length(int src, int dest) const;

	// vertex smart pointer based modifieres
	void addEdge(const VertexPtr &v1, const VertexPtr &v2, const std::vector<Point> &slab, double length, EdgeType type);
	EdgePtr findEdge(const VertexPtr &v1, const VertexPtr &v2) const;

	// Depth first search method to detect cycles in the graph.
	std::vector<EdgePtr> detectCycles() const;

	inline VertexMap::iterator vertexBegin() { return m_vertex.begin(); }
	inline VertexMap::iterator vertexEnd() { return m_vertex.end(); }
	inline VertexMap::iterator vertexErase(VertexMap::iterator it) { return m_vertex.erase(it); }

	void removeEdge(EdgePtr edge);

	void reassignIds();

private:
	static double distSqr(const Point &p1, const Point &p2);
	EdgePtr findEdge(int v1, int v2) const;

	std::vector<EdgePtr> m_edgeList;
	VertexPtr m_root;
	VertexMap m_vertex;

};
using GraphPtr = std::shared_ptr<SkeletonGraph>;

#endif // SKELETONGRAPH_H

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


#include "SkeletonGraph.h"

#include <queue>
#include <stack>
#include <algorithm>
#include <iterator>

SkeletonGraph::SkeletonGraph()
{
}

void SkeletonGraph::addVertex(const VertexPtr &vertex)
{
	if (vertex->id() < 0) vertex->setId(static_cast<int>(m_vertex.size()));
	m_vertex[vertex->id()] = vertex;
}

const VertexPtr &SkeletonGraph::root() const
{
	return m_root;
}

void SkeletonGraph::setRoot(const VertexPtr &vertex)
{
	m_root = vertex;
}

std::vector<VertexPtr> SkeletonGraph::vertices() const
{
	std::vector<VertexPtr> ret;
	ret.reserve(static_cast<int>(m_vertex.size()));
	for (const auto &v :m_vertex)
		ret.push_back(v.second);
	return ret;
}

const VertexPtr &SkeletonGraph::vertex(int id) const
{
	return m_vertex.at(id);
}

std::vector<int> SkeletonGraph::edges(int id) const
{
	const auto &list = m_vertex.at(id)->branches();
	std::vector<int> ret;
	std::transform(list.begin(), list.end(), std::back_inserter(ret), [](const EdgePtr &e){ return e->v2; });
	return ret;
}

double SkeletonGraph::length(int src, int dest) const
{
	const auto &e = findEdge(src, dest);
	if (e)
		return e->length;
	return std::numeric_limits<double>::infinity();
}

void SkeletonGraph::addEdge(const VertexPtr &v1, const VertexPtr &v2, const std::vector<Point> &slab, double length, EdgeType type)
{
	type = UndefinedEdge;
	auto edge1 = std::make_shared<SkeletonEdge>(v1->id(), v2->id(), length, slab, type);
	v1->setBranch(edge1);
	if (v1->id() != v2->id()) {
		v2->setBranch(edge1);
	}
	m_edgeList.push_back(edge1);
}

EdgePtr SkeletonGraph::findEdge(const VertexPtr &v1, const VertexPtr &v2) const
{
	return findEdge(v1->id(), v2->id());
}

double SkeletonGraph::distSqr(const Point &p1, const Point &p2)
{
	const double dx = (p1.x - p2.x);
	const double dy = (p1.y - p2.y);
	const double dz = (p1.z - p2.z);
	return dx * dx + dy * dy + dz * dz;
}

const std::vector<EdgePtr> &SkeletonGraph::edgeList() const
{
	return m_edgeList;
}

std::vector<EdgePtr> SkeletonGraph::detectCycles() const
{
	const int n = static_cast<int>(m_vertex.size());
	if (!m_root || n == 0)
		return {};

	std::vector<EdgePtr> backEdges;
	std::vector<bool> visited(n, false);

	int u = m_root ? m_root->id() : m_vertex.begin()->second->id();

	std::stack<int> stack;
	stack.push(u);

	int visitOrder = 0;
	while (!stack.empty()) {
		int t = stack.top(); stack.pop();
		if (!visited[t]) {
			auto v = m_vertex.at(t);

			// If the vertex has not been visited yet, then
			// the edge from the predecessor to this vertex
			// is mark as TREE
			if (v->predecessor())
				v->predecessor()->type = TreeEdge;

			// mark as visited
			visited[t] = true;
			v->setVisitOrder(visitOrder++);

			for (const auto &e : v->branches()) {
				// For the undefined branches:
				// We push the unvisited vertices in the stack,
				// and mark the edge to the others as BACK
				//qDebug() << e->srcId << e->v2;
				if (e->type == UndefinedEdge) {
					int ov = e->oppositeVertex(t);
					if (!visited[ov]) {
						stack.push(ov);
						m_vertex.at(ov)->setPredecessor(e);
					} else {
						e->type = CircularEdge;
						backEdges.push_back(e);
					}
				}
			}
		}
	}
	return backEdges;
}

void SkeletonGraph::removeEdge(EdgePtr edge)
{
	// remove edge from edge list
	m_edgeList.erase(std::find(m_edgeList.begin(), m_edgeList.end(), edge));
	m_vertex[edge->v1]->removeBranch(edge);
	m_vertex[edge->v2]->removeBranch(edge);
}

void SkeletonGraph::reassignIds()
{
	std::vector<VertexPtr> vec(m_vertex.size());
	int id = 0;
	for (auto it = m_vertex.begin(); it != m_vertex.end(); ++it, ++id) {
		auto &v = it->second;
		for (auto &e : v->branches()) {
			e->v1 = e->v1 == v->id() ? id : e->v1;
			e->v2 = e->v2 == v->id() ? id : e->v2;
		}
		v->setId(id);
		vec[id] = v;
	}
	m_vertex.clear();
	for (size_t i = 0; i < vec.size(); ++i)
		m_vertex[static_cast<int>(i)] = vec[i];
}

EdgePtr SkeletonGraph::findEdge(int v1, int v2) const
{
	auto cond = [v2](const EdgePtr &e) { return e->v2 == v2; };
	auto &list = m_vertex.at(v1)->branches();
	auto it = std::find_if(list.begin(), list.end(), cond);
	if (it != list.end())
		return *it;
	return nullptr;
}

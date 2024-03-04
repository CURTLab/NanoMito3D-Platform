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

#ifndef OCTREE_H
#define OCTREE_H

#include <memory>
#include <algorithm>

#include "Bounds.h"

template <typename T, typename P, size_t CAPACITY>
class Octree
{
public:
	inline constexpr Octree(P minX, P maxX, P minY, P maxY, P minZ, P maxZ) noexcept
		: m_bounds(minX, maxX, minY, maxY, minZ, maxZ)
		, m_size(0)
	{}

	inline constexpr Octree(const Bounds<P> &bound) noexcept
		: m_bounds(bound)
		, m_size(0)
	{}

	inline size_t size() const noexcept;
	inline constexpr bool empty() const noexcept { return (m_size == 0); }

	inline bool insert(const std::array<P,3> &p, T value) noexcept;

	inline bool isDivided() const noexcept;

	inline size_t countInSphere(const std::array<P,3> &pos, P radius) const;
	inline size_t countInBox(const Bounds<P> &bound) const;
	inline size_t countInBox(const std::array<P,3> &pos, const std::array<P,3> &size) const
	{ return countInBox(Bounds<P>(pos, size[0], size[1], size[2])); }

private:
	std::unique_ptr<Octree<T,P,CAPACITY>> m_childs[8];
	Bounds<P> m_bounds;
	std::pair<std::array<P,3>, T> m_data[CAPACITY];
	size_t m_size;

};

template<typename T, typename P, size_t CAPACITY>
size_t Octree<T, P, CAPACITY>::size() const noexcept
{
	if (m_size == 0ull)
		return 0ull;
	size_t ret = m_size;
	for (uint8_t i = 0; i < 8; ++i) {
		if (m_childs[i])
			ret += m_childs[i]->size();
	}
	return ret;
}

template<typename T, typename P, size_t CAPACITY>
bool Octree<T, P, CAPACITY>::insert(const std::array<P, 3> &p, T value) noexcept
{
	if (!m_bounds.contains(p))
		return false;

	if ((m_size < CAPACITY) && !isDivided()) {
		m_data[m_size++] = std::make_pair(p, value);
		return true;
	}

	const P hsizex = m_bounds.width()/P(2), hsizey = m_bounds.height()/P(2), hsizez = m_bounds.depth()/P(2);
	const Bounds<P> sub[8] = {
		Bounds<P>(m_bounds.origin(), hsizex, hsizey, hsizez),
		Bounds<P>(m_bounds.originOffsetted(hsizex, P(0), P(0)), hsizex, hsizey, hsizez),
		Bounds<P>(m_bounds.originOffsetted(0, hsizey, 0), hsizex, hsizey, hsizez),
		Bounds<P>(m_bounds.originOffsetted(hsizex, hsizey, 0), hsizex, hsizey, hsizez),
		Bounds<P>(m_bounds.originOffsetted(0, 0, hsizez), hsizex, hsizey, hsizez),
		Bounds<P>(m_bounds.originOffsetted(hsizex, 0, hsizez), hsizex, hsizey, hsizez),
		Bounds<P>(m_bounds.originOffsetted(0, hsizey, hsizez), hsizex, hsizey, hsizez),
		Bounds<P>(m_bounds.originOffsetted(hsizex, hsizey, hsizez), hsizex, hsizey, hsizez)
	};

	for (uint8_t i = 0; i < 8; ++i) {
		if (!m_childs[i] && sub[i].contains(p))
			m_childs[i].reset(new Octree<T,P,CAPACITY>(sub[i]));
		if (m_childs[i] && m_childs[i]->insert(p, value))
			return true;
	}

	return false;
}

template<typename T, typename P, size_t CAPACITY>
bool Octree<T, P, CAPACITY>::isDivided() const noexcept
{
	for (uint8_t i = 0; i < 8; ++i) {
		if (m_childs[i])
			return true;
	}
	return false;
}

template<typename T, typename P, size_t CAPACITY>
size_t Octree<T, P, CAPACITY>::countInSphere(const std::array<P, 3> &pos, P radius) const
{
	size_t counter = 0;

	const auto r2 = radius * radius;
	auto contains = [pos,r2](const std::array<P, 3> &p) {
		const auto dist2 = (pos[0]-p[0])*(pos[0]-p[0]) + (pos[1]-p[1])*(pos[1]-p[1]) + (pos[2]-p[2])*(pos[2]-p[2]);
		return dist2 <= r2;
	};

	// check if sphere intersects bounds
	std::array<P, 3> closest = {
		std::clamp(pos[0], m_bounds.minX, m_bounds.maxX),
		std::clamp(pos[1], m_bounds.minY, m_bounds.maxY),
		std::clamp(pos[2], m_bounds.minZ, m_bounds.maxZ)
	};
	if (!contains(closest))
		return counter;

	// check points
	for (size_t i = 0; i < m_size; ++i) {
		if (contains(m_data[i].first))
			++counter;
	}
	for (uint8_t i = 0; i < 8; ++i) {
		if (m_childs[i])
			counter += m_childs[i]->countInSphere(pos, radius);
	}
	return counter;
}

template<typename T, typename P, size_t CAPACITY>
size_t Octree<T, P, CAPACITY>::countInBox(const Bounds<P> &bound) const
{
	size_t counter = 0;
	if (!m_bounds.intersects(bound))
		return counter;
	for (size_t i = 0; i < m_size; ++i) {
		if (bound.contains(m_data[i].first))
			++counter;
	}
	for (uint8_t i = 0; i < 8; ++i) {
		if (m_childs[i])
			counter += m_childs[i]->countInBox(bound);
	}
	return counter;
}

#endif // OCTREE_H

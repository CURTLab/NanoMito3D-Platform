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
#ifndef BOUNDS_H
#define BOUNDS_H

#include <array>
#include <functional>

template <typename T>
class Bounds
{
public:
	inline constexpr Bounds() : minX(0), maxX(0), minY(0), maxY(0), minZ(0), maxZ(0) {}

	inline constexpr Bounds(T minX, T maxX, T minY, T maxY, T minZ, T maxZ)
		: minX(minX), maxX(maxX), minY(minY), maxY(maxY), minZ(minZ), maxZ(maxZ)
	{}

	inline constexpr Bounds(const std::array<T,3> &pos, T width, T height, T depth)
		: minX(pos[0]), maxX(pos[0] + width)
		, minY(pos[1]), maxY(pos[1] + height)
		, minZ(pos[2]), maxZ(pos[2] + depth)
	{}

	// check if point is contained within the bounds
	inline constexpr bool contains(T x, T y, T z) const
	{
		return x >= minX && x <= maxX && y >= minY && y <= maxY && z >= minZ && z <= maxZ;
	}
	inline constexpr bool contains(const std::array<T,3> &p) const { return contains(p[0], p[1], p[2]); }

	inline constexpr std::array<T,3> origin() const { return {minX, minY, minZ}; }

	inline constexpr std::array<T,3> originOffsetted(T dx, T dy, T dz) const
	{
		return {minX + dx, minY + dy, minZ + dz};
	}

	inline constexpr T width() const  { return maxX - minX; }
	inline constexpr T height() const { return maxY - minY; }
	inline constexpr T depth() const  { return maxZ - minZ; }

	// check each corner if it intersects
	inline constexpr bool intersects(const Bounds<T> &b) const {
		return contains(b.minX, b.minY, b.minZ) ||
				 contains(b.maxX, b.minY, b.minZ) ||
				 contains(b.minX, b.maxY, b.minZ) ||
				 contains(b.minX, b.minY, b.maxZ) ||
				 contains(b.maxX, b.maxY, b.minZ) ||
				 contains(b.maxX, b.minY, b.maxZ) ||
				 contains(b.minX, b.maxY, b.maxZ) ||
				 contains(b.maxX, b.maxY, b.maxZ);
	}

	inline void forEachVoxel(std::function<void(T x, T y, T z)> func, T stepSize = T(1)) const {
		for (T z = minZ; z <= maxZ; z += stepSize) {
			for (T y = minY; y <= maxY; y += stepSize) {
				for (T x = minX; x <= maxX; x += stepSize)
					func(x, y, z);
			}
		}
	}

	T minX, maxX;
	T minY, maxY;
	T minZ, maxZ;
};

#endif // BOUNDS_H

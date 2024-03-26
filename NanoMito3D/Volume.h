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

#ifndef VOLUME_H
#define VOLUME_H

#include <memory>
#include <array>
#include <string>

class VolumeData;

class Volume final
{
public:
	Volume();
	Volume(const std::array<int,3> dims);
	Volume(const std::array<int,3> dims, const std::array<float,3> voxelSize, std::array<float,3> origin = {0.f});
	Volume(const Volume &other);

	~Volume();

	Volume &operator=(const Volume &other);

	// fill the volume with value
	void fill(uint8_t value);

#ifdef USE_OPENCV
	// load a tiff stack as volume
	static Volume loadTif(const std::string &fileName, std::array<float,3> voxelSize = {1.f,1.f,1.f}, std::array<float,3> origin = {0.f});
	void saveTif(const std::string &fileName) const;
#endif // USE_OPENCV

#ifdef USE_H5
	// load and save volume as h5 file
	static Volume loadH5(const std::string &fileName, std::string name = "volume");
	bool saveH5(const std::string &fileName, std::string name = "volume", bool truncate = true, bool compressed = true) const;
#endif // USE_H5

	// checks if value is in bounds
	// returns value at x, y, z in case of in-bounds
	// returns defaultVal in case of out-bounds
	uint8_t value(int x, int y, int z, uint8_t defaultVal = 0) const;
	inline uint8_t value(std::array<int,3> pos, uint8_t defaultVal = 0) const
	{ return value(pos[0], pos[1], pos[2], defaultVal); }

	// set value at x,y,z if value is in bounds
	void setValue(int x, int y, int z, uint8_t value);
	inline void setValue(std::array<int,3> pos, uint8_t value)
	{ setValue(pos[0], pos[1], pos[2], value); }

	// get value at x, y, z without out of bound checks (assert in debug)
	uint8_t &operator()(int x, int y, int z);
	const uint8_t &operator()(int x, int y, int z) const;

	int width() const noexcept;
	int height() const noexcept;
	int depth() const noexcept;
	std::array<int, 3> size() const noexcept;

	size_t voxels() const noexcept;

	// object space
	const std::array<float,3> &voxelSize() const noexcept;
	const std::array<float,3> &origin() const noexcept;

	// return point to host or device data
	uint8_t *data();
	const uint8_t *constData() const noexcept;

	// helper functions to check if position is inside of the volume
	bool contains(int x, int y, int z) const;

	// count different voxels in two volumes
	size_t countDifferences(const Volume &other) const;

	// map voxel postions to object positions
	std::array<float,3> mapVoxel(int x, int y, int z, bool centerVoxel = false) const;
	// map object positions to voxel postions
	std::array<int,3> invMapVoxel(float x, float y, float z) const;

	// map index to x,y,z voxel position
	std::array<int,3> mapIndex(size_t index) const;

	// calculate histogram (cpu only)
	std::array<uint32_t,256> hist() const;

	operator bool() const;

private:
	std::shared_ptr<VolumeData> d;

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

	inline constexpr size_t idx(int x, int y, int z) const {
		return 1ull * x  + static_cast<size_t>(width) * y + static_cast<size_t>(width) * height * z;
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

#endif // VOLUME_H

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
#include "Volume.h"

class VolumeData
{
public:
	inline VolumeData()
		: dims{0, 0, 0}
		, voxelSize{1.f, 1.f, 1.f}
		, origin{0.f, 0.f, 0.f}
		, voxels(0)
		, data(nullptr)
	{
	}

	inline VolumeData(const int dims[3], const std::array<float,3> &voxelSize, const std::array<float, 3> &origin)
		: dims{dims[0], dims[1], dims[2]}
		, voxelSize(voxelSize)
		, origin(origin)
		, voxels(static_cast<size_t>(dims[0]) * dims[1] * dims[2])
	{
		data.reset(new uint8_t[voxels]);
	}

	std::array<int, 3> dims;
	std::array<float, 3> voxelSize;
	std::array<float, 3> origin;
	size_t voxels;
	std::unique_ptr<uint8_t[]> data;

};

Volume::Volume()
	: d(new VolumeData)
{

}

Volume::Volume(const int dims[], const float voxelSize[], std::array<float, 3> origin)
	: d(new VolumeData(dims, {voxelSize[0], voxelSize[1], voxelSize[2]}, origin))
{

}

Volume::Volume(const int dims[], const std::array<float, 3> voxelSize, std::array<float, 3> origin)
	: d(new VolumeData(dims, voxelSize, origin))
{

}

Volume::Volume(const Volume &other)
	: d(other.d)
{
}

Volume::~Volume()
{
	d = {};
}

Volume &Volume::operator=(const Volume &other)
{
	d = other.d;
	return *this;
}

void Volume::fill(uint8_t value)
{
	std::fill_n(d->data.get(), d->voxels, value);
}

int Volume::width() const
{
	return d->dims[0];
}

int Volume::height() const
{
	return d->dims[1];
}

int Volume::depth() const
{
	return d->dims[2];
}

size_t Volume::voxels() const
{
	return d->voxels;
}

const std::array<float,3> &Volume::voxelSize() const
{
	return d->voxelSize;
}

const std::array<float,3> &Volume::origin() const
{
	return d->origin;
}

uint8_t *Volume::data()
{
	return d->data.get();
}

const uint8_t *Volume::constData() const
{
	return d->data.get();
}

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

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <assert.h>
#include <opencv2/opencv.hpp>

class VolumeData
{
public:
	inline VolumeData()
		: dims{0, 0, 0}
		, voxelSize{1.f, 1.f, 1.f}
		, origin{0.f, 0.f, 0.f}
		, voxels(0)
		, hData(nullptr)
		, dData(nullptr)
	{
	}

	inline VolumeData(const std::array<int,3> &dims, const std::array<float,3> &voxelSize, const std::array<float, 3> &origin)
		: dims(dims)
		, voxelSize(voxelSize)
		, origin(origin)
		, stride{1ull, static_cast<size_t>(dims[0]), static_cast<size_t>(dims[0]) * dims[1]}
		, voxels(static_cast<size_t>(dims[0]) * dims[1] * dims[2])
		, hData(nullptr)
		, dData(nullptr)
	{
		hData = new uint8_t[voxels];
		if (hData == nullptr)
			throw std::runtime_error("Couldn't allocate data for volume!");
	}

	inline ~VolumeData() {
		delete [] hData;
		hData = nullptr;

		cudaFree(dData);
		dData = nullptr;
	}

	inline constexpr bool inBounds(int x, int y, int z) const
	{ return (x >= 0 && y >= 0 && z >= 0 && x < dims[0] && y < dims[1] && z < dims[2]); }

	inline constexpr size_t idx(int x, int y, int z) const
	{ return x * stride[0] + y * stride[1] + z * stride[2]; }

	std::array<int, 3> dims;
	std::array<float, 3> voxelSize;
	std::array<float, 3> origin;
	std::array<size_t, 3> stride;

	size_t voxels;
	// host pointer, main data
	uint8_t* hData;
	// device data for cuda
	uint8_t *dData;

};

Volume::Volume()
	: d(new VolumeData)
{

}

Volume::Volume(const std::array<int, 3> dims)
	: d(new VolumeData(dims, {1.f, 1.f, 1.f}, {0.f, 0.f, 0.f}))
{
}

Volume::Volume(const std::array<int, 3> dims, const std::array<float, 3> voxelSize, std::array<float, 3> origin)
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
	std::fill_n(d->hData, d->voxels, value);
}

Volume Volume::loadTif(const std::string &fileName, std::array<float, 3> voxelSize, std::array<float, 3> origin)
{
	std::vector<cv::Mat> stack;
	if (!cv::imreadmulti(fileName, stack, cv::IMREAD_GRAYSCALE) || stack.empty())
		throw std::runtime_error("Could not load tif stack: " + fileName);
	const int w = stack[0].cols, h = stack[0].rows;
	Volume ret({w, h, (int)stack.size()}, voxelSize, origin);

	auto type = stack[0].type();

	// find min/max for not uint8 tifs
	double min = 0., max = 255.;
	if (type != CV_8UC1) {
		cv::minMaxIdx(stack[0], &min, &max);
		for (size_t i = 1; i < stack.size(); ++i) {
			double tmpMin = 0., tmpMax = 0.;
			cv::minMaxIdx(stack[i], &tmpMin, &tmpMax);
			min = std::min(tmpMin, min);
			max = std::max(tmpMax, max);
		}
	}

	// copy each slice into the volume
	const size_t zStride = static_cast<size_t>(w) * h;
	uint8_t *ptr = ret.data();
	for (size_t i = 0; i < stack.size(); ++i, ptr += zStride) {
		if (type != CV_8UC1) {
			cv::Mat dst(w, h, CV_8UC1, ptr);
			stack[i].convertTo(dst, CV_8UC1, 255.0/(max-min), -min);
			assert(dst.ptr() == ptr);
		} else {
			assert(stack[i].isContinuous());
			std::copy_n(stack[i].ptr(), zStride, ptr);
		}
	}
	return ret;
}

uint8_t Volume::value(int x, int y, int z, uint8_t defaultVal) const
{
	return d->inBounds(x, y, z) ? d->hData[d->idx(x, y, z)] : defaultVal;
}

void Volume::setValue(int x, int y, int z, uint8_t value)
{
	if (d->inBounds(x, y, z))
		d->hData[d->idx(x, y, z)] = value;
}

uint8_t &Volume::operator()(int x, int y, int z)
{
	assert(d->inBounds(x, y, z) && "out of bounds");
	return d->hData[d->idx(x, y, z)];
}

const uint8_t &Volume::operator()(int x, int y, int z) const
{
	assert(d->inBounds(x, y, z) && "out of bounds");
	return d->hData[d->idx(x, y, z)];
}

int Volume::width() const noexcept
{
	return d->dims[0];
}

int Volume::height() const noexcept
{
	return d->dims[1];
}

int Volume::depth() const noexcept
{
	return d->dims[2];
}

std::array<int, 3> Volume::size() const noexcept
{
	return d->dims;
}

size_t Volume::voxels() const noexcept
{
	return d->voxels;
}

const std::array<float,3> &Volume::voxelSize() const noexcept
{
	return d->voxelSize;
}

const std::array<float,3> &Volume::origin() const noexcept
{
	return d->origin;
}

bool Volume::copyTo(DeviceType device)
{
	if (device == DeviceType::Device) {
		if (d->dData == nullptr) {
			cudaMalloc((void**)&d->dData, d->voxels);
			GPU::cudaCheckError();
		}
		if (cudaMemcpy(d->dData, d->hData, d->voxels, cudaMemcpyHostToDevice) != cudaSuccess)
			throw std::runtime_error("Could not copy localizations from host to device!");
		return true;
	} else if (device == DeviceType::Host) {
		if (d->dData == nullptr)
			throw std::runtime_error("No device data allocated!");
		if (cudaMemcpy(d->hData, d->dData, d->voxels, cudaMemcpyDeviceToHost) != cudaSuccess)
			throw std::runtime_error("Could not copy localizations from device to host!");
		return true;
	}
	return false;
}

uint8_t *Volume::alloc(DeviceType device)
{
	if (device == DeviceType::Device) {
		if (d->dData == nullptr) {
			if (cudaMalloc((void**)&d->dData, d->voxels) != cudaSuccess)
				throw std::runtime_error("Could allocate device memory!");
		}
		return d->dData;
	} else if (device == DeviceType::Host) {
		return d->hData;
	}
	return nullptr;
}

uint8_t *Volume::data(DeviceType device)
{
	if ((device == DeviceType::Device) && (d->dData == nullptr))
		throw std::runtime_error("No device data allocated!");
	return device == DeviceType::Device ? d->dData : d->hData;
}

const uint8_t *Volume::constData(DeviceType device) const noexcept
{
	return device == DeviceType::Device ? d->dData : d->hData;
}

bool Volume::contains(int x, int y, int z) const
{
	return d->inBounds(x, y, z);
}

size_t Volume::countDifferences(const Volume &other) const
{
	if (other.d->voxels != d->voxels)
		return std::max(other.d->voxels, d->voxels);

	size_t diff = 0;
	for (size_t i = 0; i < d->voxels; ++i)
		diff += (d->hData[i] != other.d->hData[i]);
	return diff;
}

std::array<float, 3> Volume::mapVoxel(int x, int y, int z, bool centerVoxel) const
{
	return std::array<float, 3>{x * d->voxelSize[0] + d->origin[0] + (centerVoxel ? 0.5f * d->voxelSize[0] : 0.f),
										 y * d->voxelSize[1] + d->origin[1] + (centerVoxel ? 0.5f * d->voxelSize[1] : 0.f),
										 z * d->voxelSize[2] + d->origin[2] + (centerVoxel ? 0.5f * d->voxelSize[2] : 0.f)
	};
}

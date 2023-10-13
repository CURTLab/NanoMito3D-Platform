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

#ifdef USE_H5
#include <hdf5.h>
#endif // USE_H5

#include <stdexcept>
#include <assert.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

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

	inline VolumeData(const std::array<int,3> &dims, const std::array<float,3> &voxelSize, const std::array<float, 3> &origin)
		: dims(dims)
		, voxelSize(voxelSize)
		, origin(origin)
		, stride{1ull, static_cast<size_t>(dims[0]), static_cast<size_t>(dims[0]) * dims[1]}
		, voxels(static_cast<size_t>(dims[0]) * dims[1] * dims[2])
		, data(nullptr)
	{
		data = new uint8_t[voxels];
		if (data == nullptr)
			throw std::runtime_error("Couldn't allocate data for volume!");
	}

	inline ~VolumeData() {
		delete [] data;
		data = nullptr;
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
	uint8_t* data;

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
	std::fill_n(d->data, d->voxels, value);
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

void Volume::saveTif(const std::string &fileName) const
{
	std::vector<cv::Mat> stack(d->dims[2]);
	for (int z = 0; z < d->dims[2]; ++z)
		stack[z] = cv::Mat(d->dims[1], d->dims[0], CV_8U, d->data + d->idx(0, 0, z));

	if (!cv::imwritemulti(fileName, stack))
		throw std::runtime_error("Could not save tif stack at " + fileName);
}

#ifdef USE_H5
Volume Volume::loadH5(const std::string &fileName, std::string name)
{
	hid_t file = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

	if (!H5Lexists(file, name.c_str(), H5P_DEFAULT))
		return {};

	hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
	hid_t vSpaceId = H5Dget_space(dataset);
	int rank = H5Sget_simple_extent_ndims(vSpaceId);
	std::vector<hsize_t> dims(rank);
	H5Sget_simple_extent_dims(vSpaceId, dims.data(), nullptr);

	std::array<float,3> voxelSize, origin;

	hid_t aid1 = H5Aopen(dataset, "voxelSize", H5P_DEFAULT);
	H5Aread(aid1, H5T_NATIVE_FLOAT, voxelSize.data());
	H5Aclose(aid1);

	hid_t aid2 = H5Aopen(dataset, "origin", H5P_DEFAULT);
	H5Aread(aid2, H5T_NATIVE_FLOAT, origin.data());
	H5Aclose(aid2);

	Volume ret({(int)dims[2], (int)dims[1], (int)dims[0]}, voxelSize, origin);
	H5Dread(dataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, ret.d->data);

	H5Sclose(vSpaceId);
	H5Dclose(dataset);
	H5Fclose(file);

	return ret;
}

bool Volume::saveH5(const std::string &fileName, std::string name, bool truncate, bool compressed) const
{
	if (std::filesystem::exists(fileName) && truncate)
		std::filesystem::remove(fileName);

	hid_t file;
	if (!std::filesystem::exists(fileName))
		file =  H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	else
		file =  H5Fopen(fileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
	if (file < 0)
		return false;

	hid_t plist = H5P_DEFAULT;
	if (compressed) {
		plist = H5Pcreate(H5P_DATASET_CREATE);
		hsize_t cdims[3] = {(hsize_t)d->dims[2], (hsize_t)d->dims[1], 1};
		H5Pset_chunk(plist, 3, cdims);
		H5Pset_deflate(plist, 6);
	}

	hid_t fid = H5Screate(H5S_SIMPLE);
	hsize_t fdim[3] = {(hsize_t)d->dims[2], (hsize_t)d->dims[1], (hsize_t)d->dims[0]};
	H5Sset_extent_simple(fid, 3, fdim, NULL);

	hid_t dataset = H5Dcreate2(file, name.c_str(), H5T_NATIVE_UINT8, fid, H5P_DEFAULT, plist, H5P_DEFAULT);
	if (dataset < 0) {
		H5Sclose(fid);
		H5Fclose(file);
	}

	int err = H5Dwrite(dataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, d->data);

	hsize_t adim[1] = {3};

	// create attribute float voxelSize[3]
	hid_t aid1 = H5Screate(H5S_SIMPLE);
	H5Sset_extent_simple(aid1, 1, adim, NULL);
	hid_t attr1 = H5Acreate2(dataset, "voxelSize", H5T_NATIVE_FLOAT, aid1, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attr1, H5T_NATIVE_FLOAT, d->voxelSize.data());

	// create attribute float origin[3]
	hid_t aid2 = H5Screate(H5S_SIMPLE);
	H5Sset_extent_simple(aid2, 1, adim, NULL);
	hid_t attr2 = H5Acreate2(dataset, "origin", H5T_NATIVE_FLOAT, aid1, H5P_DEFAULT, H5P_DEFAULT);
	H5Awrite(attr2, H5T_NATIVE_FLOAT, d->origin.data());

	H5Sclose(aid2);
	H5Sclose(aid1);
	H5Sclose(fid);

	if (compressed)
		H5Pclose(plist);

	H5Dclose(dataset);
	H5Fclose(file);

	return err >= 0;
}
#endif // USE_H5

uint8_t Volume::value(int x, int y, int z, uint8_t defaultVal) const
{
	return d->inBounds(x, y, z) ? d->data[d->idx(x, y, z)] : defaultVal;
}

void Volume::setValue(int x, int y, int z, uint8_t value)
{
	if (d->inBounds(x, y, z))
		d->data[d->idx(x, y, z)] = value;
}

uint8_t &Volume::operator()(int x, int y, int z)
{
	assert(d->inBounds(x, y, z) && "out of bounds");
	return d->data[d->idx(x, y, z)];
}

const uint8_t &Volume::operator()(int x, int y, int z) const
{
	assert(d->inBounds(x, y, z) && "out of bounds");
	return d->data[d->idx(x, y, z)];
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

uint8_t *Volume::data()
{
	return d->data;
}

const uint8_t *Volume::constData() const noexcept
{
	return d->data;
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
		diff += (d->data[i] != other.d->data[i]);
	return diff;
}

std::array<float, 3> Volume::mapVoxel(int x, int y, int z, bool centerVoxel) const
{
	return std::array<float, 3>{x * d->voxelSize[0] + d->origin[0] + (centerVoxel ? 0.5f * d->voxelSize[0] : 0.f),
										 y * d->voxelSize[1] + d->origin[1] + (centerVoxel ? 0.5f * d->voxelSize[1] : 0.f),
										 z * d->voxelSize[2] + d->origin[2] + (centerVoxel ? 0.5f * d->voxelSize[2] : 0.f)
	};
}

std::array<int, 3> Volume::invMapVoxel(float x, float y, float z) const
{
	return {static_cast<int>((x - d->origin[0])/d->voxelSize[0]),
			  static_cast<int>((y - d->origin[1])/d->voxelSize[1]),
				static_cast<int>((z - d->origin[2])/d->voxelSize[2])};
}

std::array<int, 3> Volume::mapIndex(size_t index) const
{
	const int z = static_cast<int>(index / d->stride[2]);
	index -= d->stride[2] * z;
	const int y = static_cast<int>(index / d->stride[1]);
	index -= d->stride[1] * y;
	const int x = static_cast<int>(index / d->stride[0]);
	return {x, y, z};
}

std::array<uint32_t, 256> Volume::hist() const
{
	std::array<uint32_t, 256> hist;
	std::fill(hist.begin(), hist.end(), uint32_t(0));

	for (size_t i = 0; i < d->voxels; ++i)
		hist[d->data[i]]++;
	return hist;
}

Volume::operator bool() const
{
	return d->voxels > 0;
}

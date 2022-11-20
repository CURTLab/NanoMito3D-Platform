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
#ifndef VOLUME_H
#define VOLUME_H

#include <memory>
#include <array>

#include "Device.h"

class VolumeData;

class Volume final
{
public:
	Volume();
	Volume(const int dims[3], const float voxelSize[3], std::array<float,3> origin = {0.f});
	Volume(const std::array<int,3> dims, const std::array<float,3> voxelSize, std::array<float,3> origin = {0.f});
	Volume(const Volume &other);
	~Volume();

	Volume &operator=(const Volume &other);

	void fill(uint8_t value);

	int width() const noexcept;
	int height() const noexcept;
	int depth() const noexcept;
	std::array<int, 3> size() const noexcept;

	size_t voxels() const noexcept;

	const std::array<float,3> &voxelSize() const noexcept;
	const std::array<float,3> &origin() const noexcept;


	bool copyTo(DeviceType device);

	uint8_t *alloc(DeviceType device);

	uint8_t *data(DeviceType device = DeviceType::Host);
	const uint8_t *constData(DeviceType device = DeviceType::Host) const noexcept;

private:
	std::shared_ptr<VolumeData> d;

};

#endif // VOLUME_H

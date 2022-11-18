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

class VolumeData;

class Volume final
{
public:
	Volume();
	Volume(const int dims[3], const float voxelSize[3], std::array<float,3> origin = {0.f});
	Volume(const Volume &other);
	~Volume();

	Volume &operator=(const Volume &other);

	void fill(uint8_t value);

	int width() const;
	int height() const;
	int depth() const;

	size_t voxels() const;

	const std::array<float,3> &voxelSize() const;
	const std::array<float,3> &origin() const;

	uint8_t *data();
	const uint8_t *constData() const;


private:
	std::shared_ptr<VolumeData> d;

};

#endif // VOLUME_H

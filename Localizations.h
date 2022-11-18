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
#ifndef LOCALIZATIONS_H
#define LOCALIZATIONS_H

#include <vector>
#include <string>
#include <array>

#include "Bounds.h"

struct Localization
{
	float x, y, z;
	uint32_t frame;
	float PAx, PAy, PAz;
	int32_t channel;

	inline constexpr std::array<float,3> position() const { return {x, y, z}; };
};

class Localizations : public std::vector<Localization>
{
public:
	Localizations();

	void load(const std::string &fileName);

	inline constexpr float width() const { return m_width * m_pixelSize; }
	inline constexpr float height() const { return m_height * m_pixelSize; }
	inline constexpr float depth() const { return m_maxZ - m_minZ; }
	inline constexpr float pixelSize() const { return m_pixelSize; }
	inline constexpr float minZ() const { return m_minZ; }
	inline constexpr float maxZ() const { return m_maxZ; }

	inline constexpr Bounds<float> bounds() const {
		return {0.f, m_width * m_pixelSize,  0.f, m_height * m_pixelSize, m_minZ, m_maxZ };
	}

private:
	int m_width;
	int m_height;
	float m_pixelSize;
	float m_minZ;
	float m_maxZ;

};

#endif // LOCALIZATIONS_H

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

struct Localization
{
	uint32_t frame;
	int32_t channel;
	float x, y, z;
	float PAx, PAy, PAz;
};

class Localizations : public std::vector<Localization>
{
public:
	Localizations();

	void load(const std::string &fileName);

private:

};

#endif // LOCALIZATIONS_H

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

#ifndef RENDERING_H
#define RENDERING_H

#include "Localizations.h"
#include "Volume.h"

namespace Rendering {

Volume render_gpu(Localizations &locs, std::array<float,3> voxelSize, int windowSize);
Volume render_cpu(Localizations &locs, std::array<float,3> voxelSize, int windowSize);

void renderHistgram3D_gpu(const Localizations &locs, uint32_t *output, std::array<int,3> size, std::array<float,3> voxelSize, std::array<float,3> origin);
void renderHistgram3D_cpu(const Localizations &locs, uint32_t *output, std::array<int,3> size, std::array<float,3> voxelSize, std::array<float,3> origin);

}

#endif // RENDERING_H

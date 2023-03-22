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
#ifndef DEVICE_H
#define DEVICE_H

#include <cuda_device_runtime_api.h>

#include "Types.h"

#ifdef __CUDACC__
#define HOST_DEV __device__ __host__
#else
#define HOST_DEV
#endif

struct vec3 {
	float x, y, z;
};

namespace GPU {

void initGPU();
bool isGPUAvailable();
void cudaCheckError();

}

#endif // DEVICE_H

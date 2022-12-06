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

#include "../Device.h"

#include <stdexcept>

namespace GPU {

static bool IS_DEVICE_AVAILABLE = false;
static bool INITIALIZED = false;

}

void GPU::initGPU() {
	if (INITIALIZED)
		return;
	double hTmp = 1;
	double *dTmp;
	cudaMalloc(&dTmp, sizeof(double));
	cudaMemcpy(dTmp, &hTmp, sizeof(double), cudaMemcpyHostToDevice);
	cudaFree(dTmp);

	int nDevices = 0;
	cudaGetDeviceCount(&nDevices);
	IS_DEVICE_AVAILABLE = nDevices > 0;
}

bool GPU::isGPUAvailable()
{
	initGPU();
	return IS_DEVICE_AVAILABLE;
}

void GPU::cudaCheckError()
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		throw std::runtime_error(std::string("Cuda error ") + cudaGetErrorName(err) + ": " + cudaGetErrorString(err));
}

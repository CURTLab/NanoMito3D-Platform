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

#include "Rendering.h"
#include "Device.h"

#include <thrust/fill.h>
#include <thrust/device_vector.h>

#include <algorithm>

#define BLOCK_SIZE 1024

// original from https://forums.developer.nvidia.com/t/atomicmin-on-char-is-there-a-way-to-compare-char-to-in-to-use-atomicmin/22246/2
__device__ uint8_t atomicMaxU8(uint8_t* address, uint8_t val)
{
	unsigned int *base_address = (unsigned int *)((size_t)address & ~3);
	unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
	unsigned int sel = selectors[(size_t)address & 3];
	unsigned int old, assumed, max_, new_;

	old = *base_address;
	do {
		assumed = old;
		max_ = max(val, (uint8_t)__byte_perm(old, 0, ((size_t)address & 3)));
		new_ = __byte_perm(old, max_, sel);
		old = atomicCAS(base_address, assumed, new_);
	} while (assumed != old);

	return old;
}

__global__ void drawPSF_kernel(uint8_t *dVolume, const Localization *dLocs, uint32_t n, const int3 volumeDims, const float3 voxelSize, int windowSize)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	const Localization &l = dLocs[i];

	const int ix = static_cast<int>(std::round((l.x / voxelSize.x)));
	const int iy = static_cast<int>(std::round((l.y / voxelSize.y)));
	const int iz = static_cast<int>(std::round((l.z / voxelSize.z)));

	const size_t strideZ = volumeDims.x * volumeDims.y;

	const int w = windowSize/2;
	for (int z = -w; z <= w; ++z) {
		for (int y = -w; y <= w; ++y) {
			for (int x = -w; x <= w; ++x) {
				if ((ix + x < 0) || (iy + y < 0) || (iz + z < 0) ||
					 (ix + x >= volumeDims.x) || (iy + y >= volumeDims.y) || (iz + z >= volumeDims.z))
					continue;
				const float tx = ((ix + x) * voxelSize.x - l.x) / l.PAx;
				const float ty = ((iy + y) * voxelSize.y - l.y) / l.PAy;
				const float tz = ((iz + z) * voxelSize.z - l.z) / l.PAz;
				const float e = min((255.f/windowSize)*expf(-0.5f * tx * tx -0.5f * ty * ty -0.5f * tz * tz), 255.f);
				const uint8_t val = static_cast<uint8_t>(e);
				const size_t addr = (ix + x) + volumeDims.x * (iy + y) + strideZ * (iz + z);
				// safely write max to volume
				atomicMaxU8(dVolume + addr, val);
			}
		}
	}
}

Volume Rendering::render_gpu(Localizations &locs, std::array<float,3> voxelSize, int windowSize)
{
	int3 dims;
	dims.x = static_cast<int>(std::ceilf(locs.width()  / voxelSize[0]));
	dims.y = static_cast<int>(std::ceilf(locs.height() / voxelSize[1]));
	dims.z = static_cast<int>(std::ceilf(locs.depth()  / voxelSize[2]));

	Volume hVolume({dims.x, dims.y, dims.z}, voxelSize, {0.f, 0.f, locs.minZ()});

	thrust::device_vector<uint8_t> dVolume(hVolume.voxels());
	thrust::fill_n(dVolume.begin(), dVolume.size(), 0);

	// copy all localizations
	const uint32_t n = static_cast<uint32_t>(locs.size());
	const size_t bytes = locs.size() * sizeof(Localization);

	Localization *dLocs = nullptr;
	cudaMalloc(&dLocs, bytes);
	cudaMemcpy(dLocs, locs.data(), bytes, cudaMemcpyHostToDevice);

	GPU::cudaCheckError();
	cudaDeviceSynchronize();

	const dim3 block(BLOCK_SIZE);
	const dim3 grid((n + block.x - 1)/block.x);
	drawPSF_kernel<<<grid,block>>>(thrust::raw_pointer_cast(dVolume.data()),
											 dLocs, n,
											 dims,
											 make_float3(voxelSize[0], voxelSize[1], voxelSize[2]),
											 windowSize
											 );

	GPU::cudaCheckError();
	cudaDeviceSynchronize();

	cudaFree(dLocs);

	cudaMemcpy(hVolume.data(), thrust::raw_pointer_cast(dVolume.data()), hVolume.voxels(), cudaMemcpyDeviceToHost);

	GPU::cudaCheckError();
	cudaDeviceSynchronize();

	return hVolume;
}

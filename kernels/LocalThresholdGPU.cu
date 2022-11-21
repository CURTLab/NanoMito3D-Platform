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

#include "LocalThreshold.h"
#include "Device.h"

#include <stdexcept>
#include <string>
#include <assert.h>

#define BLOCK_SIZE 512
#define BLOCK_SIZE2 16
#define BLOCK_SIZE3 8

HOST_DEV uint8_t LocalThreshold::otsuThreshold(const uint16_t hist[256], int numPixels)
{
	int i;
	uint8_t threshold = 0;
	const float term = 1.f / numPixels;

	float total_mean = 0.f;
	for (i = 0; i < 256; ++i)
		total_mean += i * term * hist[i];

	float max_bcv = 0.f;
	float cnh = 0.f;
	float mean = 0.f;
	for (i = 0; i < 256; ++i) {
		const float norm = term * hist[i];
		cnh += norm;
		mean += i * norm;

		float p = max(1E-7f, cnh);

		float bcv = total_mean * cnh - mean;
		bcv *= bcv / (p * (1.f - p));

		if (max_bcv < bcv) {
			max_bcv = bcv;
			threshold = i;
		}
	}
	return threshold;
}

HOST_DEV uint8_t LocalThreshold::isoDataThreshold(const uint16_t hist[256], int numPixels)
{
	int i;

	float toth = 0.f, h = 0.f;
	float totl = 0.f, l = 0.f;
	for (i = 1; i < 256; ++i) {
		toth += static_cast<float>(hist[i]);
		h += i * static_cast<float>(hist[i]);
	}

	uint8_t threshold = 255;
	for (i = 1; i < 255; ++i) {
		totl += hist[i];
		l += static_cast<float>(hist[i]) * i;
		toth -= hist[i+1];
		h -= static_cast<float>(hist[i+1]) * (i+1);
		if (totl > 0 && toth > 0 && i == (uint8_t)(0.5 * (l/totl + h/toth))) {
			threshold = i;
		}
	}
	return threshold;
}

#define VOLUMEFILTER_MAXWEIGHTS VOLUMEFILTER_MAXSIZE*VOLUMEFILTER_MAXSIZE*VOLUMEFILTER_MAXSIZE
__constant__ int32_t c_filterOffsets[VOLUMEFILTER_MAXWEIGHTS];

__global__ void local_threshold_kernel(LocalThreshold::Method method, const uint8_t *d_input, uint8_t *d_output, int width, int height, int depth, int64_t voxels, int radius)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	const int fsize = radius * radius * radius;
	const int idx = z * width * height + y * width + x;

	if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth)
		return;

	int i;

	uint16_t hist[256];
	for (i = 0; i < 256; ++i) hist[i] = 0;

	int num_pixels = 0;
	const int32_t *f = c_filterOffsets;
	for (i = 0; i < fsize; ++i) {
		const int idx2 = idx + (*f++);
		if (idx2 >= 0 && idx2 < voxels) {
			hist[d_input[idx2]]++;
			num_pixels++;
		}
	}

	if (method == LocalThreshold::Otsu)
		d_output[idx] = (d_input[idx] >= LocalThreshold::otsuThreshold(hist, num_pixels) ? 255 : 0);
	else if (method == LocalThreshold::IsoData)
		d_output[idx] = (d_input[idx] >= LocalThreshold::isoDataThreshold(hist, num_pixels) ? 255 : 0);
	else
		d_output[idx] = 0;
}

void LocalThreshold::localThrehsold_gpu(Method method, Volume input, Volume output, int windowSize)
{
	// check output dims
	if ((input.width() != output.width()) ||
		 (input.height() != output.height()) ||
		 (input.depth() != output.depth())) {
		// realloc output if dims are different
		output = Volume(input.size(), input.voxelSize(), input.origin());
	}
	uint8_t *d_output = nullptr;
	cudaMalloc(&d_output, input.voxels());

	if (input.constData(DeviceType::Device) == nullptr)
		input.copyTo(DeviceType::Device);
	const uint8_t *d_input = input.constData(DeviceType::Device);

	if (windowSize > VOLUMEFILTER_MAXSIZE)
		throw std::runtime_error("Max filter size is " + std::to_string(VOLUMEFILTER_MAXSIZE) + "!");

	int32_t *filterOffsets = new int32_t[windowSize * windowSize * windowSize];
	int32_t *idx = filterOffsets;
	const int r = windowSize/2;
	for (int k = -r; k <= r; ++k) {
		for (int j = -r; j <= r; ++j) {
			for (int i = -r; i <= r; ++i)
				*idx++ = i + j * static_cast<size_t>(input.width()) + k * static_cast<size_t>(input.width()) * input.height();
		}
	}
	cudaMemcpyToSymbol(c_filterOffsets, filterOffsets, windowSize * windowSize * windowSize * sizeof(int32_t));
	delete [] filterOffsets;

	int64_t voxels = static_cast<int64_t>(input.voxels());
	const size_t zStride = static_cast<size_t>(input.width()) * input.height();

#if 0
	const int dz = 8;

	const dim3 block(BLOCK_SIZE3, BLOCK_SIZE3, BLOCK_SIZE3);
	const dim3 grid((input.width() + block.x - 1)/block.x,
						 (input.height() + block.y - 1)/block.y,
						 (dz + block.z - 1)/block.z);

	for (int i = 0; i < input.depth(); i += dz, voxels -= dz * zStride) {
		const auto z = std::min(input.depth() - i, dz);
		assert(z > 0);
		local_threshold_kernel<<<grid,block>>>(method, d_input + i * zStride, d_output + i * zStride, input.width(), input.height(), z, voxels, windowSize);
		GPU::cudaCheckError();
		cudaDeviceSynchronize();
	}
#else
	// 2D kernel is faster then 3D
	const dim3 block(BLOCK_SIZE2, BLOCK_SIZE2);
	const dim3 grid((input.width() + block.x - 1)/block.x,
						 (input.height() + block.y - 1)/block.y);

	for (int i = 0; i < input.depth(); ++i, voxels -= zStride) {
		local_threshold_kernel<<<grid,block>>>(method, d_input + i * zStride, d_output + i * zStride, input.width(), input.height(), 1, voxels, windowSize);
		GPU::cudaCheckError();
		cudaDeviceSynchronize();
	}
#endif

	cudaMemcpy(output.data(DeviceType::Host), d_output, input.voxels(), cudaMemcpyDeviceToHost);
	GPU::cudaCheckError();
}

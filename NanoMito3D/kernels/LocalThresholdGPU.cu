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

#define BLOCK_SIZE 256
#define BLOCK_SIZE2 16
#define BLOCK_SIZE3 8

//using idx_t = int64_t;
using idx_t = int;

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

__global__ void local_threshold_kernel2D(LocalThreshold::Method method, const uint8_t *d_input, uint8_t *d_output, int z, int width, int height, int depth, int64_t voxels, int windowSize)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const int64_t fsize = static_cast<int64_t>(windowSize) * windowSize * windowSize;
	const int64_t idx = static_cast<int64_t>(z) * width * height + static_cast<int64_t>(y) * width + static_cast<int64_t>(x);

	if (x < 0 || x >= width || y < 0 || y >= height || z < 0 || z >= depth)
		return;

	uint16_t hist[256];
	for (int i = 0; i < 256; ++i) hist[i] = 0;

	int num_pixels = fsize;
	const idx_t *f = c_filterOffsets;
	for (idx_t i = 0; i < fsize; ++i, f++) {
		const idx_t idx2 = idx + *f;
		if (idx2 >= 0 && idx2 < voxels)
			hist[d_input[idx2]]++;
		else
			hist[0]++;
	}

	if (method == LocalThreshold::Otsu)
		d_output[idx] = (d_input[idx] >= LocalThreshold::otsuThreshold(hist, num_pixels) ? 255 : 0);
	else if (method == LocalThreshold::IsoData)
		d_output[idx] = (d_input[idx] >= LocalThreshold::isoDataThreshold(hist, num_pixels) ? 255 : 0);
	else
		d_output[idx] = 0;
}

__global__ void local_threshold_kernel1D(LocalThreshold::Method method, const uint8_t *d_input, uint8_t *d_output, idx_t voxels, idx_t nFilter)
{
	const int block = blockIdx.x * blockDim.x;
	const int idx = block + threadIdx.x;

	if (idx >= voxels)
		return;

	uint16_t hist[256];
	for (int i = 0; i < 256; ++i) hist[i] = 0;

	for (idx_t i = 0; i < nFilter; ++i) {
		// index for 3D window
		const idx_t idx2 = idx + c_filterOffsets[i];
		const idx_t histIdx = (idx2 >= 0 && idx2 < voxels) ? d_input[idx2] : 0;
		hist[histIdx]++;
	}

	if (hist[0] == nFilter)
		d_output[idx] = 0;
	else if (method == LocalThreshold::Otsu)
		d_output[idx] = (d_input[idx] >= LocalThreshold::otsuThreshold(hist, nFilter) ? 255 : 0);
	else if (method == LocalThreshold::IsoData)
		d_output[idx] = (d_input[idx] >= LocalThreshold::isoDataThreshold(hist, nFilter) ? 255 : 0);
	else
		d_output[idx] = 0;
}

void LocalThreshold::localThrehsold_gpu(Method method, const Volume &input, Volume &output, int windowSize)
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

	uint8_t *d_input = nullptr;
	cudaMalloc(&d_input, input.voxels());
	cudaMemcpy(d_input, input.constData(), input.voxels(), cudaMemcpyHostToDevice);

	if (windowSize > VOLUMEFILTER_MAXSIZE)
		throw std::runtime_error("Max filter size is " + std::to_string(VOLUMEFILTER_MAXSIZE) + "!");

	idx_t nFilter = static_cast<idx_t>(windowSize) * windowSize * windowSize;
	idx_t *filterOffsets = new idx_t[nFilter];
	idx_t *idx = filterOffsets;
	const idx_t r = windowSize/2;
	for (idx_t k = -r; k <= r; ++k) {
		for (idx_t j = -r; j <= r; ++j) {
			for (idx_t i = -r; i <= r; ++i)
				*idx++ = i + j * input.width() + k * input.width() * input.height();
		}
	}
	cudaMemcpyToSymbol(c_filterOffsets, filterOffsets, windowSize * windowSize * windowSize * sizeof(idx_t));
	delete [] filterOffsets;

	idx_t voxels = static_cast<idx_t>(input.voxels());
#if 1
	// 1D kernel
	int batchSize = BLOCK_SIZE * 1024;
	const dim3 block(BLOCK_SIZE);
	const dim3 grid((static_cast<uint32_t>(batchSize) + block.x - 1)/block.x);
	for (idx_t i = 0; i < voxels; i += batchSize)
		local_threshold_kernel1D<<<grid,block>>>(method, d_input + i, d_output + i, voxels - i, nFilter);
#else
	const size_t zStride = static_cast<size_t>(input.width()) * input.height();
	// 2D kernel
	const dim3 block(BLOCK_SIZE2, BLOCK_SIZE2);
	const dim3 grid((input.width() + block.x - 1)/block.x,
					(input.height() + block.y - 1)/block.y);

	for (int i = 0; i < input.depth(); ++i)
		local_threshold_kernel2D<<<grid,block>>>(method, d_input, d_output, i, input.width(), input.height(), input.depth(), voxels, windowSize);
#endif

	cudaMemcpy(output.data(), d_output, input.voxels(), cudaMemcpyDeviceToHost);
	GPU::cudaCheckError();

	cudaFree(c_filterOffsets);
	cudaFree(d_output);
	cudaFree(d_input);
}

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

#include "GaussianFilter.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "Device.h"

#define BLOCK_SIZE 16
#define BLOCK_SIZE3 8

#define M_PIF 3.141592653589793238462643383279502884e+00f

__global__ void generateGaussian_kernel(float *d_kernel, int size, float sigma)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= size || y >= size || z >= size)
		return;
	float g = 0.5f/(sigma*sigma);
	float f = 1.f/(2.f*M_PIF*sigma*sigma);
	float i = x - size/2;
	float j = y - size/2;
	float k = z - size/2;
	d_kernel[x + y * size] = f * exp(-g*i*i-g*j*j-g*k*k);
}

__global__ void filter3D_kernel(const uint8_t *d_input, uint8_t *d_output, int width, int height, int depth, int size, const float *d_kernel)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int r = size/2;
	size_t idx;

	const size_t stride[3] = {1ull, static_cast<size_t>(width), static_cast<size_t>(width) * height};

	if (x >= width || y >= height || z >= depth)
		return;

	const float *dk = d_kernel;
	float sum = 0.f;
	float val = 0.f;
	for (int k = -r; k <= r; ++k) {
		int zi = z + k;
		for (int j = -r; j <= r; ++j) {
			int yi = y + j;
			for (int i = -r; i <= r; ++i, ++dk) {
				int xi = x + i;
				if (xi >= 0 && xi < width && yi >= 0 && yi < height && zi >= 0 && zi < depth) {
					idx = stride[0] * xi + stride[1] * yi + stride[2] * zi;
					val += d_input[idx] * (*dk);
					sum += *dk;
				}
			}
		}
	}

	idx = stride[0] * x + stride[1] * y + stride[2] * z;
	d_output[idx] = val / sum;
}

void GaussianFilter::gaussianFilter_gpu(const uint8_t *input, uint8_t *output, int width, int height, int depth, int size, float sigma)
{
	const int bytes = width * height * depth;
	const int kbytes = size * size * size * sizeof(float);

	uint8_t *d_input, *d_output;
	float *d_kernel;
	cudaMalloc(&d_input, bytes);
	cudaMalloc(&d_output, bytes);
	cudaMalloc(&d_kernel, kbytes);

	cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);

	const dim3 block(BLOCK_SIZE3,BLOCK_SIZE3,BLOCK_SIZE3);

	const dim3 kgrid((size + block.x - 1)/block.x,
						  (size + block.y - 1)/block.y,
						  (size + block.z - 1)/block.z);
	generateGaussian_kernel<<<kgrid,block>>>(d_kernel, size, sigma);
	GPU::cudaCheckError();
	cudaDeviceSynchronize();

	const dim3 grid((width + block.x - 1)/block.x,
						 (height + block.y - 1)/block.y,
						 (depth + block.z - 1)/block.z);
	filter3D_kernel<<<grid,block>>>(d_input, d_output, width, height, depth, size, d_kernel);
	GPU::cudaCheckError();
	cudaDeviceSynchronize();

	cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);
	GPU::cudaCheckError();

	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_kernel);
}

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

#include "GaussianFilter.h"

#include <memory>
#include <cmath>

#define M_PIF 3.141592653589793238462643383279502884e+00f

// powf(sqrtf(2.f*M_PIF),3)
#define M_EXP3N 15.7496099457224f

void generateGaussianKernel(float *d_kernel, int size, std::array<float, 3> sigma)
{
	for (int x = 0; x < size; ++x) {
		for (int y = 0; y < size; ++y) {
			for (int z = 0; z < size; ++z) {
				float f = 1.f / (M_EXP3N * sigma[0] * sigma[1] * sigma[2]);
				float i = (x - size * 0.5f) / sigma[0];
				float j = (y - size * 0.5f) / sigma[1];
				float k = (z - size * 0.5f) / sigma[2];
				d_kernel[x + y * size + z * size * size] = f * std::exp(-0.5f * i * i - 0.5f * j * j - 0.5f * k * k);
			}
		}
	}
}

void filter3D(const uint8_t *d_input, uint8_t *d_output, int width, int height, int depth, int size, const float *d_kernel)
{
	int r = size / 2;

	const size_t stride[3] = {1, static_cast<size_t>(width), static_cast<size_t>(width) * static_cast<size_t>(height)};
	auto idx = [&](int x, int y, int z) -> size_t { return stride[0] * static_cast<size_t>(x) + stride[1] * static_cast<size_t>(y) + stride[2] * static_cast<size_t>(z); };

	for (int x = 0; x < width; ++x) {
		for (int y = 0; y < height; ++y) {
			for (int z = 0; z < depth; ++z) {
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
								val += d_input[idx(xi, yi, zi)] * (*dk);
								sum += *dk;
							}
						}
					}
				}
				d_output[idx(x, y, z)] = static_cast<uint8_t>(std::fmin(std::fmax(0.f, val / sum), 255.f));
			}
		}
	}
}

void GaussianFilter::gaussianFilter_cpu(const uint8_t *input, uint8_t *output, int width, int height, int depth, int size, std::array<float,3> sigma)
{
	std::unique_ptr<float[]> kernel(new float[size * size * size]);
	generateGaussianKernel(kernel.get(), size, sigma);

	filter3D(input, output, width, height, depth, size, kernel.get());
}

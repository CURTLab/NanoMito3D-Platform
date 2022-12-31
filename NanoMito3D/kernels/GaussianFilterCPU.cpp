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

#include <opencv2/opencv.hpp>


void GaussianFilter::gaussianFilter_cpu(const uint8_t *input, uint8_t *output, int width, int height, int depth, int size, float sigma)
{
	cv::Mat kernel = cv::getGaussianKernel(size, sigma, CV_32F);


	const size_t stride[3] = {1ull, static_cast<size_t>(width), static_cast<size_t>(width) * static_cast<size_t>(height)};
	const auto idx = [&](int x, int y, int z) -> size_t { return stride[0] * static_cast<size_t>(x) + stride[1] * static_cast<size_t>(y) + stride[2] * static_cast<size_t>(z); };

	// first pass
	for (int i = 0; i < depth; ++i) {
		cv::Mat in(width, height, CV_8U, (void*)(input + idx(0,0,i)));
		cv::Mat out(width, height, CV_8U, (void*)(output + idx(0,0,i)));
		cv::GaussianBlur(in, out, {size, size}, sigma, sigma);
	}

	// 1d convolution in z directions
	for (int x = 0; x < width; ++x) {
		for (int y = 0; y < height; ++y) {
			for (int z = 0; z < depth; ++z) {
				float sum = 0.f;
				float result = 0.f;
				for (int i = -size/2; i <= size/2; ++i) {
					if ((i + z < 0) || (i + z >= depth))
						continue;
					sum += kernel.at<float>(i + z + size/2);
					result += kernel.at<float>(i + z + size/2) * output[idx(x, y, z + i)];
				}
				output[idx(x, y, z)] = static_cast<uint8_t>(fmin(fmax(0.f, result/sum), 255.f));
			}
		}
	}
}

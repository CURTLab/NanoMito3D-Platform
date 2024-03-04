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

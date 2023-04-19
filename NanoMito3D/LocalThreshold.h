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
#ifndef LOCALTHRESHOLD_H
#define LOCALTHRESHOLD_H

#include "Volume.h"
#include "Device.h"

#include <functional>

#define VOLUMEFILTER_MAXSIZE 17

namespace LocalThreshold
{

enum Method {
	Otsu,
	IsoData
};

HOST_DEV uint8_t otsuThreshold(const uint16_t hist[256], int numPixels);
HOST_DEV uint8_t isoDataThreshold(const uint16_t hist[256], int numPixels);

void localThrehsold_gpu(Method method, const Volume &input, Volume &output, int windowSize);
void localThrehsold_cpu(Method method, const Volume &input, Volume &output, int windowSize, std::function<void(uint32_t, uint32_t)> cb = {});

}

#endif // LOCALTHRESHOLD_H

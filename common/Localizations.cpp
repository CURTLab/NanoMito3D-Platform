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
#include "Localizations.h"

#include <fstream>
#include <sstream>
#include <array>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include "proto/TSFProto.pb.h"

Localizations::Localizations()
	: m_width(0), m_height(0), m_pixelSize(1.f), m_minZ(0.f), m_maxZ(0.f), m_channels(1), m_numFrames(0)
{
	static_assert(sizeof(Localization) == 40);
}

Localizations::~Localizations()
{
}

void Localizations::load(const std::string &fileName, std::function<void (uint32_t, uint32_t, const Localization &)> cb)
{
	// TSF file format: https://github.com/nicost/TSFProto

	std::ifstream stream(fileName, std::ios_base::binary);
	if (!stream) {
		throw std::runtime_error("Could not open file (" + fileName + ")!");
	}

	int32_t magic = 1;
	stream.read((char*)&magic, sizeof(magic));
	if (magic != 0)
		throw std::runtime_error("Magic number is not 0, is this a tsf file?");

	union {
		uint64_t v;
		uint8_t d[8];
	} offset;
	stream.read((char*)&offset.v, sizeof(offset));

	// swap endianness
	std::swap(offset.d[0], offset.d[7]);
	std::swap(offset.d[1], offset.d[6]);
	std::swap(offset.d[2], offset.d[5]);
	std::swap(offset.d[3], offset.d[4]);

	TSF::SpotList spotList;
	uint32_t mSize = 0;
	std::string buffer;
	TSF::Spot spot;

	stream.seekg(offset.v, std::ios_base::cur);
	{
		google::protobuf::io::IstreamInputStream input(&stream);
		google::protobuf::io::CodedInputStream codedInput(&input);

		if (!codedInput.ReadVarint32(&mSize))
			throw std::runtime_error("Failed to read SpotList offset");

		if (codedInput.ReadString(&buffer, mSize))
			spotList.ParseFromString(buffer);
		else
			throw std::runtime_error("Failed to read SpotList data");
	}

	m_pixelSize = spotList.has_pixel_size() ? spotList.pixel_size() : 1.f;
	m_width = spotList.has_nr_pixels_x() ? spotList.nr_pixels_x() : 0;
	m_height = spotList.has_nr_pixels_y() ? spotList.nr_pixels_y() : 0;
	m_channels = spotList.has_nr_channels() ? spotList.nr_channels() : 1;
	m_numFrames = spotList.has_nr_frames() ? spotList.nr_frames() : 0;

	m_minZ = std::numeric_limits<float>::max();
	m_maxZ = -std::numeric_limits<float>::max();

	// load localizations from file
	uint32_t spots = (uint32_t)-1;
	if (spotList.has_nr_spots())
		spots = static_cast<uint32_t>(spotList.nr_spots());
	reserve(spots);

	Localization l;

	stream.clear();
	stream.seekg(12, std::ios_base::beg);
	if (12 != stream.tellg()) {
		throw std::runtime_error("Failed to set filepointer.  Try setting the read flag");
	} else {
		google::protobuf::io::IstreamInputStream input(&stream);
		google::protobuf::io::CodedInputStream codedInput(&input);

		clear();
		for (uint32_t i = 0; i < spots; ++i) {
			if (!codedInput.ReadVarint32(&mSize))
				throw std::runtime_error("Failed to read Spot size");
			if (codedInput.ReadString(&buffer, mSize))
				spot.ParseFromString(buffer);
			else
				throw std::runtime_error("Failed to read Spot data");

			if (!spot.has_x() || !spot.has_y() || !spot.has_z())
				throw std::runtime_error("XYZ spots are required");

			// generate Localization entry
			l.x = spot.x();
			l.y = spot.y();
			l.z = spot.z();
			l.PAx = spot.has_x_precision() ? spot.x_precision() : 25.f;
			l.PAy = spot.has_y_precision() ? spot.y_precision() : 25.f;
			l.PAz = spot.has_z_precision() ? spot.z_precision() : 75.f;
			l.channel = spot.has_channel() ? spot.channel() : -1;
			l.frame = spot.has_frame() ? spot.frame() : 0;
			l.intensity = spot.has_intensity() ? spot.intensity() : 500.f;
			l.background = spot.has_background() ? spot.background() : 100.f;

			if (spot.location_units() == TSF::UM) {
				l.x *= 1000.f;
				l.y *= 1000.f;
				l.z *= 1000.f;
			} else if (spot.location_units() == TSF::PIXELS) {
				l.x *= m_pixelSize;
				l.y *= m_pixelSize;
				l.z *= m_pixelSize;
			}

			m_minZ = std::min(m_minZ, l.z);
			m_maxZ = std::max(m_maxZ, l.z);

			cb(i, spots, l);

			push_back(l);
		}
	}


	stream.close();
}

void Localizations::copyMetaDataFrom(const Localizations &other)
{
	m_width = other.m_width;
	m_height = other.m_height;
	m_pixelSize = other.m_pixelSize;
	m_minZ = other.m_minZ;
	m_maxZ = other.m_maxZ;
	m_channels = other.m_channels;
}

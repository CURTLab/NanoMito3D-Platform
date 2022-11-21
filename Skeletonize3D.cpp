/****************************************************************************
 * Skeletonize3D plugin for ImageJ(C).
 * Copyright (C) 2008 Ignacio Arganda-Carreras
 * Copyright (C) 2022 Fabian Hauser, adatpted for C++
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation (http://www.gnu.org/licenses/gpl.txt )
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 ****************************************************************************/

#include "Skeletonize3D.h"

#include <algorithm>
#include <vector>
#include <iostream>

namespace Skeleton3D
{

static constexpr char eulerLUT[256] = { 0,  1,  0, -1,  0, -1,  0,  1,  0, -3,  0, -1,  0, -1,  0,  1,  0, -1,  0,  1,  0,  1,  0, -1,  0,  3,
													 0,  1,  0,  1,  0, -1,  0, -3,  0, -1,  0,  3,  0,  1,  0,  1,  0, -1,  0,  3,  0,  1,  0, -1,  0,  1,
													 0,  1,  0, -1,  0,  3,  0,  1,  0,  1,  0, -1,  0, -3,  0,  3,  0, -1,  0,  1,  0,  1,  0,  3,  0, -1,
													 0,  1,  0, -1,  0,  1,  0,  1,  0, -1,  0,  3,  0,  1,  0,  1,  0, -1,  0,  1,  0,  3,  0,  3,  0,  1,
													 0,  5,  0,  3,  0,  3,  0,  1,  0, -1,  0,  1,  0,  1,  0, -1,  0,  3,  0,  1,  0,  1,  0, -1,  0, -7,
													 0, -1,  0, -1,  0,  1,  0, -3,  0, -1,  0, -1,  0,  1,  0, -1,  0,  1,  0,  1,  0, -1,  0,  3,  0,  1,
													 0,  1,  0, -1,  0, -3,  0, -1,  0,  3,  0,  1,  0,  1,  0, -1,  0,  3,  0,  1,  0, -1,  0,  1,  0,  1,
													 0, -1,  0,  3,  0,  1,  0,  1,  0, -1,  0, -3,  0,  3,  0, -1,  0,  1,  0,  1,  0,  3,  0, -1,  0,  1,
													 0, -1,  0,  1,  0,  1,  0, -1,  0,  3,  0,  1,  0,  1,  0, -1,  0,  1,  0,  3,  0,  3,  0,  1,  0,  5,
													 0,  3,  0,  3,  0,  1,  0, -1,  0,  1,  0,  1,  0, -1,  0,  3,  0,  1,  0,  1,  0, -1};


static constexpr std::array<int,3> borderOffsets[6] = {{0, -1, 0}, // North
																		 {0, +1, 0}, // South
																		 {+1, 0, 0}, // East
																		 {-1, 0, 0}, // West
																		 {0, 0, +1}, // Up
																		 {0, 0, -1}, // Bottom
																		};

void getNeighborhood(const Volume &v, const std::array<int,3> &p, bool neighbors[27])
{
	const int x = p[0], y = p[1], z = p[2];

	neighbors[ 0] = v.value(x-1, y-1, z-1);
	neighbors[ 1] = v.value(x  , y-1, z-1);
	neighbors[ 2] = v.value(x+1, y-1, z-1);

	neighbors[ 3] = v.value(x-1, y,   z-1);
	neighbors[ 4] = v.value(x,   y,   z-1);
	neighbors[ 5] = v.value(x+1, y,   z-1);

	neighbors[ 6] = v.value(x-1, y+1, z-1);
	neighbors[ 7] = v.value(x,   y+1, z-1);
	neighbors[ 8] = v.value(x+1, y+1, z-1);

	neighbors[ 9] = v.value(x-1, y-1, z  );
	neighbors[10] = v.value(x,   y-1, z  );
	neighbors[11] = v.value(x+1, y-1, z  );

	neighbors[12] = v.value(x-1, y,   z  );
	neighbors[13] = v.value(x,   y,   z  );
	neighbors[14] = v.value(x+1, y,   z  );

	neighbors[15] = v.value(x-1, y+1, z  );
	neighbors[16] = v.value(x,   y+1, z  );
	neighbors[17] = v.value(x+1, y+1, z  );

	neighbors[18] = v.value(x-1, y-1, z+1);
	neighbors[19] = v.value(x,   y-1, z+1);
	neighbors[20] = v.value(x+1, y-1, z+1);

	neighbors[21] = v.value(x-1, y,   z+1);
	neighbors[22] = v.value(x,   y,   z+1);
	neighbors[23] = v.value(x+1, y,   z+1);

	neighbors[24] = v.value(x-1, y+1, z+1);
	neighbors[25] = v.value(x,   y+1, z+1);
	neighbors[26] = v.value(x+1, y+1, z+1);
}

uint8_t indexOctantNEB(bool neighbors[27]) {
	uint8_t n= 1;
	if (neighbors[2])  n |= 128;
	if (neighbors[1])  n |=  64;
	if (neighbors[11]) n |=  32;
	if (neighbors[10]) n |=  16;
	if (neighbors[5])  n |=   8;
	if (neighbors[4])  n |=   4;
	if (neighbors[14]) n |=   2;
	return n;
}

uint8_t indexOctantNWB(bool neighbors[27]) {
	uint8_t n = 1;
	if (neighbors[0])  n |= 128;
	if (neighbors[9])  n |=  64;
	if (neighbors[3])  n |=  32;
	if (neighbors[12]) n |=  16;
	if (neighbors[1])  n |=   8;
	if (neighbors[10]) n |=   4;
	if (neighbors[4])  n |=   2;
	return n;
}

uint8_t indextOctantSEB(bool neighbors[27]) {
	uint8_t n = 1;
	if (neighbors[8])
		n |= 128;
	if (neighbors[7])
		n |=  64;
	if (neighbors[17])
		n |=  32;
	if (neighbors[16])
		n |=  16;
	if (neighbors[5])
		n |=   8;
	if (neighbors[4])
		n |=   4;
	if (neighbors[14])
		n |=   2;
	return n;
}

uint8_t indexOctantSWB(bool neighbors[27]) {
	uint8_t n = 1;
	if (neighbors[6])
		n |= 128;
	if (neighbors[15])
		n |=  64;
	if (neighbors[7])
		n |=  32;
	if (neighbors[16])
		n |=  16;
	if (neighbors[3])
		n |=   8;
	if (neighbors[12])
		n |=   4;
	if (neighbors[4])
		n |=   2;
	return n;
}

uint8_t indexOctantNEU(bool neighbors[27]) {
	uint8_t n = 1;
	if (neighbors[20])
		n |= 128;
	if (neighbors[23])
		n |=  64;
	if (neighbors[19])
		n |=  32;
	if (neighbors[22])
		n |=  16;
	if (neighbors[11])
		n |=   8;
	if (neighbors[14])
		n |=   4;
	if (neighbors[10])
		n |=   2;
	return n;
}

uint8_t indexOctantNWU(bool neighbors[27]) {
	uint8_t n = 1;
	if (neighbors[18])
		n |= 128;
	if (neighbors[21])
		n |=  64;
	if (neighbors[9])
		n |=  32;
	if (neighbors[12])
		n |=  16;
	if (neighbors[19])
		n |=   8;
	if (neighbors[22])
		n |=   4;
	if (neighbors[10])
		n |=   2;
	return n;
}

uint8_t indexOctantSEU(bool neighbors[27]) {
	uint8_t n = 1;
	if (neighbors[26])
		n |= 128;
	if (neighbors[23])
		n |=  64;
	if (neighbors[17])
		n |=  32;
	if (neighbors[14])
		n |=  16;
	if (neighbors[25])
		n |=   8;
	if (neighbors[22])
		n |=   4;
	if (neighbors[16])
		n |=   2;
	return n;
}

uint8_t indexOctantSWU(bool neighbors[27]) {
	uint8_t n = 1;
	if (neighbors[24])
		n |= 128;
	if (neighbors[25])
		n |=  64;
	if (neighbors[15])
		n |=  32;
	if (neighbors[16])
		n |=  16;
	if (neighbors[21])
		n |=   8;
	if (neighbors[22])
		n |=   4;
	if (neighbors[12])
		n |=   2;
	return n;
}

bool isEndPoint(bool neighbors[27])
{
	int numberOfNeighbors = -1;
	for (int i = 0; i < 27; ++i)
		numberOfNeighbors += neighbors[i];
	return numberOfNeighbors == 1;
}

bool isEulerInvariant(bool neighbors[27])
{
	int eulerChar = 0;
	uint8_t n;

	// Octant SWU
	n = indexOctantSWU(neighbors);
	eulerChar += eulerLUT[n];

	// Octant SEU
	n = indexOctantSEU(neighbors);
	eulerChar += eulerLUT[n];

	// Octant NWU
	n = indexOctantNWU(neighbors);
	eulerChar += eulerLUT[n];

	// Octant NEU
	n = indexOctantNEU(neighbors);
	eulerChar += eulerLUT[n];

	// Octant SWB
	n = indexOctantSWB(neighbors);
	eulerChar += eulerLUT[n];

	// Octant SEB
	n = indextOctantSEB(neighbors);
	eulerChar += eulerLUT[n];

	// Octant NWB
	n = indexOctantNWB(neighbors);
	eulerChar += eulerLUT[n];

	// Octant NEB
	n = indexOctantNEB(neighbors);
	eulerChar += eulerLUT[n];

	return (eulerChar == 0);
}

/**
 * This is a recursive method that calculates the number of connected
 * components in the 3D neighborhood after the center pixel would
 * have been removed.
 *
 * @param octant
 * @param label
 * @param cube
 */
void octreeLabeling(int octant, int label, int cube[26])
{
	// check if there are points in the octant with value 1
	if( octant==1 )
	{
		// set points in this octant to current label
		// and recursive labeling of adjacent octants
		if( cube[0] == 1 )
			cube[0] = label;
		if( cube[1] == 1 )
		{
			cube[1] = label;
			octreeLabeling( 2, label, cube);
		}
		if( cube[3] == 1 )
		{
			cube[3] = label;
			octreeLabeling( 3, label, cube);
		}
		if( cube[4] == 1 )
		{
			cube[4] = label;
			octreeLabeling( 2, label, cube);
			octreeLabeling( 3, label, cube);
			octreeLabeling( 4, label, cube);
		}
		if( cube[9] == 1 )
		{
			cube[9] = label;
			octreeLabeling( 5, label, cube);
		}
		if( cube[10] == 1 )
		{
			cube[10] = label;
			octreeLabeling( 2, label, cube);
			octreeLabeling( 5, label, cube);
			octreeLabeling( 6, label, cube);
		}
		if( cube[12] == 1 )
		{
			cube[12] = label;
			octreeLabeling( 3, label, cube);
			octreeLabeling( 5, label, cube);
			octreeLabeling( 7, label, cube);
		}
	}
	if( octant==2 )
	{
		if( cube[1] == 1 )
		{
			cube[1] = label;
			octreeLabeling( 1, label, cube);
		}
		if( cube[4] == 1 )
		{
			cube[4] = label;
			octreeLabeling( 1, label, cube);
			octreeLabeling( 3, label, cube);
			octreeLabeling( 4, label, cube);
		}
		if( cube[10] == 1 )
		{
			cube[10] = label;
			octreeLabeling( 1, label, cube);
			octreeLabeling( 5, label, cube);
			octreeLabeling( 6, label, cube);
		}
		if( cube[2] == 1 )
			cube[2] = label;
		if( cube[5] == 1 )
		{
			cube[5] = label;
			octreeLabeling( 4, label, cube);
		}
		if( cube[11] == 1 )
		{
			cube[11] = label;
			octreeLabeling( 6, label, cube);
		}
		if( cube[13] == 1 )
		{
			cube[13] = label;
			octreeLabeling( 4, label, cube);
			octreeLabeling( 6, label, cube);
			octreeLabeling( 8, label, cube);
		}
	}
	if( octant==3 )
	{
		if( cube[3] == 1 )
		{
			cube[3] = label;
			octreeLabeling( 1, label, cube);
		}
		if( cube[4] == 1 )
		{
			cube[4] = label;
			octreeLabeling( 1, label, cube);
			octreeLabeling( 2, label, cube);
			octreeLabeling( 4, label, cube);
		}
		if( cube[12] == 1 )
		{
			cube[12] = label;
			octreeLabeling( 1, label, cube);
			octreeLabeling( 5, label, cube);
			octreeLabeling( 7, label, cube);
		}
		if( cube[6] == 1 )
			cube[6] = label;
		if( cube[7] == 1 )
		{
			cube[7] = label;
			octreeLabeling( 4, label, cube);
		}
		if( cube[14] == 1 )
		{
			cube[14] = label;
			octreeLabeling( 7, label, cube);
		}
		if( cube[15] == 1 )
		{
			cube[15] = label;
			octreeLabeling( 4, label, cube);
			octreeLabeling( 7, label, cube);
			octreeLabeling( 8, label, cube);
		}
	}
	if( octant==4 )
	{
		if( cube[4] == 1 )
		{
			cube[4] = label;
			octreeLabeling( 1, label, cube);
			octreeLabeling( 2, label, cube);
			octreeLabeling( 3, label, cube);
		}
		if( cube[5] == 1 )
		{
			cube[5] = label;
			octreeLabeling( 2, label, cube);
		}
		if( cube[13] == 1 )
		{
			cube[13] = label;
			octreeLabeling( 2, label, cube);
			octreeLabeling( 6, label, cube);
			octreeLabeling( 8, label, cube);
		}
		if( cube[7] == 1 )
		{
			cube[7] = label;
			octreeLabeling( 3, label, cube);
		}
		if( cube[15] == 1 )
		{
			cube[15] = label;
			octreeLabeling( 3, label, cube);
			octreeLabeling( 7, label, cube);
			octreeLabeling( 8, label, cube);
		}
		if( cube[8] == 1 )
			cube[8] = label;
		if( cube[16] == 1 )
		{
			cube[16] = label;
			octreeLabeling( 8, label, cube);
		}
	}
	if( octant==5 )
	{
		if( cube[9] == 1 )
		{
			cube[9] = label;
			octreeLabeling( 1, label, cube);
		}
		if( cube[10] == 1 )
		{
			cube[10] = label;
			octreeLabeling( 1, label, cube);
			octreeLabeling( 2, label, cube);
			octreeLabeling( 6, label, cube);
		}
		if( cube[12] == 1 )
		{
			cube[12] = label;
			octreeLabeling( 1, label, cube);
			octreeLabeling( 3, label, cube);
			octreeLabeling( 7, label, cube);
		}
		if( cube[17] == 1 )
			cube[17] = label;
		if( cube[18] == 1 )
		{
			cube[18] = label;
			octreeLabeling( 6, label, cube);
		}
		if( cube[20] == 1 )
		{
			cube[20] = label;
			octreeLabeling( 7, label, cube);
		}
		if( cube[21] == 1 )
		{
			cube[21] = label;
			octreeLabeling( 6, label, cube);
			octreeLabeling( 7, label, cube);
			octreeLabeling( 8, label, cube);
		}
	}
	if( octant==6 )
	{
		if( cube[10] == 1 )
		{
			cube[10] = label;
			octreeLabeling( 1, label, cube);
			octreeLabeling( 2, label, cube);
			octreeLabeling( 5, label, cube);
		}
		if( cube[11] == 1 )
		{
			cube[11] = label;
			octreeLabeling( 2, label, cube);
		}
		if( cube[13] == 1 )
		{
			cube[13] = label;
			octreeLabeling( 2, label, cube);
			octreeLabeling( 4, label, cube);
			octreeLabeling( 8, label, cube);
		}
		if( cube[18] == 1 )
		{
			cube[18] = label;
			octreeLabeling( 5, label, cube);
		}
		if( cube[21] == 1 )
		{
			cube[21] = label;
			octreeLabeling( 5, label, cube);
			octreeLabeling( 7, label, cube);
			octreeLabeling( 8, label, cube);
		}
		if( cube[19] == 1 )
			cube[19] = label;
		if( cube[22] == 1 )
		{
			cube[22] = label;
			octreeLabeling( 8, label, cube);
		}
	}
	if( octant==7 )
	{
		if( cube[12] == 1 )
		{
			cube[12] = label;
			octreeLabeling( 1, label, cube);
			octreeLabeling( 3, label, cube);
			octreeLabeling( 5, label, cube);
		}
		if( cube[14] == 1 )
		{
			cube[14] = label;
			octreeLabeling( 3, label, cube);
		}
		if( cube[15] == 1 )
		{
			cube[15] = label;
			octreeLabeling( 3, label, cube);
			octreeLabeling( 4, label, cube);
			octreeLabeling( 8, label, cube);
		}
		if( cube[20] == 1 )
		{
			cube[20] = label;
			octreeLabeling( 5, label, cube);
		}
		if( cube[21] == 1 )
		{
			cube[21] = label;
			octreeLabeling( 5, label, cube);
			octreeLabeling( 6, label, cube);
			octreeLabeling( 8, label, cube);
		}
		if( cube[23] == 1 )
			cube[23] = label;
		if( cube[24] == 1 )
		{
			cube[24] = label;
			octreeLabeling( 8, label, cube);
		}
	}
	if( octant==8 )
	{
		if( cube[13] == 1 )
		{
			cube[13] = label;
			octreeLabeling( 2, label, cube);
			octreeLabeling( 4, label, cube);
			octreeLabeling( 6, label, cube);
		}
		if( cube[15] == 1 )
		{
			cube[15] = label;
			octreeLabeling( 3, label, cube);
			octreeLabeling( 4, label, cube);
			octreeLabeling( 7, label, cube);
		}
		if( cube[16] == 1 )
		{
			cube[16] = label;
			octreeLabeling( 4, label, cube);
		}
		if( cube[21] == 1 )
		{
			cube[21] = label;
			octreeLabeling( 5, label, cube);
			octreeLabeling( 6, label, cube);
			octreeLabeling( 7, label, cube);
		}
		if( cube[22] == 1 )
		{
			cube[22] = label;
			octreeLabeling( 6, label, cube);
		}
		if( cube[24] == 1 )
		{
			cube[24] = label;
			octreeLabeling( 7, label, cube);
		}
		if( cube[25] == 1 )
			cube[25] = label;
	}

}

/**
 * Check if current point is a Simple Point.
 * This method is named 'N(v)_labeling' in [Lee94].
 * Outputs the number of connected objects in a neighborhood of a point
 * after this point would have been removed.
 *
 * @param neighbors neighbor pixels of the point
 * @return true or false if the point is simple or not
 */
bool isSimplePoint(bool neighbors[27])
{
	int cube[26];
	int i;
	for (i = 0; i < 13; ++i)  // i =  0..12 -> cube[0..12]
		cube[i] = neighbors[i];
	for (i = 14; i < 27; ++i) // i = 14..26 -> cube[13..25]
		cube[i-1] = neighbors[i];
	// set initial label
	int label = 2;
	// for all points in the neighborhood
	for (i = 0; i < 26; ++i) {
		if (cube[i] == 1) { // voxel has not been labeled yet
			// start recursion with any octant that contains the point i
			switch(i)
			{
			case 0:
			case 1:
			case 3:
			case 4:
			case 9:
			case 10:
			case 12:
				octreeLabeling(1, label, cube);
				break;
			case 2:
			case 5:
			case 11:
			case 13:
				octreeLabeling(2, label, cube);
				break;
			case 6:
			case 7:
			case 14:
			case 15:
				octreeLabeling(3, label, cube);
				break;
			case 8:
			case 16:
				octreeLabeling(4, label, cube);
				break;
			case 17:
			case 18:
			case 20:
			case 21:
				octreeLabeling(5, label, cube);
				break;
			case 19:
			case 22:
				octreeLabeling(6, label, cube);
				break;
			case 23:
			case 24:
				octreeLabeling(7, label, cube);
				break;
			case 25:
				octreeLabeling(8, label, cube);
				break;
			}
			label++;
			if (label-2 >= 2)
			{
				return false;
			}
		}
	}
	return true;
}

bool isSimplePoint2(bool neighbors[27])
{
	int cube[26];
	int i;
	for (i = 0; i < 13; ++i)  // i =  0..12 -> cube[0..12]
		cube[i] = neighbors[i];
	for (i = 14; i < 27; ++i) // i = 14..26 -> cube[13..25]
		cube[i-1] = neighbors[i];
	// set initial label
	int label = 2;
	// for all points in the neighborhood
	for (i = 0; i < 26; ++i) {
		if (cube[i] == 1) { // voxel has not been labeled yet
			// start recursion with any octant that contains the point i
			switch(i)
			{
			case 0:
			case 1:
			case 3:
			case 4:
			case 9:
			case 10:
			case 12:
				octreeLabeling(1, label, cube);
				break;
			case 2:
			case 5:
			case 11:
			case 13:
				octreeLabeling(2, label, cube);
				break;
			case 6:
			case 7:
			case 14:
			case 15:
				octreeLabeling(3, label, cube);
				break;
			case 8:
			case 16:
				octreeLabeling(4, label, cube);
				break;
			case 17:
			case 18:
			case 20:
			case 21:
				octreeLabeling(5, label, cube);
				break;
			case 19:
			case 22:
				octreeLabeling(6, label, cube);
				break;
			case 23:
			case 24:
				octreeLabeling(7, label, cube);
				break;
			case 25:
				octreeLabeling(8, label, cube);
				break;
			}
			label++;
			if (label-2 >= 2)
			{
				return false;
			}
		}
	}
	return true;
}

// simple helper for std array int3
inline constexpr std::array<int,3> operator+(const std::array<int,3> &p1, const std::array<int,3> &p2)
{
	return {p1[0] + p2[0], p1[1] + p2[1], p1[2] + p2[2]};
}

}

void Skeleton3D::skeletonize(const Volume &input, Volume &o)
{
	// be sure that values for skinning are either 0 or 255
	Volume result(input.size(), input.voxelSize(), input.origin());
	std::transform(input.constData(), input.constData() + input.voxels(), result.data(), [](uint8_t v) { return v > 0 ? 255 : 0; });

	std::vector<std::array<int,3>> simpleBorderPoints;
	simpleBorderPoints.reserve(256);

	bool neighbors[27];

	int iterations = 0;
	// Loop through the image several times until there is no change.
	int unchangedBorders = 0;
	// loop until no change for all the six border types
	while (unchangedBorders < 6) {
		unchangedBorders = 0;
		++iterations;

		for (int currentBorder = 0; currentBorder < 6; ++currentBorder) {
			// Loop through the image
			for (int z = 0; z < input.depth(); ++z) {
				for (int y = 0; y < input.height(); ++y) {
					for (int x = 0; x < input.width(); ++x) {
						const std::array<int,3> p{x, y, z};

						// check if point is foreground
						if (result(x, y, z) != 255)
							continue;

						// check 6-neighbors if point is a border point of type currentBorder
						bool isBorderPoint = false;
						if (result.value(p + borderOffsets[currentBorder]) == 0)
							isBorderPoint = true;

						if (!isBorderPoint)
							continue; // current point is not deletable

						getNeighborhood(result, p, neighbors);

						if (isEndPoint(neighbors))
							continue; // current point is not deletable

						// Check if point is Euler invariant (condition 1 in Lee[94])
						if (!isEulerInvariant(neighbors))
							continue; // current point is not deletable

						// Check if point is simple (deletion does not change connectivity in the 3x3x3 neighborhood)
						// (conditions 2 and 3 in Lee[94])
						if (!isSimplePoint(neighbors))
							continue; // current point is not deletable

						simpleBorderPoints.push_back(p);
					}
				}
			}

			bool noChange = true;
			for (const auto &p : simpleBorderPoints) {
				getNeighborhood(result, p, neighbors);

				if (isSimplePoint2(neighbors)) {
					result.setValue(p, 0);
					noChange = false;
				}
			}

			if (noChange)
				unchangedBorders++;

			simpleBorderPoints.clear();
		}
	}

	// finally assign output
	o = result;
}

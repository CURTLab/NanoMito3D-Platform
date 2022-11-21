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

#ifndef SKELETONIZE3D_H
#define SKELETONIZE3D_H

#include "Volume.h"

namespace Skeleton3D
{

void skeletonize(const Volume &input, Volume &output);

}

#endif // SKELETONIZE3D_H

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

#ifndef VOLUMEWIDGET_H
#define VOLUMEWIDGET_H

#include <QWidget>

#include "Volume.h"
#include "SkeletonGraph.h"
#include "Localizations.h"

class VolumeWidgetPrivate;

class VolumeWidget : public QWidget
{
	Q_OBJECT
	Q_DECLARE_PRIVATE_D(m_d, VolumeWidget)
public:
	VolumeWidget(QWidget *parent = nullptr);
	~VolumeWidget();

	void clear();

	// sets the bounds and axis annotations to the bounds in µm (xmin,xmax, ymin,ymax, zmin,zmax)
	void setBounds(double bounds[6]);

	void addVolume(Volume volume, std::array<double,4> color, bool copyData = true);
	void addLocalizations(const Localizations &locs, float pointSize, std::array<double,3> color);
	void addSpheres(const std::vector<std::array<float,3>> &points, float r, std::array<double,3> color);
	void addGraph(std::shared_ptr<SkeletonGraph> graph, const Volume &volume, float r, std::array<double,3> color);
	void addClassifiedVolume(Volume volume, int classes, bool copyData = true);

	void saveAsPNG(const QString &fileName);

public slots:
	void resetCamera();
	void replot();

private:
	VolumeWidgetPrivate * const m_d;

};

#endif // VOLUMEWIDGET_H

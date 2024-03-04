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

#ifndef IMAGEPLOTWIDGET_H
#define IMAGEPLOTWIDGET_H

#include <QWidget>
#include <opencv2/opencv.hpp>

class ImagePlotWidgetPrivate;

enum class ColorMap
{
	Gray,
	Hot
};

class ImagePlotWidget : public QWidget
{
	Q_DECLARE_PRIVATE_D(m_d, ImagePlotWidget)
public:
	ImagePlotWidget(QWidget *parent = nullptr);
	virtual ~ImagePlotWidget();

	virtual void setImage(const cv::Mat &image);

	void setColorMap(ColorMap cmap);

	void replot();
	void clear();
	void clearAnnotation();

	// annotation
	void addCircles(const QVector<QPointF> &dataPoints, QColor color, qreal radius);

protected:
	void *plot();

private:
	ImagePlotWidgetPrivate* const m_d;

};

#endif // IMAGEPLOTWIDGET_H

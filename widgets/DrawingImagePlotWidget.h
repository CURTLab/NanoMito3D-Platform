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
#ifndef DRAWINGIMAGEPLOTWIDGET_H
#define DRAWINGIMAGEPLOTWIDGET_H

#include "ImagePlotWidget.h"

class DrawingImagePlotWidget : public ImagePlotWidget
{
public:
	DrawingImagePlotWidget(QWidget *parent = nullptr);

	QImage paintOverlay() const;
	void setPaintOverlay(QImage image) const;
	void clearOverlay();

	void setPaintToolWidth(int width);
	void setPaintToolColor(QColor color);

	virtual void setImage(const cv::Mat &image) override;

private:
	class QwtPlotPicker *m_painter;
	class PaintCanvasItem *m_paintCanvas;

	int m_toolWidth;
	QColor m_toolColor;

};

#endif // DRAWINGIMAGEPLOTWIDGET_H

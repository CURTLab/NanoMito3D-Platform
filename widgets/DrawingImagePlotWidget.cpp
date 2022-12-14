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
#include "DrawingImagePlotWidget.h"
#include "qwt_plot.h"

#include <qwt_scale_map.h>
#include <qwt_picker_machine.h>
#include <qwt_plot_item.h>
#include <qwt_picker_machine.h>
#include <qwt_plot_picker.h>

#include <QPainter>

class PaintCanvasItem : public QwtPlotItem
{
public:
	PaintCanvasItem(QSize size)
		: canvasSize(size)
	{
		setZ(10);
		setItemInterest(ScaleInterest, true);
		setItemInterest(LegendInterest, false);
		setItemAttribute(Legend, false);

		image = QImage(canvasSize, QImage::Format_ARGB32);
		image.fill(Qt::transparent);
	}

	QSize canvasSize;
	QImage image;

	virtual inline int rtti() const override { return QwtPlotItem::Rtti_PlotUserItem + 1; }

	virtual void draw( QPainter *painter,
							 const QwtScaleMap &xMap, const QwtScaleMap &yMap,
							 const QRectF & ) const override
	{
		QRectF r = QwtScaleMap::transform(xMap, yMap, boundingRect());
		painter->save();
		painter->setOpacity(0.4);
		painter->drawImage(r, image);
		painter->restore();
	}

	virtual QRectF boundingRect() const override
	{
		return QRectF(0, 0, canvasSize.width(), canvasSize.height());
	}
};

DrawingImagePlotWidget::DrawingImagePlotWidget(QWidget *parent)
	: ImagePlotWidget(parent)
	, m_painter(nullptr)
	, m_paintCanvas(nullptr)
{
	QwtPlot *plot = (QwtPlot*)(ImagePlotWidget::plot());

	m_painter = new QwtPlotPicker(QwtPlot::xBottom, QwtPlot::yLeft,
											QwtPicker::NoRubberBand, QwtPicker::AlwaysOff,
											plot->canvas());
	m_painter->setStateMachine(new QwtPickerDragPointMachine);
	m_painter->setEnabled(true);

	connect(m_painter, &QwtPlotPicker::moved, this, [this,plot](const QPointF &pos){
		if (m_paintCanvas && m_painter->isEnabled()) {
			const QwtScaleMap xMap = plot->canvasMap(QwtPlot::xBottom);
			const QwtScaleMap yMap = plot->canvasMap(QwtPlot::yLeft);

			double r1 = abs(xMap.invTransform(0.5*m_toolWidth) - xMap.invTransform(0.0));
			double r2 = abs(yMap.invTransform(0.5*m_toolWidth) - yMap.invTransform(0.0));

			const bool erase = (m_toolColor == Qt::transparent);
			QPainter p(&m_paintCanvas->image);
			p.setCompositionMode(erase?QPainter::CompositionMode_Clear:QPainter::CompositionMode_SourceOver);
			p.setPen(Qt::NoPen);
			p.setBrush(m_toolColor);
			p.drawEllipse(pos, r1, r2);
			plot->replot();
		}
	});
}

QImage DrawingImagePlotWidget::paintOverlay() const
{
	return (m_paintCanvas ? m_paintCanvas->image : QImage{});
}

void DrawingImagePlotWidget::setPaintOverlay(QImage image) const
{
	if (m_paintCanvas && !image.isNull() && !m_paintCanvas->canvasSize.isNull())
		m_paintCanvas->image = image.scaled(m_paintCanvas->canvasSize);
}

void DrawingImagePlotWidget::clearOverlay()
{
	if (m_paintCanvas)
		m_paintCanvas->image.fill(Qt::transparent);
	((QwtPlot*)plot())->replot();
}

void DrawingImagePlotWidget::setPaintToolWidth(int width)
{
	m_toolWidth = width;

	QPixmap cursor(width + 1, width + 1);
	cursor.fill( QColor(255,255,255,0) );
	QPainter p;
	p.begin(&cursor);
	p.setRenderHint(QPainter::Antialiasing);
	p.setPen(Qt::white);
	p.setBrush(QColor(255, 255, 255, 100));
	p.drawEllipse(0, 0, width, width);
	p.end();
	((QwtPlot*)plot())->canvas()->setCursor(QCursor(cursor, width / 2, width / 2));
}

void DrawingImagePlotWidget::setPaintToolColor(QColor color)
{
	m_toolColor = color;
}

void DrawingImagePlotWidget::setImage(const cv::Mat &image)
{
	if (image.empty())
		return;

	const QSize size(image.rows, image.cols);
	if (m_paintCanvas == nullptr || m_paintCanvas->canvasSize != size) {
		if (m_paintCanvas != nullptr) {
			m_paintCanvas->detach();
			delete m_paintCanvas;
		}
		m_paintCanvas = new PaintCanvasItem(size);
		m_paintCanvas->attach((QwtPlot*)plot());
	}

	ImagePlotWidget::setImage(image);
}

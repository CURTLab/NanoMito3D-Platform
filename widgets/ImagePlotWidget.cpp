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

#include "ImagePlotWidget.h"

#include <QVBoxLayout>
#include <QApplication>
#include <QtMath>

#include <qwt_plot.h>
#include <qwt_plot_layout.h>
#include <qwt_scale_widget.h>
#include <qwt_painter.h>
#include <qwt_plot_spectrogram.h>
#include <qwt_plot_rescaler.h>
#include <qwt_color_map.h>

class RasterData : public QwtRasterData
{
	cv::Mat m_image;
public:
	inline RasterData(cv::Mat image, QwtInterval interval)
	{ setData(image, interval); }

	virtual inline double value(double x, double y) const override {
		if (m_image.empty())
			return 0.0;
		const double d = m_image.at<double>(static_cast<int>(std::floor(x)), static_cast<int>(std::floor(y)));
		return d;
	}

	const cv::Mat &image() const { return m_image; }

	inline void setData(cv::Mat image, QwtInterval interval) {
		image.convertTo(m_image, CV_64F);
		setInterval(Qt::XAxis, QwtInterval(0, image.rows-1E-9));
		setInterval(Qt::YAxis, QwtInterval(0, image.cols-1E-9));
		setInterval(Qt::ZAxis, interval);
	}

	inline void updateScale(double min, double max)
	{
		setInterval(Qt::ZAxis, {min, max});
	}

	inline void clear() {
		m_image = cv::Mat{};
		setInterval(Qt::XAxis, QwtInterval(0, 1));
		setInterval(Qt::YAxis, QwtInterval(0, 1));
		setInterval(Qt::ZAxis, QwtInterval(0, 1000));
	}

};

class ImagePlotWidgetScaleDraw : public QwtScaleDraw
{
public:
	inline ImagePlotWidgetScaleDraw() : QwtScaleDraw(), m_unitScale(1.0)
	{
		setTickLength(QwtScaleDiv::MajorTick, 14);
		setTickLength(QwtScaleDiv::MediumTick, 6);
		setTickLength(QwtScaleDiv::MinorTick, 4);
	}

	virtual inline double extent(const QFont &) const override { return 18; }
	virtual inline QwtText label(double val) const override { return QString::number(qRound(val)); }

	double m_unitScale;
	QString m_unitText;

protected:
	virtual inline void drawTrack(QPainter *) const {}

	virtual inline void drawBackbone(QPainter *p) const override {
		p->save();
		p->setRenderHint(QPainter::Antialiasing, true);

		const QPalette palette = qApp->palette();
		QColor colorFG, colorBG;
		colorFG = 0xc0c0c0;
		colorBG = palette.color(QPalette::Base);

		QPen pen(colorFG, 0); // zero width pen is cosmetic pen
		p->setPen(pen);

		const double ext = extent(QFont());
		QRectF rulerRect;
		switch (alignment()) {
		case RightScale:
			rulerRect = QRectF(pos().x(), pos().y(), ext, length()-1);
			break;
		case LeftScale:
			rulerRect = QRectF(pos().x() - ext, pos().y(), ext, length()-1);
			break;
		case BottomScale:
			rulerRect = QRectF(pos().x(), pos().y(), length()-1, ext);
			break;
		case TopScale:
			rulerRect = QRectF(pos().x(), pos().y() - ext, length()-1, ext);
			break;
		}
		p->fillRect(rulerRect, colorBG);
		p->drawRect(rulerRect);
		p->restore();
	}

	virtual inline void drawLabel(QPainter *p, double value) const override {
		p->setRenderHints(QPainter::TextAntialiasing|QPainter::Antialiasing, true);

		const QPalette palette = qApp->palette();
		QColor color;
		color = palette.color(QPalette::ButtonText);

		//const bool roundingAlignment = QwtPainter::roundingAlignment(p);

		double tval = scaleMap().transform(value);
		//if (roundingAlignment) tval = qRound(tval);

		QFont font = p->font();
		font.setPointSizeF(8);

		const double ext = extent(font);
		QwtText lbl = tickLabel(font, value * m_unitScale);
		if (lbl.isEmpty())
			return;
		if (!m_unitText.isEmpty())
			lbl.setText(lbl.text() + m_unitText);
		const QSizeF labelSize = lbl.textSize(font);

		QPointF labelPos;
		QTransform transform;
		switch (alignment()) {
		case BottomScale:
			labelPos = QPointF(tval + 2, pos().y() + 4);
			if (labelPos.x() + labelSize.width() > pos().x() + length())
				return;
			transform.translate(labelPos.x(), labelPos.y());
			break;
		case TopScale:
			labelPos = QPointF(tval + 2, pos().y() + 4 - ext);
			if (labelPos.x() + labelSize.width() > pos().x() + length())
				return;
			transform.translate(labelPos.x(), labelPos.y());
			break;
		case LeftScale:
			labelPos = QPointF(pos().x() + 4 - labelSize.height(), tval + 2);
			if (labelPos.y() + labelSize.width() > pos().y() + length())
				return;
			transform.translate(labelPos.x(), labelPos.y());
			transform.rotate(90);
			break;
		case RightScale:
			labelPos = QPointF(pos().x() + ext - 4 - labelSize.height(), tval + 2);
			if (labelPos.y() + labelSize.width() > pos().y() + length())
				return;
			transform.translate(labelPos.x(), labelPos.y());
			transform.rotate(90);
			break;
		}

		p->save();
		p->setFont(font);
		p->setWorldTransform(transform, true);
		p->setPen(color);

		lbl.draw(p, QRectF(QPointF(0,0), labelSize));
		p->restore();
	}

};

class ImagePlotWidgetPrivate : public QwtPlot
{
public:
	QwtPlotSpectrogram *spectrogram;
	//QwtRasterData *data;
	RasterData *data;
	QwtPlotRescaler *rescaler;

	inline ImagePlotWidgetPrivate(QWidget *parent)
		: QwtPlot(parent)
		, spectrogram(nullptr)
		, data(nullptr)
		, rescaler(nullptr)
	{
		axisWidget(QwtPlot::xBottom)->setScaleDraw(new ImagePlotWidgetScaleDraw);
		axisWidget(QwtPlot::yLeft)->setScaleDraw(new ImagePlotWidgetScaleDraw);

		setAxisAutoScale(QwtPlot::xBottom, false);
		setAxisAutoScale(QwtPlot::yLeft, false);

		spectrogram = new QwtPlotSpectrogram;
		spectrogram->setRenderThreadCount(0);
		spectrogram->setCachePolicy(QwtPlotRasterItem::PaintCache);
		spectrogram->setColorMap(new QwtLinearColorMap(Qt::black, Qt::white));
		spectrogram->attach(this);

		rescaler = new QwtPlotRescaler(canvas(), QwtPlot::xBottom, QwtPlotRescaler::Fitting);
		rescaler->setExpandingDirection(QwtPlotRescaler::ExpandBoth);
		rescaler->setAspectRatio(QwtPlot::yLeft, 1.0);
		rescaler->setAspectRatio(QwtPlot::yRight, 0.0);
		rescaler->setAspectRatio(QwtPlot::xTop, 0.0);

		plotLayout()->setAlignCanvasToScales(true);
		plotLayout()->setCanvasMargin(0);
		plotLayout()->setSpacing(0);

		axisWidget(QwtPlot::yLeft)->setMargin(1);
		axisWidget(QwtPlot::yRight)->setMargin(1);
		axisWidget(QwtPlot::xTop)->setMargin(1);
		axisWidget(QwtPlot::xBottom)->setMargin(1);
	}

};

ImagePlotWidget::ImagePlotWidget(QWidget *parent)
	: m_d(new ImagePlotWidgetPrivate(this))
{
	QVBoxLayout *layout = new QVBoxLayout;
	layout->setObjectName("layer_plot");
	layout->setContentsMargins(0, 0, 0, 0);
	setLayout(layout);
	layout->addWidget(m_d.get());
}

void ImagePlotWidget::setImage(const cv::Mat &image)
{
	if (image.empty())
		return;

	double minVal, maxVal;
	cv::minMaxLoc(image, &minVal, &maxVal);

	QwtInterval intensity(minVal, maxVal);

	qDebug() << image.rows << image.cols << minVal << maxVal;

	m_d->data = new RasterData(image, intensity);
	m_d->spectrogram->setData(m_d->data);

	// Only leave a small border to the image.
	// If image size is selected as inverval program will crash!
	QwtInterval xInt(0, image.rows-1E-9);
	QwtInterval yInt(0, image.cols-1E-9);

	m_d->setAxisScale(QwtPlot::xBottom, xInt.minValue(), xInt.maxValue());
	// in order to invert the y axis
	m_d->setAxisScale(QwtPlot::yLeft, yInt.maxValue(), yInt.minValue());
	m_d->setAxisScaleDiv(QwtPlot::yLeft, QwtScaleDiv(yInt.maxValue(), yInt.minValue()));

	m_d->rescaler->setIntervalHint(QwtPlot::xBottom, xInt);
	m_d->rescaler->setIntervalHint(QwtPlot::yLeft, yInt);
	m_d->rescaler->setIntervalHint(QwtPlot::yRight, intensity);
	m_d->rescaler->rescale();

	m_d->spectrogram->invalidateCache();
	m_d->replot();
}

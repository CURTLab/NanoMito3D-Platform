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

#include "MatPlotWidget.h"

#include <QVBoxLayout>
#include <qwt_plot.h>
#include <qwt_plot_item.h>
#include <qwt_plot_scaleitem.h>
#include <qwt_scale_widget.h>
#include <qwt_plot_grid.h>
#include <qwt_plot_layout.h>
#include <qwt_plot_canvas.h>
#include <qwt_plot_marker.h>
#include <qwt_plot_barchart.h>
#include <qwt_column_symbol.h>

class MatPlotLabelScaleDraw : public QwtScaleDraw
{
	QStringList m_labels;
public:
	inline MatPlotLabelScaleDraw(const QStringList &labels, double rotation = 0.0)
		: QwtScaleDraw()
		, m_labels(labels)
	{
		setTickLength(QwtScaleDiv::MajorTick, 0);
		setTickLength(QwtScaleDiv::MediumTick, 0);
		setTickLength(QwtScaleDiv::MinorTick, 0);

		enableComponent(QwtScaleDraw::Backbone, false);
		enableComponent(QwtScaleDraw::Ticks, false);

		setLabelRotation(rotation);
	}

	virtual QwtText label(double value) const override
	{
		QwtText lbl;

		const int index = qRound(value);
		if (index >= 0 && index < m_labels.size())
			lbl = m_labels[index];

		return lbl;
	}

};

class MatPlotColorChartItem : public QwtPlotBarChart
{
	QVector<QColor> m_colors;
public:
	MatPlotColorChartItem(QVector<QColor> colors) : m_colors(colors) {}

	virtual QwtColumnSymbol *specialSymbol(
			int index, const QPointF& ) const
	{
		QwtColumnSymbol *symbol = new QwtColumnSymbol(QwtColumnSymbol::Box);
		symbol->setLineWidth(2);
		symbol->setFrameStyle(QwtColumnSymbol::Plain);

		QColor c( Qt::white );
		if (index >= 0 && index < m_colors.size())
			c = m_colors[index];
		symbol->setPalette(c);

		return symbol;
	}

};

class MatPlotTick : public QwtPlotItem
{
public:
	MatPlotTick()
		: QwtPlotItem(QwtText("Tick"))
		, m_tickLen(10)
	{
		setItemInterest(QwtPlotItem::ScaleInterest, true);
		setZ(11.0);
	}

	virtual void draw(QPainter *painter,
							const QwtScaleMap &xMap, const QwtScaleMap &yMap,
							const QRectF &canvasRect) const override
	{
		painter->setPen(QPen(QColor(131,131,131), 1.25));
		painter->drawRect(QRectF(1.0, 1.0, canvasRect.width()-2.0, canvasRect.height()-2.0));

		const auto x_div = m_xScaleDiv.ticks(QwtScaleDiv::MajorTick);
		for (const double &tick:x_div) {
			const double x = xMap.transform(tick) + 0.25;
			const QLineF lines[2] = {QLineF(x, canvasRect.height()-(m_tickLen + 1), x, canvasRect.height() - 1),
											 QLineF(x, 1, x, m_tickLen + 1)};
			painter->drawLines(lines, 2);
		}

		const auto y_div = m_yScaleDiv.ticks(QwtScaleDiv::MajorTick);
		for (const double &tick:y_div) {
			const double y = yMap.transform(tick) + 0.25;
			const QLineF lines[2] = {QLineF(canvasRect.width()-(m_tickLen - 1), y, canvasRect.width() - 1, y),
											 QLineF(1, y, m_tickLen + 1, y)};
			painter->drawLines(lines, 2);
		}
	}

	virtual void updateScaleDiv(
			const QwtScaleDiv &xScaleDiv, const QwtScaleDiv &yScaleDiv) override
	{
		if (m_xScaleDiv != xScaleDiv) {
			m_xScaleDiv = xScaleDiv;
			itemChanged();
		}
		if (m_yScaleDiv != yScaleDiv) {
			m_yScaleDiv = yScaleDiv;
			itemChanged();
		}
	}

private:
	QwtScaleDiv m_xScaleDiv;
	QwtScaleDiv m_yScaleDiv;
	double m_tickLen;

};

class MatPlotScaleDraw : public QwtScaleDraw
{
	QStringList m_labels;
public:
	inline MatPlotScaleDraw() : QwtScaleDraw()
	{
		setTickLength(QwtScaleDiv::MajorTick, 0);
		setTickLength(QwtScaleDiv::MediumTick, 0);
		setTickLength(QwtScaleDiv::MinorTick, 0);

		enableComponent(QwtScaleDraw::Backbone, false);
		enableComponent(QwtScaleDraw::Ticks, false);
	}

	virtual QwtText label(double val) const override
	{
		QwtText text(QString::number(val));
		QFont font("Helvetica", 9, 1);
		text.setFont(font);
		return text;
	}

};

class MatPlotWidgetPrivate
{
public:
	inline MatPlotWidgetPrivate()
		: plot(new QwtPlot)
		, tick(new MatPlotTick)
		, grid(new QwtPlotGrid)
	{
		tick->attach(plot);

		plot->setMinimumSize(100, 100);

		plot->axisWidget(QwtPlot::xBottom)->setScaleDraw(new MatPlotScaleDraw);
		//plot->axisWidget(QwtPlot::xTop)->setScaleDraw(new MatPlotScaleDraw);
		plot->axisWidget(QwtPlot::yLeft)->setScaleDraw(new MatPlotScaleDraw);
		//plot->axisWidget(QwtPlot::yRight)->setScaleDraw(new MatPlotScaleDraw);

		grid->setMajorPen(QPen(QColor(235, 235, 235), 1.25));
		grid->attach(plot);

		plot->plotLayout()->setAlignCanvasToScales(true);
		plot->plotLayout()->setCanvasMargin(0, QwtPlot::xBottom);
		plot->plotLayout()->setCanvasMargin(0, QwtPlot::xTop);
		plot->plotLayout()->setCanvasMargin(0, QwtPlot::yLeft);
		plot->plotLayout()->setCanvasMargin(0, QwtPlot::yRight);
		plot->setStyleSheet("background-color: transparent;");

		canvas = qobject_cast<QwtPlotCanvas*>(plot->canvas());
		canvas->setFrameStyle(QFrame::Plain);
		canvas->setBorderRadius(0.0);
		canvas->setCursor(Qt::ArrowCursor);
		canvas->setStyleSheet("background: #fff;");
	}


	inline ~MatPlotWidgetPrivate()
	{
		delete plot;
	}

	QwtPlot *plot;
	MatPlotTick *tick;
	QwtPlotGrid *grid;
	QwtPlotCanvas *canvas;
};

MatPlotWidget::MatPlotWidget(QWidget *parent)
	: QWidget(parent)
	, m_d(new MatPlotWidgetPrivate)
{
	Q_D(MatPlotWidget);

	QVBoxLayout *layout = new QVBoxLayout;
	layout->setContentsMargins(0, 0, 0, 0);
	setLayout(layout);

	layout->addWidget(d->plot);
}

void MatPlotWidget::addBars(const QStringList &values, const QVector<double> &height, const QVector<QColor> &colors, double width)
{
	Q_D(MatPlotWidget);

	QwtPlotBarChart *chart = new MatPlotColorChartItem(colors);

	chart->setLayoutPolicy( QwtPlotBarChart::AutoAdjustSamples );
	chart->setSpacing(20);
	chart->setMargin(3);
	chart->setSamples(height);
	chart->attach(d->plot);

	d->plot->axisWidget(QwtPlot::xBottom)->setScaleDraw(new MatPlotLabelScaleDraw(values));
	d->plot->setAxisScale(QwtPlot::xBottom, -1, height.size(), 1.0);

	d->plot->replot();
}

void MatPlotWidget::addText(const QPointF &postion, const QString &text, QColor color, Qt::Alignment alignment)
{
	Q_D(MatPlotWidget);

	QFont font("Helvetica", 9, 1);

	QwtText title(text);
	title.setFont(font);
	title.setColor(color);

	QwtPlotMarker *item = new QwtPlotMarker();
	item->setLabel(title);
	item->setXValue(postion.x());
	item->setYValue(postion.y());
	item->setLabelAlignment(alignment);
	item->attach(d->plot);

	d->plot->replot();
}

void MatPlotWidget::replot()
{
	Q_D(MatPlotWidget);
	d->plot->replot();
	d->plot->repaint();
}

void MatPlotWidget::clear()
{
	Q_D(MatPlotWidget);
	d->plot->detachItems(QwtPlotItem::Rtti_PlotBarChart);
	d->plot->detachItems(QwtPlotItem::Rtti_PlotSpectrogram);
	d->plot->detachItems(QwtPlotItem::Rtti_PlotHistogram);
	d->plot->detachItems(QwtPlotItem::Rtti_PlotCurve);
	d->plot->detachItems(QwtPlotItem::Rtti_PlotMarker);
	d->plot->axisWidget(QwtPlot::xBottom)->setScaleDraw(new MatPlotScaleDraw);
	d->plot->axisWidget(QwtPlot::yLeft)->setScaleDraw(new MatPlotScaleDraw);
	d->plot->setAxisAutoScale(QwtPlot::xBottom, true);
	d->plot->setAxisAutoScale(QwtPlot::yLeft, false);
	d->plot->replot();
}

void MatPlotWidget::setTitle(const QString &title)
{
	Q_D(MatPlotWidget);
	QFont font("Helvetica", 9, 1);
	font.setBold(true);
	QwtText text(title);
	text.setFont(font);
	d->plot->setTitle(text);
}

void MatPlotWidget::setXLabel(const QString &label)
{
	Q_D(MatPlotWidget);
	QwtText text(label);
	QFont font("Helvetica", 9, 1);
	text.setFont(font);
	d->plot->setAxisTitle(QwtPlot::xBottom, text);
}

void MatPlotWidget::setYLabel(const QString &label)
{
	Q_D(MatPlotWidget);
	QwtText text(label);
	QFont font("Helvetica", 9, 1);
	text.setFont(font);
	d->plot->setAxisTitle(QwtPlot::yLeft, text);
}

void MatPlotWidget::setXScale(double min, double max)
{
	Q_D(MatPlotWidget);
	d->plot->setAxisAutoScale(QwtPlot::xBottom, false);
	d->plot->setAxisScale(QwtPlot::xBottom, min, max);
	d->plot->replot();
}

void MatPlotWidget::setYScale(double min, double max)
{
	Q_D(MatPlotWidget);
	d->plot->setAxisAutoScale(QwtPlot::yLeft, false);
	d->plot->setAxisScale(QwtPlot::yLeft, min, max);
	d->plot->replot();
}

void MatPlotWidget::setLimits(double lowerx, double upperx, double lowery, double uppery)
{
	Q_D(MatPlotWidget);
	d->plot->setAxisAutoScale(QwtPlot::xBottom, false);
	d->plot->setAxisScale(QwtPlot::xBottom, lowerx, upperx);
	d->plot->setAxisAutoScale(QwtPlot::yLeft, false);
	d->plot->setAxisScale(QwtPlot::yLeft, lowery, uppery);
	d->plot->replot();
}

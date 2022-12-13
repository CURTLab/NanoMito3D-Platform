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

#ifndef HISTOGRAMWIDGET_H
#define HISTOGRAMWIDGET_H

#include <QWidget>

class MatPlotWidgetPrivate;

class MatPlotWidget : public QWidget
{
	Q_DECLARE_PRIVATE_D(m_d, MatPlotWidget)
public:
	MatPlotWidget(QWidget *parent = nullptr);

	void hist(const QVector<double> &values, double min, double max, int nbins = 100);
	void bar(const QStringList &values, const QVector<double> &height, const QVector<QColor> &colors, double width = 0.8);

	void replot();
	void clear();

	void setTitle(const QString &title);
	void setXLabel(const QString &label);
	void setYLabel(const QString &label);

	void setXScale(double min, double max);
	void setYScale(double min, double max);
	void setLimits(double lowerx, double upperx, double lowery, double uppery);

	std::tuple<double,double> xScale() const;
	std::tuple<double,double> yScale() const;

	// annotations
	void addText(const QPointF &postion, const QString &text, QColor color = Qt::black, Qt::Alignment alignment = Qt::AlignCenter);
	void addVLine(qreal position, QColor color, qreal width = 1.0);

private:
	std::unique_ptr<MatPlotWidgetPrivate> const m_d;

};

#endif // HISTOGRAMWIDGET_H

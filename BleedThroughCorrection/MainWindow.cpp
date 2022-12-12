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

#include "MainWindow.h"
#include "ui_MainWindow.h"

#include <QDebug>
#include <QMessageBox>
#include <QFileDialog>
#include <QPainter>

#include <opencv2/ml.hpp>

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, m_ui(new Ui::MainWindow)
{
	m_ui->setupUi(this);

	/*connect(m_ui->spin_pen_width, qOverload<int>(&QSpinBox::valueChanged),
			  m_ui->preview,  &DrawingPlotWidget::setPaintToolWidth
			  );*/

	QPixmap pix(18, 18);
	pix.fill(Qt::green);

	QTreeWidgetItem *item = new QTreeWidgetItem;
	item->setIcon(0, QIcon(pix));
	item->setText(0, "Background");
	item->setIcon(1, QIcon(":/icons/visibility_dark.png"));

	m_ui->treeLabels->addTopLevelItem(item);
	m_ui->treeLabels->setCurrentItem(item);

	pix.fill(Qt::blue);
	item = new QTreeWidgetItem;
	item->setIcon(0, QIcon(pix));
	item->setText(0, "Foreground");
	item->setIcon(1, QIcon(":/icons/visibility_dark.png"));
	m_ui->treeLabels->addTopLevelItem(item);

	m_ui->treeLabels->resizeColumnToContents(0);
	m_ui->treeLabels->resizeColumnToContents(1);
	m_ui->treeLabels->setRootIsDecorated(false);

	QActionGroup *g = new QActionGroup(this);
	g->addAction(m_ui->actionBrush);
	g->addAction(m_ui->actionErase);
	g->setExclusive(true);

	connect(g, &QActionGroup::triggered, [this](QAction *a) {
		if (a == m_ui->actionErase)
			m_ui->preview->setPaintToolColor(Qt::transparent);
		else if (a == m_ui->actionBrush)
			m_ui->preview->setPaintToolColor(m_ui->treeLabels->currentItem()->icon(0).pixmap(1).toImage().pixelColor(0,0));
	});
	connect(m_ui->treeLabels, &QTreeWidget::currentItemChanged,
			  [this](QTreeWidgetItem *current, QTreeWidgetItem *) {
		if (m_ui->actionBrush->isChecked())
			m_ui->preview->setPaintToolColor(current->icon(0).pixmap(1).toImage().pixelColor(0,0));
	});

	m_ui->buttonTrain->setEnabled(false);
	m_ui->buttonSaveModel->setEnabled(false);
	m_ui->buttonApply->setEnabled(false);

	setWindowTitle("Bleed-Through Correction");
}

MainWindow::~MainWindow()
{
}

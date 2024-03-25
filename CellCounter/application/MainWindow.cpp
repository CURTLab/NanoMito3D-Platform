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
#include "./ui_MainWindow.h"

#include <opencv2/imgcodecs.hpp>
#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, m_ui(new Ui::MainWindow)
	, m_nms(0, 19)
	, m_bar(new QProgressBar)
{
	m_ui->setupUi(this);

	m_ui->splitter->setSizes({250, width()-250});
	m_ui->buttonCalculate->setEnabled(false);

	m_ui->statusbar->addPermanentWidget(m_bar);
	m_bar->setVisible(false);

	try {
		m_model.loadModel("CellCounterModel.onnx");
	} catch(std::exception &e) {
		QMessageBox::critical(nullptr, tr("Error"), tr("Could not load dnn model! Reason: %1").arg(e.what()));
		exit(-1);
	}

	connect(&m_model, &CellCounter::progressRangeChanged,
			  m_bar, &QProgressBar::setRange
			  );
	connect(&m_model, &CellCounter::progressValueChanged,
			  m_bar, &QProgressBar::setValue
			  );
	connect(m_ui->spinThreshold, &QDoubleSpinBox::editingFinished,
			  this, &MainWindow::countCells
			  );

	connect(m_ui->buttonSelectFile, &QAbstractButton::released, this, [this]() {
		QString fileName = QFileDialog::getOpenFileName(this, tr("Open image"), DEV_PATH "/examples", tr("Image (*.png *.jpg *.jpeg *.tif *.tiff)"));
		if (fileName.isEmpty())
			return;
		import(fileName);
	});

	connect(m_ui->buttonCalculate, &QAbstractButton::released, this, [this]() {
		m_ui->frame->setEnabled(false);
		m_bar->setVisible(true);
		try {
			m_model.predictAsync(m_image, m_ui->comboSubdiv->currentText().toInt(), m_ui->spinBatchSize->value(), m_ui->checkMultipass->isChecked());
		} catch(std::exception &e) {
			QMessageBox::critical(nullptr, tr("Prediction error"), tr("Reason: %1").arg(e.what()));
		}
	});

	connect(&m_model, &CellCounter::finished,
			  this, [this]() {
		m_bar->setVisible(false);

		const cv::Mat &output = m_model.result();
		if (output.empty()) {
			QMessageBox::critical(this, tr("Error"), tr("Could not process image!\nReason: %1").arg(m_model.lastError()));
			m_ui->frame->setEnabled(true);
			return;
		}

		m_ui->plot2->setImage(output);

		countCells();

		m_ui->frame->setEnabled(true);
	});
}

MainWindow::~MainWindow()
{
}

void MainWindow::import(const QString &fileName)
{
	cv::Mat image = cv::imread(fileName.toStdString(), cv::IMREAD_GRAYSCALE);
	if (image.empty())
		return;
	cv::transpose(image, m_image);

	m_ui->editFile->setText(fileName);

	m_fileName = QFileInfo(fileName).baseName();
	setWindowTitle("Cell Counter - " + m_fileName);

	m_ui->plot1->clear();
	m_ui->plot2->clear();

	m_ui->plot1->setImage(m_image);

	m_ui->histogram->clear();
	m_ui->spinCells->setValue(0);
	m_ui->buttonCalculate->setEnabled(true);
}

void MainWindow::countCells()
{
	if (m_model.result().empty())
		return;

	m_ui->plot1->clearAnnotation();
	m_ui->plot2->clearAnnotation();
	m_ui->histogram->clear();

	const int winSize = m_ui->spinWinSize->value();

	QVector<QPointF> points;
	QVector<double> strengths;
	const double threshold = m_ui->spinThreshold->value();

	m_nms.setWindowSize(winSize);
	m_nms.setBorder(0);

	for (const auto &f : m_nms.findAll(m_model.result())) {
		if (qFuzzyIsNull(f.val))
			continue;
		if (f.val >= threshold)
			points << QPointF(f.x, f.y);
		strengths << f.val;
	}

	m_ui->histogram->hist(strengths, 0.01, 1.0, 50);
	m_ui->histogram->addVLine(threshold, Qt::red);

	const double pixelSize_mm = m_ui->spinPixelSize->value() * 1E-3;
	const double area = (m_image.size[0] * m_image.size[1] * pixelSize_mm) * pixelSize_mm;
	qDebug() << points.size() / area << "cells/mm²";

	m_ui->spinCells->setValue(points.size());
	m_ui->spinCellsPer->setValue(points.size() / area);

	m_ui->plot1->addCircles(points, Qt::red, 7);
	m_ui->plot2->addCircles(points, Qt::red, 7);
}

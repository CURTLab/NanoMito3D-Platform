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

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, m_ui(new Ui::MainWindow)
{
	m_ui->setupUi(this);

	m_ui->splitter->setSizes({250, width()-250});

	connect(m_ui->buttonSelectFile, &QAbstractButton::released, this, [this]() {
		QString fileName = QFileDialog::getOpenFileName(this, tr("Open image"), DEV_PATH "/examples", tr("Image (*.png *.jpg *.jpeg *.tif *.tiff)"));
		if (fileName.isEmpty())
			return;
		import(fileName);
	});

}

MainWindow::~MainWindow()
{
}

void MainWindow::import(const QString &fileName)
{
	m_image = cv::imread(fileName.toStdString(), cv::IMREAD_GRAYSCALE);
	if (m_image.empty())
		return;
	m_ui->editFile->setText(fileName);

	m_fileName = QFileInfo(fileName).baseName();
	setWindowTitle("Cell Counter - " + m_fileName);

	m_ui->plot1->clear();
	m_ui->plot2->clear();

	m_ui->plot1->setImage(m_image);

	m_ui->histogram->clear();
	m_ui->spinCells->setValue(0);
	m_result = cv::Mat{};
}

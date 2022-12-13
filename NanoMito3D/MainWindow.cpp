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

#include "Version.h"

#include "Localizations.h"
#include "Octree.h"

#include "DensityFilter.h"
#include "Device.h"
#include "GaussianFilter.h"
#include "Rendering.h"
#include "LocalThreshold.h"
#include "Skeletonize3D.h"
#include "AnalyzeSkeleton.h"
#include "Segments.h"

#include <chrono>
#include <QDebug>

#include <QMessageBox>
#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, m_ui(new Ui::MainWindow)
	, m_bar(new QProgressBar)
{
	m_ui->setupUi(this);

	GPU::initGPU();

	m_ui->splitter->setSizes({250, width()-250});
	m_ui->plot->setTitle("Classification of voxels");
	m_ui->statusbar->addPermanentWidget(m_bar);
	m_bar->setVisible(false);

	// default settings
	m_ui->spinVoxelSizeWidth->setValue(85);
	m_ui->spinVoxelSizeHeight->setValue(85);
	m_ui->spinVoxelSizeDepth->setValue(25);
	m_ui->spinMaxPAX->setValue(100);
	m_ui->spinMaxPAY->setValue(100);
	m_ui->spinMaxPAZ->setValue(200);
	m_ui->spinRadius->setValue(250);
	m_ui->spinMinPoints->setValue(10);
	m_ui->spinGaussianSigma->setValue(1.5);
	m_ui->spinWindowSize->setValue(5);

	m_ui->buttonRender->setEnabled(false);
	m_ui->buttonAnalyse->setEnabled(false);
	m_ui->buttonClassify->setEnabled(false);

	setWindowTitle(tr("NanoMito3D r%1").arg(GIT_REVISION));

	m_analyis.loadModel(DEV_PATH "/examples/mitoTrainDataSet.csv");
	//m_analyis.loadModel(DEV_PATH "/examples/mito-model.json");

	connect(&m_analyis, &AnalyzeMitochondria::progressRangeChanged,
			  m_bar, &QProgressBar::setRange);

	connect(&m_analyis, &AnalyzeMitochondria::progressChanged,
			  m_bar, &QProgressBar::setValue);

	connect(m_ui->buttonSelectFile, &QAbstractButton::released,
			  this, [this]() {
		m_bar->setVisible(false);
		QString fileName = QFileDialog::getOpenFileName(this, "Open localization file", DEV_PATH "/examples", "TSF File (*.tsf)");
		if (!fileName.isEmpty()) {
			m_ui->statusbar->showMessage(tr("Load %1").arg(QFileInfo(fileName).fileName()));
			m_ui->frame->setEnabled(false);
			m_analyis.load(fileName);
		}
	});

	connect(&m_analyis, &AnalyzeMitochondria::error, this, [this](QString title, QString errorMessage){
		m_ui->frame->setEnabled(true);
		QMessageBox::critical(this, title, errorMessage);
			  });

	connect(&m_analyis, &AnalyzeMitochondria::localizationsLoaded, this, [this]() {
		m_ui->buttonRender->setEnabled(true);
		m_ui->buttonAnalyse->setEnabled(false);
		m_ui->editFile->setText(m_analyis.fileName());
		m_ui->statusbar->showMessage(tr("Loaded %1 localizations form %2 successfully!")
											  .arg(m_analyis.localizations().size())
											  .arg(QFileInfo(m_analyis.fileName()).fileName()));
		if (m_analyis.availableChannels() > 1) {
			m_ui->comboChannel->clear();
			m_ui->comboChannel->addItem(tr("All channels"));
			for (int i = 0; i < m_analyis.availableChannels(); ++i)
				m_ui->comboChannel->addItem(tr("Channel %1").arg(i+1));
			m_ui->comboChannel->setEnabled(true);
		} else {
			m_ui->comboChannel->setCurrentIndex(0);
			m_ui->comboChannel->setEnabled(false);
		}
		m_ui->frame->setEnabled(true);
	});

	connect(m_ui->buttonRender, &QAbstractButton::released,
			  this, [this]() {
		m_ui->frame->setEnabled(false);
		m_bar->setVisible(false);
		m_ui->statusbar->showMessage(tr("Start rendering volume"));
		m_analyis.render(voxelSize(), maxPA(), m_ui->spinWindowSize->value(), m_ui->comboChannel->currentIndex(),
							  m_ui->groupDensityFilter->isChecked(), m_ui->spinMinPoints->value(), m_ui->spinRadius->value(),
							  m_ui->checkUseGPU->isChecked());
	});

	connect(&m_analyis, &AnalyzeMitochondria::volumeRendered, this, [this]() {
		m_ui->buttonRender->setEnabled(true);
		m_ui->buttonAnalyse->setEnabled(true);
		m_ui->statusbar->showMessage(tr("Rendered volume successfully!"));
		m_ui->volumeView->clear();
		m_ui->volumeView->setVolume(m_analyis.volume(), {0, 0, 1, 255});
		m_ui->frame->setEnabled(true);
	});

	connect(m_ui->buttonAnalyse, &QAbstractButton::released,
			  this, [this]() {
		m_ui->frame->setEnabled(false);
		m_bar->setVisible(true);
		m_ui->statusbar->showMessage(tr("Analyzing volume ..."));
		m_analyis.analyze(m_ui->spinGaussianSigma->value(), (ThresholdMethods)m_ui->comboThreshold->currentIndex(), m_ui->checkUseGPU->isChecked());
	});

	connect(&m_analyis, &AnalyzeMitochondria::volumeAnalyzed, this, [this]() {
		m_ui->statusbar->showMessage(tr("Volume successfully analyzed!"));

		const auto &segments = m_analyis.segments();
		const float r = std::max({segments.volume.voxelSize()[0], segments.volume.voxelSize()[1], segments.volume.voxelSize()[2]});

		m_ui->buttonClassify->setEnabled(true);
		m_ui->volumeView->clear();
		m_ui->volumeView->setVolume(segments.volume, {0., 0., 1., 0.4});
		std::vector<std::array<float,3>> endPoints;
		for (const auto &s : segments) {
			//m_ui->volumeView->addGraph(s.graph, segments.volume, 0.5f * r, {0.f, 1.f, 0.f});
			for (const auto &p : s.endPoints)
				endPoints.push_back(p);
		}
		m_ui->volumeView->addSpheres(endPoints, 0.8f * r, {1.f,0.f,0.f});
		m_ui->frame->setEnabled(true);
	});

	connect(m_ui->buttonClassify, &QAbstractButton::released,
			  this, [this]() {
		m_ui->frame->setEnabled(false);
		m_ui->statusbar->showMessage(tr("Classify analyzed volume ..."));
		m_analyis.classify();
	});

	connect(&m_analyis, &AnalyzeMitochondria::volumeClassified, this, [this]() {
		m_ui->statusbar->showMessage(tr("Volume successfully classified!"));
		m_ui->volumeView->clear();
		m_ui->volumeView->addClassifiedVolume(m_analyis.classifiedVolume(), m_analyis.numClasses());

		auto bars = m_analyis.classificationResult();
		if (bars.size() == 3) {
			m_ui->plot->clear();
			m_ui->plot->bar({"Puncture", "Rod", "Network"}, bars, {QColor(0,255,0),QColor(64, 224, 208),QColor(0,0,255)});
			double max = 0;
			for (int i = 0; i < bars.size(); ++i) {
				max = std::max(max, bars[i]);
				m_ui->plot->addText({(double)i, bars[i]}, QString("%1%").arg(bars[i]*100.0, 0, 'f', 2), Qt::black, Qt::AlignHCenter|Qt::AlignTop);
			}
			m_ui->plot->setYScale(0, max + 0.1);
		} else {
			QMessageBox::critical(this, "Error", "Expected three classes as result!");
		}

		m_ui->frame->setEnabled(true);
	});
}

MainWindow::~MainWindow()
{
}

std::array<float, 3> MainWindow::voxelSize() const
{
	return {static_cast<float>(m_ui->spinVoxelSizeWidth->value()),
			  static_cast<float>(m_ui->spinVoxelSizeHeight->value()),
			  static_cast<float>(m_ui->spinVoxelSizeDepth->value())};
}

std::array<float, 3> MainWindow::maxPA() const
{
	return {static_cast<float>(m_ui->spinMaxPAX->value()),
			  static_cast<float>(m_ui->spinMaxPAY->value()),
			  static_cast<float>(m_ui->spinMaxPAZ->value())};
}

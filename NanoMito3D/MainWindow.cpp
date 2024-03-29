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

#include "MainWindow.h"
#include "ui_MainWindow.h"

#include "Version.h"
#include "Localizations.h"
#include "Segments.h"

#include <QDebug>

#include <QMessageBox>
#include <QFileDialog>
#include <QColorDialog>

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, m_ui(new Ui::MainWindow)
	, m_bar(new QProgressBar)
{
	m_ui->setupUi(this);

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

	m_ui->actionExportVolume->setEnabled(false);
	m_ui->actionExportFilteredVolume->setEnabled(false);
	m_ui->actionExportSkeleton->setEnabled(false);
	m_ui->actionExportSegmentation->setEnabled(false);

	m_ui->comboThreshold->addItem("Local IsoData", QVariant::fromValue(ThresholdMethods::LocalISOData));
	m_ui->comboThreshold->addItem("Local Otsu", QVariant::fromValue(ThresholdMethods::LocalOtsu));

#ifndef CUDA_SUPPORT
	m_ui->checkUseGPU->setVisible(false);
#endif // CUDA_SUPPORT

	setWindowTitle("NanoMito3D " APP_VERSION);

	const QString modelFile = "mitoTrainDataSet.csv";
	if (QFile(modelFile).exists()) {
		if (m_analyis.loadModel(modelFile))
			m_ui->editModel->setText(QFileInfo(modelFile).absoluteFilePath());
	}

	connect(m_ui->actionExportVolume, &QAction::triggered,
			this, [this]() {
		exportVolumeDialog(m_analyis.volume(), "volume");
	});

	connect(m_ui->actionExportFilteredVolume, &QAction::triggered,
			this, [this]() {
		exportVolumeDialog(m_analyis.filteredVolume(), "filtered volume");
	});

	connect(m_ui->actionExportSkeleton, &QAction::triggered,
			  this, [this]() {
		exportVolumeDialog(m_analyis.skeleton(), "skeleton volume");
	});

	connect(m_ui->actionExportSegmentation, &QAction::triggered,
			this, &MainWindow::exportSegmentation);

	connect(m_ui->actionExportRenderer, &QAction::triggered,
			  this, [this]() {
		QFileInfo fi(m_currentFile);
		QString fileName = fi.absoluteFilePath() + "/" + fi.baseName() + ".png";
		fileName = QFileDialog::getSaveFileName(this, tr("Export Renderer"), fileName, "Image (*.png)");
		if (fileName.isEmpty())
			return;
		m_ui->volumeView->saveAsPNG(fileName);
	});

	connect(m_ui->actionSetBackgroundColor, &QAction::triggered, this, [this]() {
		QColor color = QColorDialog::getColor(m_ui->volumeView->backgroundColor(), this, "Set background color");
		if (color.isValid())
			m_ui->volumeView->setBackgroundColor(color);
	});

	connect(m_ui->actionQuit, &QAction::triggered, this, &MainWindow::close);

	connect(&m_analyis, &AnalyzeMitochondria::progressRangeChanged,
			  m_bar, &QProgressBar::setRange);

	connect(&m_analyis, &AnalyzeMitochondria::progressChanged,
			  m_bar, &QProgressBar::setValue);

	connect(&m_analyis, &AnalyzeMitochondria::error, this, [this](QString title, QString errorMessage){
		m_ui->frame->setEnabled(true);
		QMessageBox::critical(this, title, errorMessage);
			  });

#ifndef RELEASE_VERSION
	m_currentFile = DEV_PATH "examples/";
#endif

	// localizations loading section
	connect(m_ui->buttonSelectFile, &QAbstractButton::released,
			this, [this]() {
		m_bar->setVisible(false);
		QString fileName = QFileDialog::getOpenFileName(this, "Open localization file", m_currentFile, "TSF file (*.tsf)");
		if (!fileName.isEmpty()) {
			m_ui->statusbar->showMessage(tr("Load %1").arg(QFileInfo(fileName).fileName()));
			m_ui->frame->setEnabled(false);

			m_ui->buttonRender->setEnabled(false);
			m_ui->buttonAnalyse->setEnabled(false);
			m_ui->buttonClassify->setEnabled(false);

			m_ui->actionExportVolume->setEnabled(false);
			m_ui->actionExportFilteredVolume->setEnabled(false);
			m_ui->actionExportSkeleton->setEnabled(false);

			m_bar->setVisible(true);
			m_analyis.load(fileName);
			m_currentFile = fileName;
		}
	});

	// model selection
	connect(m_ui->buttonSelectModel, &QAbstractButton::released,
			this, [this]() {
		QString fileName = QFileDialog::getOpenFileName(this, "Open model file", m_ui->editModel->text(), "Model file (*.json *.csv)");
		if (fileName.isEmpty())
			return;
		if (m_analyis.loadModel(fileName))
			m_ui->editModel->setText(QFileInfo(fileName).absoluteFilePath());
	});

	connect(&m_analyis, &AnalyzeMitochondria::localizationsLoaded, this, [this]() {
		// reset UI
		m_bar->setVisible(false);
		m_ui->actionExportVolume->setEnabled(false);
		m_ui->actionExportFilteredVolume->setEnabled(false);
		m_ui->actionExportSegmentation->setEnabled(false);
		m_ui->actionExportFilteredVolume->setEnabled(false);
		m_ui->buttonRender->setEnabled(true);
		m_ui->buttonAnalyse->setEnabled(false);
		m_ui->editFile->setText(m_analyis.fileName());
		m_ui->statusbar->showMessage(tr("Loaded %1 localizations form %2 successfully!")
											  .arg(m_analyis.localizations().size())
											  .arg(QFileInfo(m_analyis.fileName()).fileName()));

		// Channel colors: Red, Blue, Green
		std::array<double,3> colors[] = {{1, 0, 0}, {0, 0, 1}, {0, 1, 0}};
		const auto nColors = sizeof(colors) / sizeof(colors[0]);

		if (m_analyis.availableChannels() > 1) {
			m_ui->comboChannel->clear();
			m_ui->comboChannel->addItem(tr("All channels"));
			for (int i = 0; i < m_analyis.availableChannels(); ++i) {
				QPixmap pix(18, 18);
				pix.fill(QColor::fromRgbF(colors[i][0], colors[i][1], colors[i][2]));
				m_ui->comboChannel->addItem(QIcon(pix), tr("Channel %1").arg(i+1));
			}
			m_ui->comboChannel->setEnabled(true);
		} else {
			m_ui->comboChannel->setCurrentIndex(0);
			m_ui->comboChannel->setEnabled(false);
		}
		m_ui->frame->setEnabled(true);

		m_ui->volumeView->clear();
		for (int channel = 1; channel <= m_analyis.localizations().channels(); ++channel) {
			m_ui->volumeView->addLocalizations(m_analyis.localizations(), 1.2f, colors[(channel - 1) % nColors], channel);
		}
		m_ui->volumeView->resetCamera();

		m_ui->plot->clear();
	});

	// volume rendering section
	connect(m_ui->buttonRender, &QAbstractButton::released,
			  this, [this]() {
		m_ui->frame->setEnabled(false);
		m_bar->setVisible(true);
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
		m_ui->volumeView->addVolume(m_analyis.volume(), {0, 0, 1, 255});
		m_ui->frame->setEnabled(true);
		m_ui->actionExportVolume->setEnabled(true);
		m_bar->setVisible(false);
	});

	// analyse volume section
	connect(m_ui->buttonAnalyse, &QAbstractButton::released,
			  this, [this]() {
		m_ui->frame->setEnabled(false);
		m_bar->setVisible(true);
		m_ui->statusbar->showMessage(tr("Analyzing volume ..."));
		m_analyis.analyze(m_ui->spinGaussianSigma->value(),  m_ui->comboThreshold->currentData().value<ThresholdMethods>(), m_ui->checkUseGPU->isChecked());
	});

	connect(&m_analyis, &AnalyzeMitochondria::volumeAnalyzed, this, [this]() {
		m_ui->statusbar->showMessage(tr("Volume successfully analyzed!"));

		const auto &segments = m_analyis.segments();
		const float r = 200.f;//std::max({segments.volume.voxelSize()[0], segments.volume.voxelSize()[1], segments.volume.voxelSize()[2]});

		m_ui->buttonClassify->setEnabled(true);
		m_ui->actionExportSegmentation->setEnabled(true);
		m_ui->volumeView->clear();
		m_ui->volumeView->addVolume(segments.volume, {0., 0., 1., 0.4});
		std::vector<std::array<float,3>> endPoints;
		for (const auto &s : segments) {
			m_ui->volumeView->addGraph(s->graph, segments.volume, 0.5f * r, {0.f, 1.f, 0.f});
			for (const auto &p : s->endPoints)
				endPoints.push_back(p);
		}
		m_ui->volumeView->addSpheres(endPoints, 0.8f * r, {1.f,0.f,0.f});
		m_ui->actionExportFilteredVolume->setEnabled(true);
		m_ui->actionExportSkeleton->setEnabled(true);
		m_ui->frame->setEnabled(true);
	});

	// classify segments section
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
		m_ui->actionExportSegmentation->setEnabled(true);

		// text color
		QColor textColor = palette().color(QPalette::WindowText);

		auto bars = m_analyis.classificationResult();
		if (bars.size() == 3) {
			m_ui->plot->clear();
			m_ui->plot->bar({"Puncture", "Rod", "Network"}, bars, {QColor(0,255,0),QColor(64, 224, 208),QColor(0,0,255)});
			double max = 0;
			for (int i = 0; i < bars.size(); ++i) {
				max = std::max(max, bars[i]);
				m_ui->plot->addText({(double)i, bars[i]}, QString("%1%").arg(bars[i]*100.0, 0, 'f', 2), textColor, Qt::AlignHCenter|Qt::AlignTop);
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

void MainWindow::exportSegmentation()
{
	QFileInfo fi(m_currentFile);
	QString fileName = fi.absoluteFilePath() + "/" + fi.baseName() + ".csv";
	fileName = QFileDialog::getSaveFileName(this, tr("Export Segmentation"), fileName, "CSV file (*.csv)");
	if (fileName.isEmpty())
		return;
	QFile file(fileName);
	if (!file.open(QIODevice::WriteOnly)) {
		QMessageBox::critical(this, tr("Error"), tr("Could not export segmentation!"));
		return;
	}

	QTextStream stream(&file);
	stream << "id;class;numBranches;numEndPoints;numJunctionVoxels;numJunctions;numSlabs;numTriples;numQuadruples;"
			  "averageBranchLength;maximumBranchLength;shortestPath;voxels;width;height;depth;signalCount\n";
	for (const auto &s : m_analyis.segments()) {
		stream << s->id << ';' << s->prediction << ";" << s->data.numBranches << ";" << s->data.numEndPoints << ";"
				  << s->data.numJunctionVoxels << ";" << s->data.numJunctions << ";" << s->data.numSlabs << ";"
				  << s->data.numTriples << ";" << s->data.numQuadruples << ";" << s->data.averageBranchLength << ";"
				  << s->data.maximumBranchLength << ";" << s->data.shortestPath << ";" << s->data.voxels << ";"
				  << s->data.width << ";" << s->data.height << ";" << s->data.depth << ";" << s->data.signalCount << "\n";
	}
}

void MainWindow::exportVolumeDialog(const Volume &volume, const QString &name)
{
	QFileInfo fi(m_currentFile);
	QString fileName = fi.absoluteFilePath() + "/" + fi.baseName() + ".tif";
	fileName = QFileDialog::getSaveFileName(this, tr("Export %1").arg(name), fileName, "TIF Stack (*.tif)");
	if (fileName.isEmpty())
		return;
	try {
		volume.saveTif(fileName.toStdString());
		m_ui->statusbar->showMessage(tr("Exported %1 to '%2'").arg(name, fileName));
	} catch(std::runtime_error &e) {
		QMessageBox::critical(this, tr("Error"), tr("Could not export %1 '%2'. Reason: %3").arg(name, fileName, e.what()));
	}
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

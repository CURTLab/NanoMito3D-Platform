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

#include "Rendering.h"

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, m_ui(new Ui::MainWindow)
	, m_bar(new QProgressBar)
{
	m_ui->setupUi(this);

	m_ui->statusbar->addPermanentWidget(m_bar);
	m_bar->setVisible(false);
	m_ui->preview->setColorMap(ColorMap::Hot);

	m_ui->splitter->setSizes({250, width()-250});

	connect(m_ui->spinPenWidth, qOverload<int>(&QSpinBox::valueChanged),
			  m_ui->preview,  &DrawingImagePlotWidget::setPaintToolWidth
			  );

	QPixmap pix(18, 18);
	pix.fill(m_labelColors[0]);

	QTreeWidgetItem *item = new QTreeWidgetItem;
	item->setIcon(0, QIcon(pix));
	item->setText(0, tr("Bleed-Through Signal"));

	m_ui->treeLabels->addTopLevelItem(item);
	m_ui->treeLabels->setCurrentItem(item);

	pix.fill(m_labelColors[1]);
	item = new QTreeWidgetItem;
	item->setIcon(0, QIcon(pix));
	item->setText(0, tr("Correct Signal"));
	m_ui->treeLabels->addTopLevelItem(item);

	m_ui->preview->setPaintToolColor(m_labelColors[0]);
	m_ui->preview->setPaintToolWidth(25);

	m_ui->treeLabels->resizeColumnToContents(0);
	m_ui->treeLabels->setRootIsDecorated(false);

	connect(m_ui->treeLabels, &QTreeWidget::currentItemChanged,
			  [this](QTreeWidgetItem *current, QTreeWidgetItem *) {
		if (m_ui->actionBrush->isChecked())
			m_ui->preview->setPaintToolColor(current->icon(0).pixmap(1).toImage().pixelColor(0,0));
	});

	connect(m_ui->comboMode,  qOverload<int>(&QComboBox::currentIndexChanged),
			  this, [this](int index) {
		if (index == 0) {
			m_ui->preview->setPaintToolColor(m_ui->treeLabels->currentItem()->icon(0).pixmap(1).toImage().pixelColor(0,0));
		} else if (index == 1) {
			m_ui->preview->setPaintToolColor(Qt::transparent);
		}
	});

	setWindowTitle("Bleed-Through Correction");

	m_ui->buttonRender->setEnabled(false);
	m_ui->buttonCorrect->setEnabled(false);

	connect(m_ui->buttonSelectFile, &QAbstractButton::clicked,
			  this, [this]() {
		m_bar->setVisible(false);
		QString fileName = QFileDialog::getOpenFileName(this, "Open localization file", DEV_PATH "/examples", "TSF File (*.tsf)");
		if (!fileName.isEmpty()) {
			m_ui->statusbar->showMessage(tr("Load %1").arg(QFileInfo(fileName).fileName()));
			m_ui->frame->setEnabled(false);
			m_bar->setVisible(true);
			m_correction.load(fileName);
		}
	});

	connect(&m_correction, &Correction::progressRangeChanged,
			  m_bar, &QProgressBar::setRange);

	connect(&m_correction, &Correction::progressChanged,
			  m_bar, &QProgressBar::setValue);

	connect(&m_correction, &Correction::error, this, [this](QString title, QString errorMessage){
		m_ui->frame->setEnabled(true);
		QMessageBox::critical(this, title, errorMessage);
			  });

	connect(&m_correction, &Correction::localizationsLoaded, this, [this]() {
		m_bar->setVisible(false);
		m_ui->editFile->setText(m_correction.fileName());
		m_ui->statusbar->showMessage(tr("Loaded %1 localizations form %2 successfully!")
											  .arg(m_correction.localizations().size())
											  .arg(QFileInfo(m_correction.fileName()).fileName()));
		if (m_correction.availableChannels() > 1) {
			m_ui->comboChannel->clear();
			for (int i = 0; i < m_correction.availableChannels(); ++i)
				m_ui->comboChannel->addItem(tr("Channel %1").arg(i+1));
			m_ui->comboChannel->setEnabled(true);
			m_ui->buttonRender->setEnabled(true);
		} else {
			QMessageBox::critical(this, tr("Error"), tr("Localization file only contains one color channel!"));
		}
		m_ui->frame->setEnabled(true);
	});

	connect(m_ui->buttonRender, &QAbstractButton::clicked,
			  this, [this]() {
		m_renderSize = m_ui->spinRenderSize->value();
		m_currentChannel = m_ui->comboChannel->currentIndex() + 1;
		cv::Mat image;
		Rendering::histogram2D(m_correction.localizations(), image, m_renderSize, [this](int, const Localization &l) {
			return (l.PAx > 75.f || l.PAy > 75.f || l.channel != m_currentChannel);
		});
		m_ui->preview->setImage(image);
		m_ui->buttonCorrect->setEnabled(true);
	});

	connect(m_ui->buttonCorrect, &QAbstractButton::clicked,
			  this, [this]() {
		m_ui->frame->setEnabled(false);
		m_correction.correct(m_ui->preview->paintOverlay(), m_renderSize, m_labelColors, m_currentChannel);
	});

	connect(&m_correction, &Correction::correctionFinished, this, [this]() {
		cv::Mat image;
		Rendering::histogram2D(m_correction.correctedLocalizations(), image, m_renderSize, [this](int, const Localization &l) {
			return (l.PAx > 75.f || l.PAy > 75.f || l.channel != m_currentChannel);
		});
		m_ui->preview->setImage(image);
		m_ui->frame->setEnabled(true);
	});

}

MainWindow::~MainWindow()
{
}

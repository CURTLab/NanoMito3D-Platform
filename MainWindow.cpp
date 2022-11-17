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
#include "Localizations.h"

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent)
	, m_ui(new Ui::MainWindow)
{
	m_ui->setupUi(this);

#ifdef QT_DEBUG
	try {
		Localizations locs;
		locs.load(DEV_PATH "/examples/WOP_CD62p_AntiMito_C1000_dual_dSTORM_red_blue_034_v3.tsf");
		m_ui->statusbar->showMessage(tr("Loaded %1 localizations from file.").arg(locs.size()));
	} catch(std::exception &e) {
		m_ui->statusbar->showMessage(tr("Error: ") + e.what());
	}

#endif
}

MainWindow::~MainWindow()
{
}

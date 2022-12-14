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

#include "Correction.h"

#include <QThreadPool>
#include <QDebug>

Correction::Correction(QObject *parent)
	: QObject{parent}
{

}

void Correction::load(const QString &fileName, bool threaded)
{
	auto func = [this,fileName]() {
		try {
			uint32_t nLocs = 0;
			m_locs.load(fileName.toStdString(), [this,&nLocs](uint32_t i, uint32_t n, const Localization &) {
				if (nLocs != n) {
					nLocs = n;
					emit progressRangeChanged(0, static_cast<int>(n)-1);
				}
				if (i % 100 == 1)
					emit progressChanged(i);
			});
			emit progressChanged(static_cast<int>(nLocs)-1);
			m_fileName = fileName;
			emit localizationsLoaded();
		} catch(std::exception &e) {
			qCritical().nospace() << tr("Correction::load Error: ") + e.what();
			emit error(tr("Load localizations error"), e.what());
		}
	};

	if (threaded)
		QThreadPool::globalInstance()->start(func);
	else
		func();
}

int Correction::availableChannels() const
{
	return m_locs.channels();
}

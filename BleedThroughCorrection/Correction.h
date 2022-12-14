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

#ifndef CORRECTION_H
#define CORRECTION_H

#include <QObject>

#include "Localizations.h"

class Correction : public QObject
{
	Q_OBJECT
public:
	explicit Correction(QObject *parent = nullptr);

	// load localization file synchron or asynchron (threaded)
	void load(const QString &fileName, bool threaded = true);

	int availableChannels() const;

	inline constexpr const QString &fileName() const { return m_fileName; }
	inline constexpr const Localizations &localizations() const { return m_locs; }

signals:
	void localizationsLoaded();

	void error(QString title, QString errorMessage);
	void progressRangeChanged(int min, int max);
	void progressChanged(int value);

private:
	QString m_fileName;
	Localizations m_locs;

};

#endif // CORRECTION_H

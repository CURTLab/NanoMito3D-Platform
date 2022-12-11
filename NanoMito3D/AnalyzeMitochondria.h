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

#ifndef ANALYZEMITOCHONDRIA_H
#define ANALYZEMITOCHONDRIA_H

#include <QObject>

#include "Volume.h"
#include "Localizations.h"
#include "Segments.h"

enum class ThresholdMethods {
	LocalISOData,
	LocalOtsu
};

class AnalyzeMitochondria : public QObject
{
	Q_OBJECT
public:
	explicit AnalyzeMitochondria(QObject *parent = nullptr);

	// load localization file synchron or asynchron (threaded)
	void load(const QString &fileName, bool threaded = true);

	int availableChannels() const;

	// render the loaded localizations
	void render(std::array<float,3> voxelSize, std::array<float,3> maxPA, int windowSize, int channel, bool densityFilter, int minPts, float radius, bool useGPU, bool threaded = true);

	// analyze the volume rendered
	void analyze(float sigma, ThresholdMethods thresholdMethod, bool useGPU, bool threaded = true);

	inline constexpr const QString &fileName() const { return m_fileName; }
	inline constexpr const Localizations &localizations() const { return m_locs; }
	inline constexpr const Volume &volume() const { return m_volume; }
	inline constexpr const Segments &segments() const { return m_segments; }

signals:
	void localizationsLoaded();
	void volumeRendered();
	void volumeAnalyzed();

	void error(QString title, QString errorMessage);
	void progressRangeChanged(int min, int max);
	void progressChanged(int value);

private:
	QString m_fileName;
	Localizations m_locs;
	Volume m_volume;
	Volume m_skeleton;
	Segments m_segments;

};

#endif // ANALYZEMITOCHONDRIA_H
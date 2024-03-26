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
#include <QImage>

#include <opencv2/ml.hpp>

#include "Localizations.h"

class Correction : public QObject
{
	Q_OBJECT
public:
	explicit Correction(QObject *parent = nullptr);

	// load localization file synchron or asynchron (threaded)
	void load(const QString &fileName, bool threaded = true);

	void loadRawImageFromSingleStack(const QString &registrationPath, const QString &stackPath, bool threaded = true);
	void loadTwoRawImageStacks(const QString &channel1Path, const QString &channel2Path, bool threaded = true);

	void correct(QImage labeling, float renderSize, QVector<QColor> labelColors, int channel, int iterations, bool threaded = true);

	int availableChannels() const;

	inline constexpr const QString &fileName() const { return m_fileName; }
	inline constexpr const Localizations &localizations() const { return m_locs; }
	inline constexpr Localizations &correctedLocalizations() { return m_corrected; }

signals:
	void localizationsLoaded();
	void imageStacksLoaded();
	void correctionFinished();

	void error(QString title, QString errorMessage);
	void progressRangeChanged(int min, int max);
	void progressChanged(int value);
	void showMessage(QString massage);

private:
	static constexpr int CHANNELS = 2;

	struct Features {
		float intensity;
		float background;
		float PAx, PAy, PAz;
		float raw[9 * CHANNELS];
	};
	static constexpr const char *featureNames[] = {"Intensity", "Background", "PAX", "PAY", "PAZ"};

	void extractFeatures(const Localization &l, Features &f) const;
	void extractRawImageFeatures(const Localization &l, Features &f) const;

	QString m_fileName;
	Localizations m_locs;
	Localizations m_corrected;
	std::vector<cv::Mat> m_rawImageStack[CHANNELS];
	int m_rawImageWidth;
	int m_rawImageHeight;

};

#endif // CORRECTION_H

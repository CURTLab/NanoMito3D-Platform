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

#include <opencv2/ml.hpp>

#include "Volume.h"
#include "Localizations.h"
#include "Segments.h"
#include "AnalyzeSkeleton.h"

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

	// analyze skeleton
	void analyzeSkeleton(Volume filteredVolume, Volume skeleton, bool useGPU, bool threaded = true);

	// classify the analyzed volume
	void classify(bool threaded = true);

	// load model eighter from csv training dataset or pretrained json model file
	void loadModel(const QString &fileName);

	inline constexpr const QString &fileName() const { return m_fileName; }
	inline constexpr const Localizations &localizations() const { return m_locs; }
	inline constexpr Localizations &localizations() { return m_locs; }
	inline constexpr const Volume &volume() const { return m_volume; }
	inline constexpr const Segments &segments() const { return m_segments; }
	inline constexpr const Volume &filteredVolume() const { return m_filteredVolume; }
	inline constexpr const Volume &skeleton() const { return m_skeleton; }
	inline constexpr const Volume &classifiedVolume() const { return m_classifiedVolume; }
	inline constexpr const GenericVolume<uint32_t> &hist() const { return m_hist; }
	inline constexpr const int numClasses() const { return m_numClasses; }
	inline constexpr const QVector<double> &classificationResult() const { return m_classificationResult; }

	inline void setVolume(Volume volume) { m_volume = volume; }

signals:
	void localizationsLoaded();
	void volumeRendered();
	void volumeAnalyzed();
	void volumeClassified();

	void error(QString title, QString errorMessage);
	void progressRangeChanged(int min, int max);
	void progressChanged(int value);

private:
	QString m_fileName;
	Localizations m_locs;
	Volume m_volume;
	Volume m_filteredVolume;
	Volume m_skeleton;
	Segments m_segments;
	cv::Ptr<cv::ml::RTrees> m_dtree;
	GenericVolume<int> m_labeledVolume;
	GenericVolume<uint32_t> m_hist;
	Volume m_classifiedVolume;
	int m_numClasses;
	QVector<double> m_classificationResult;

};

#endif // ANALYZEMITOCHONDRIA_H

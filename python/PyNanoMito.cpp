#include <Python.h>

#include <iostream>
#include <array>
#include <unordered_map>

#include "Version.h"
#include "Localizations.h"
#include "DensityFilter.h"
#include "Rendering.h"
#include "GaussianFilter.h"
#include "LocalThreshold.h"
#include "Skeletonize3D.h"
#include "AnalyzeSkeleton.h"
#include "Segments.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

PyObject* createDataFrame(const std::unordered_map<std::string, std::vector<double>>& data) {
	// Extract column names and data
	std::vector<std::string> column_names;
	std::vector<std::vector<double>> columns_data;

	for (const auto& kv : data) {
		column_names.push_back(kv.first);
		columns_data.push_back(kv.second);
	}

	// Convert C++ data to Python lists
	PyObject* py_data = PyDict_New();
	for (size_t i = 0; i < column_names.size(); ++i) {
		PyObject* py_column_data = PyList_New(columns_data[i].size());
		for (size_t j = 0; j < columns_data[i].size(); ++j) {
			PyList_SetItem(py_column_data, j, PyFloat_FromDouble(columns_data[i][j]));
		}
		PyDict_SetItemString(py_data, column_names[i].c_str(), py_column_data);
	}

	// Import pandas module
	PyObject* pandas_module = PyImport_ImportModule("pandas");

	// Create DataFrame
	PyObject* df = PyObject_CallMethod(pandas_module, "DataFrame", "(O)", py_data);

	// Cleanup
	Py_DECREF(py_data);
	Py_DECREF(pandas_module);

	return df;
}

inline void array_cleanup(PyObject *capsule) {
	void *memory = PyCapsule_GetPointer(capsule, NULL);
	//std::cout << "Free array!" << std::endl;
	PySys_WriteStdout("Free array %x: %s!\n", memory, PyCapsule_GetName(capsule));
	free(memory);
}

PyObject*
PyNanoMito_version(PyObject *self, PyObject *args)
{
	return Py_BuildValue("s", "PyNanoMito " APP_VERSION);
}

PyObject*
PyNanoMito_segment(PyObject *self, PyObject* args, PyObject* kwds)
{
	static char* kwlist[] = { const_cast<char*>("file_name"),
							  const_cast<char*>("channel"),
							  const_cast<char*>("verbose"),
							  nullptr };

	int channel = 2;
	int verbose = 0;

	char *file_name;
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|Ip", kwlist, &file_name, &channel, &verbose)) {
		PyErr_Format(PyExc_TypeError, "file_name parameter required!");
		Py_RETURN_NONE;
	}

	std::array<float, 3> voxelSize({85.f, 85.f, 25.f});
	std::array<float, 3> maxPA({100.f, 100.f, 200.f});
	int windowSize = 5;
	int minPts = 10;
	float radius = 250;
	float sigma = 1.5f;

	if (verbose)
		std::cout << "FileName: " << file_name << std::endl;

	Localizations locs;
	Segments segments;

	try {
		locs.load(file_name, {});

		if (verbose)
			std::cout << "Num locs: " << locs.size() << std::endl;

		locs.erase(std::remove_if(locs.begin(), locs.end(), [&maxPA,channel](const Localization &l) {
			if (channel > 0 && l.channel != channel)
				return true;
			return (l.PAx > maxPA[0] || l.PAy > maxPA[1] || l.PAz > maxPA[2]);
		}), locs.end());

		if (verbose)
			std::cout << "Num locs (filter1): " << locs.size() << std::endl;

		locs.erase(DensityFilter::remove_gpu(locs, minPts, radius), locs.end());

		if (verbose)
			std::cout << "Num locs (filter2): " << locs.size() << std::endl;

		auto volume = Rendering::render_gpu(locs, voxelSize, windowSize);

		if (verbose)
			std::cout << "Volume: " << volume.width() << "x" << volume.height() << "x" << volume.depth() << std::endl;

		auto filteredVolume = Volume(volume.size(), volume.voxelSize(), volume.origin());

		std::array<float,3> sigmaScaled;
		sigmaScaled[0] = sigma * volume.voxelSize()[0] / volume.voxelSize()[0];
		sigmaScaled[1] = sigma * volume.voxelSize()[0] / volume.voxelSize()[1];
		sigmaScaled[2] = sigma * volume.voxelSize()[0] / volume.voxelSize()[2];

		windowSize = (int)(sigma * 4) | 1;
		GaussianFilter::gaussianFilter_gpu(volume.constData(), filteredVolume.data(), volume.width(), volume.height(), volume.depth(), windowSize, sigmaScaled);

		if (verbose)
			std::cout << "Volume (gaussian): " << filteredVolume.width() << "x" << filteredVolume.height() << "x" << filteredVolume.depth() << std::endl;

		LocalThreshold::localThrehsold_gpu(LocalThreshold::IsoData, filteredVolume, filteredVolume, 11);

		if (verbose)
			std::cout << "Volume (threshold): " << filteredVolume.width() << "x" << filteredVolume.height() << "x" << filteredVolume.depth() << std::endl;

		Volume skeleton;
		Skeleton3D::skeletonize(filteredVolume, skeleton);

		Skeleton3D::Analysis analysis;
		auto trees = analysis.calculate(skeleton, {}, Skeleton3D::Analysis::NoPruning, true, 0.0, true, false);

		GenericVolume<int> labeledVolume(volume.width(), volume.height(), volume.depth());
		std::fill_n(labeledVolume.data, volume.voxels(), 0);

		Volume segmentedVolume(volume.size(), volume.voxelSize(), volume.origin());
		segmentedVolume.fill(0);

		int id = 1;
		for (int i = 0; i < trees.size(); ++i) {
			const auto &t = trees[i];

			auto [segment,voxels,box] = trees.extractVolume(filteredVolume, 1, i);

			if ((box.width() <= 1) || (box.height() <= 1) || (box.depth() <= 1) || (voxels < 50)) {
				continue;
			}

			// draw segment to new volume and count SMLM signals
			uint32_t signalCount1 = 0;
			box.forEachVoxel([&](int x, int y, int z) {
				if (segment(x, y, z)) {
					segmentedVolume(x, y, z) = 255;
					labeledVolume(x, y, z) = id;
				}
			});

			auto s = std::make_shared<Segment>();
			s->id = id++;
			s->boundingBox = box;
			s->graph = t.graph;

			// fill segment
			s->data.numBranches = t.numberOfBranches;
			s->data.numEndPoints = t.numberOfEndPoints;
			s->data.numJunctionVoxels = t.numberOfJunctionVoxels;
			s->data.numJunctions = t.numberOfJunctions;
			s->data.numSlabs = t.numberOfSlabs;
			s->data.numTriples = t.numberOfTriplePoints;
			s->data.numQuadruples = t.numberOfQuadruplePoints;
			s->data.averageBranchLength = t.averageBranchLength;
			s->data.maximumBranchLength = t.maximumBranchLength;
			s->data.shortestPath = t.shortestPath;
			s->data.voxels = voxels;
			// add 1 since bounding box calculates (max-min)
			s->data.width = box.width() + 1;
			s->data.height = box.height() + 1;
			s->data.depth = box.depth() + 1;
			s->data.signalCount = signalCount1;

			for (const auto &p : t.endPoints)
				s->endPoints.push_back(skeleton.mapVoxel(p.x, p.y, p.z, true));

			s->vol = segment;

			segments.push_back(s);
		}

		if (verbose)
			std::cout << "Segments: " << segments.size() << std::endl;

	} catch(std::exception &e) {
		PyErr_Format(PyExc_TypeError, e.what());
		Py_RETURN_NONE;
	}

	const char *column_names[] = {"numBranches", "numEndPoints", "numJunctionVoxels", "numJunctions", "numSlabs", "numTriples",
								 "numQuadruples", "averageBranchLength", "maximumBranchLength", "shortestPath", "voxels",
								 "width", "height", "depth"};
	PyObject* py_data = PyDict_New();
	for (size_t i = 0; i < sizeof(column_names)/sizeof(column_names[0]); ++i) {
		PyObject* py_column_data = PyList_New(segments.size());
		for (size_t j = 0; j < segments.size(); ++j) {
			PyList_SetItem(py_column_data, j, PyFloat_FromDouble(segments[j]->data.values[i]));
		}
		PyDict_SetItemString(py_data, column_names[i], py_column_data);
	}

	// Import pandas module
	PyObject* pandas_module = PyImport_ImportModule("pandas");

	// Create DataFrame
	PyObject* df = PyObject_CallMethod(pandas_module, "DataFrame", "(O)", py_data);

	// Cleanup
	Py_DECREF(py_data);
	Py_DECREF(pandas_module);

	return df;
}

PyMethodDef PyNanoMitoMethods[] = {
	{"version", PyNanoMito_version, METH_NOARGS,
	 "Return the version of PyNanoMito."},
	{"segment", (PyCFunction)PyNanoMito_segment, METH_VARARGS | METH_KEYWORDS,
	 "Returns pandas dataframe of segments."},
	{NULL, NULL, 0, NULL}        /* Sentinel */
};

// Module entry point Python 3
static PyModuleDef PyNanoMitoModuleDef = {
	PyModuleDef_HEAD_INIT,
	"",
	NULL,
	-1,
	PyNanoMitoMethods,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC
PyInit_pynanomito(void)
{
	PyObject* m = PyModule_Create(&PyNanoMitoModuleDef);

	// import numpy functionality
	import_array();

	return m;
}

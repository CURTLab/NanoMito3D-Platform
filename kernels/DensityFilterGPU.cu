#include <cuda.h>
#include <cuda_runtime.h>

#include "DensityFilter.h"

#include <memory>
#include <algorithm>

#include "cuNSearch.h"

#if 0
__device__ float sqr(float x) { return x*x; }

__global__ void filter_kernel(const Localization *dLocs, size_t nLocs, size_t offset, size_t n, uint8_t *dFiltered, int minPts, float r2)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		 return;

	const float x = dLocs[i + offset].x;
	const float y = dLocs[i + offset].y;
	const float z = dLocs[i + offset].z;

	size_t count = 0;
	for (size_t j = 0; j < nLocs; ++j) {
		const float dist = sqr(dLocs[j].x - x) + sqr(dLocs[j].y - y) + sqr(dLocs[j].z - z);
		count += size_t(dist < r2);
	}

	dFiltered[i + offset] = count < minPts;
}

// simple brute force density filter
Localizations::const_iterator DensityFilter::remove_gpu(Localizations &locs, int minPoints, float radius)
{
	std::unique_ptr<uint8_t[]> filtered(new uint8_t[locs.size()]);

	const size_t n = 32768; //65536;

	uint8_t *dFiltered = nullptr;
	cudaMalloc(&dFiltered, locs.size());

	dim3 numThreads(1024);
	dim3 numBlocks((n + numThreads.x - 1) / numThreads.x);
	// minPoints + 1 to account for the point itself

	for (int i = 0; i < locs.size(); i += n) {
		const size_t m = std::min(n, locs.size() - i);
		filter_kernel<<<numBlocks, numThreads>>>(locs.constData(DeviceType::Device), locs.size(), i, m, dFiltered, minPoints + 1, radius * radius);
		GPU::cudaCheckError();
		cudaDeviceSynchronize();
	}

	cudaMemcpy(filtered.get(), dFiltered, locs.size(), cudaMemcpyDeviceToHost);
	GPU::cudaCheckError();

	cudaFree(dFiltered);


	return std::remove_if(locs.begin(), locs.end(), [&filtered,&locs](const Localization &l) -> bool {
		const size_t idx = static_cast<size_t>(&l - locs.data());
		return filtered[idx];
	});
}
#else

Localizations::const_iterator DensityFilter::remove_gpu(Localizations &locs, int minPoints, float radius)
{
	const size_t nLocs = locs.size();

	float *pts = new float[nLocs * 3];
	for (size_t i = 0; i < nLocs; ++i) {
		pts[3*i + 0] = locs[i].x;
		pts[3*i + 1] = locs[i].y;
		pts[3*i + 2] = locs[i].z;
	}

	cuNSearch::NeighborhoodSearch nsearch(radius);

	//Add point set from the test data
	auto pointSetIndex = nsearch.add_point_set(pts, nLocs, false, true);
	nsearch.find_neighbors();

	auto &pointSet = nsearch.point_set(pointSetIndex);

	const auto ret = std::remove_if(locs.begin(), locs.end(), [&locs,&pointSet,minPoints](const Localization &l) -> bool {
		const size_t idx = static_cast<size_t>(&l - locs.data());
		return pointSet.n_neighbors(0, idx) < minPoints;
	});

	return ret;
}
#endif

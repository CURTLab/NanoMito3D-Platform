#include "../Device.h"

#include <stdexcept>

namespace GPU {

static bool IS_DEVICE_AVAILABLE = false;
static bool INITIALIZED = false;

}

void GPU::initGPU() {
	if (INITIALIZED)
		return;
	double hTmp = 1;
	double *dTmp;
	cudaMalloc(&dTmp, sizeof(double));
	cudaMemcpy(dTmp, &hTmp, sizeof(double), cudaMemcpyHostToDevice);
	cudaFree(dTmp);

	int nDevices = 0;
	cudaGetDeviceCount(&nDevices);
	IS_DEVICE_AVAILABLE = nDevices > 0;
}

bool GPU::isGPUAvailable()
{
	initGPU();
	return IS_DEVICE_AVAILABLE;
}

void GPU::cudaCheckError()
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
}

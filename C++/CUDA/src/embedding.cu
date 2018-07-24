#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void single_embedding_kernel(float* state_space, float* x, int emb_dim, int a, unsigned int tau)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = tid / emb_dim;
	int j = tid % emb_dim;
	state_space[tid] = x[(a - j) * tau + i];
}

__global__ void joint_embedding_kernel(float* state_space, float* x, float* y, int emb_dim, int a, unsigned int tau)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = tid / emb_dim;
	int j = tid % emb_dim;
	int time = (a - j) * tau + i;
	state_space[tid] = x[time] + y[time];
}

cudaError_t dev_alloc_timeseries(float** dev_x, float** dev_y, int dev_x_size) {
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)dev_x, dev_x_size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)dev_y, dev_x_size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	return cudaStatus;
}

cudaError_t dev_copy_timeseries(float* dev_x, float* dev_y, float* host_x, float* host_y, int n) {
	// Copy input vector from host memory to GPU buffers.
	// Leave the pitch with memory garbage
	cudaError_t cudaStatus;

	cudaStatus = cudaMemcpy(dev_x, host_x, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in dev_copy_timeseries!");
	}

	cudaStatus = cudaMemcpy(dev_y, host_y, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed in dev_copy_timeseries!");
	}

	return cudaStatus;
}

cudaError_t dev_alloc_manifolds(float** dev_state_space_X, float** dev_state_space_Y, float** dev_state_space_J, float** dev_state_space_Z, int dev_state_space_size) {
	// Allocate GPU buffers for the state space and time series
	// The kernel will over-index n, but that's not a problem, memory garbage can be used as pitch
	// This way branch divergence can not happen

	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)dev_state_space_X, dev_state_space_size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)dev_state_space_Y, dev_state_space_size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)dev_state_space_J, dev_state_space_size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)dev_state_space_Z, dev_state_space_size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	return cudaStatus;
}

cudaError_t embed_manifolds(float** dev_state_space_X,
	                        float** dev_state_space_Y,
	                        float** dev_state_space_J,
	                        float** dev_state_space_Z,
                            int& dev_state_space_size,
	                        float* host_x,
                            float* host_y,
	                        int n,
                            int& dev_x_size,
	                        int emb_dim,
	                        int tau,
	                        int num_threads_per_block) {
	// time-series are row-like in memory, therefore the block layout in the grid, and the thread layout in a block should be row-like as well for better memory coalesce

	cudaError_t cudaStatus;

	int a = emb_dim - 1;
	int offset = a * tau;
	int num_rows = n - offset;
	int num_elems = num_rows * emb_dim;

	int num_blocks = (num_elems + (num_threads_per_block - 1)) / num_threads_per_block;

	// Allocate GPU memory
	float* dev_x = 0;
	float* dev_y = 0;

	dev_state_space_size = num_blocks * num_threads_per_block;
	int i = dev_state_space_size / emb_dim;
	dev_x_size = offset + i;

	cudaStatus = dev_alloc_timeseries(&dev_x, &dev_y, dev_x_size);
	cudaStatus = dev_alloc_manifolds(dev_state_space_X, dev_state_space_Y, dev_state_space_J, dev_state_space_Z, dev_state_space_size);

	// Copy time-series to GPU memory
	cudaStatus = dev_copy_timeseries(dev_x, dev_y, host_x, host_y, n);

	// Launch the embedding kernels on the GPU
	single_embedding_kernel<<<num_blocks, num_threads_per_block>>>(*dev_state_space_X, dev_x, emb_dim, a, tau);
	single_embedding_kernel<<<num_blocks, num_threads_per_block>>>(*dev_state_space_Y, dev_y, emb_dim, a, tau);
	joint_embedding_kernel<<<num_blocks, num_threads_per_block>>>(*dev_state_space_J, dev_x, dev_y, emb_dim, a, tau);

	// Check for any errors launching the kernels
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "embedding kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching embedding_kernel!\n", cudaStatus);
	}

	cudaFree(dev_x);
	cudaFree(dev_y);
	return cudaStatus;
}




#include <iostream>
#include "embedding.cuh"

using namespace std;

bool floats_equal(double a, double b, double epsilon = 0.000001)
{
	return fabs(a - b) < epsilon;
}

int test_single_embedding() {
	bool match = true;

	float host_x[10] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
	float host_y[10] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };

	float expected_state_space[6 * 3] = {
		5.0, 3.0, 1.0,
		6.0, 4.0, 2.0,
		7.0, 5.0, 3.0,
		8.0, 6.0, 4.0,
		9.0, 7.0, 5.0,
		10.0, 8.0, 6.0 };

	float* dev_state_space_X = 0;
	float* dev_state_space_Y = 0;
	float* dev_state_space_J = 0;
	float* dev_state_space_Z = 0;

	int dev_state_space_size = 0;
	int dev_x_size = 0;

	cudaError_t cudaStatus;

	cudaStatus = embed_manifolds(&dev_state_space_X, &dev_state_space_Y, &dev_state_space_J, &dev_state_space_Z, dev_state_space_size, host_x, host_y, 10, dev_x_size, 3, 2, 5);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "embed_manifolds failed!");
		return 1;
	}

	// Copy output vector from GPU buffer to host memory.
	float* host_state_space_X = new float[6 * 3];

	cudaStatus = cudaMemcpy(host_state_space_X, dev_state_space_X, 18 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		match = false;
		goto Error;
	}

	for (int i = 0; i < 6*3; i++) {
		if (!floats_equal(expected_state_space[i], host_state_space_X[i])) {
			match = false;
			break;
		}
	}

Error:
	cudaFree(dev_state_space_X);
	cudaFree(dev_state_space_Y);
	cudaFree(dev_state_space_J);
	cudaFree(dev_state_space_Z);
	delete[] host_state_space_X;

	if (match) {
		return 0;
	}
	else {
		return 1;
	}
}

int test_joint_embedding() {
	bool match = true;

	float host_x[10] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
	float host_y[10] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };

	float expected_state_space[6 * 3] = {
		10.0, 6.0, 2.0,
		12.0, 8.0, 4.0,
		14.0, 10.0, 6.0,
		16.0, 12.0, 8.0,
		18.0, 14.0, 10.0,
		20.0, 16.0, 12.0 };

	float* dev_state_space_X = 0;
	float* dev_state_space_Y = 0;
	float* dev_state_space_J = 0;
	float* dev_state_space_Z = 0;

	int dev_state_space_size = 0;
	int dev_x_size = 0;

	cudaError_t cudaStatus;

	cudaStatus = embed_manifolds(&dev_state_space_X, &dev_state_space_Y, &dev_state_space_J, &dev_state_space_Z, dev_state_space_size, host_x, host_y, 10, dev_x_size, 3, 2, 5);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "embed_manifolds failed!");
		return 1;
	}

	// Copy output vector from GPU buffer to host memory.
	float* host_state_space_J = new float[6 * 3];

	cudaStatus = cudaMemcpy(host_state_space_J, dev_state_space_J, 18 * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		match = false;
		goto Error;
	}

	for (int i = 0; i < 6 * 3; i++) {
		if (!floats_equal(expected_state_space[i], host_state_space_J[i])) {
			match = false;
			break;
		}
	}

Error:
	cudaFree(dev_state_space_X);
	cudaFree(dev_state_space_Y);
	cudaFree(dev_state_space_J);
	cudaFree(dev_state_space_Z);
	delete[] host_state_space_J;

	if (match) {
		return 0;
	}
	else {
		return 1;
	}
}

int main() {
	cout << "Running unit tests..." << endl;

	cout << "Test single embedding failed: " << test_single_embedding() << endl;
	cout << "Test joint embedding failed: " << test_joint_embedding() << endl;

	// max data size given the row-like grid-block layout: (2^31-1) * 2^10

	int a = 0;
	cin >> a;
}

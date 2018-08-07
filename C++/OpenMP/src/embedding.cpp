#include <cmath>
#include <algorithm>
#include <iostream>
using namespace std;


double* embed(double* x, const unsigned int n, const unsigned int emb_dim, const unsigned int tau) {
	const int a = emb_dim - 1;
	const int num_rows = n - a * tau;
	
	double* state_space = new double[num_rows*emb_dim];
	
	int time_delay;
	
	#pragma omp parallel for collapse(2)
	for (int i=0; i<num_rows; i++) {
		for (int j=0; j<emb_dim; j++) {
			time_delay = (a - j) * tau;
			state_space[i*emb_dim + j] = x[time_delay + i];
		}
	}
	
	return state_space;
}

double** get_manifolds(double* x, double* y, const unsigned int n, const unsigned int emb_dim, const unsigned int tau, unsigned int downsample_rate){
	const int c = emb_dim - 1;
	const int num_rows = n - c * tau;
	const double a = sqrt(29.0 / 31.0);
	
	double *X, *Y, *J, *Z;
	
	#pragma omp parallel sections
	{
		#pragma omp section // embed manifold X
		X = embed(x, n, emb_dim, tau);
		
		#pragma omp section // embed manifold Y
		Y = embed(y, n, emb_dim, tau);
		
		#pragma omp section // embed manifold J
		{
			double* j = new double[n];

			#pragma omp parallel for simd
			for (int i = 0; i < n; i++) {
				j[i] = a*x[i] + y[i];
			}

			J = embed(j, n, emb_dim, tau);
			
			delete[] j;
		}
	}
	
	// embed manifold Z
	unsigned int* x_s = new unsigned int[num_rows];
	unsigned int* y_s = new unsigned int[num_rows];
	
	#pragma omp parallel for simd
	for (unsigned int i=0; i<num_rows; i++) {
		x_s[i] = i;
		y_s[i] = i;
	}

	random_shuffle(y_s, y_s+num_rows);
	
	Z = new double[num_rows*emb_dim];
	
	#pragma omp parallel for collapse(2)
	for (int i=0; i<num_rows; i++) {
		for (int j=0; j<emb_dim; j++) {
			Z[i*emb_dim + j] = a * X[ x_s[i]*emb_dim + j ] + Y[ y_s[i]*emb_dim + j ];
		}
	}
	
	delete[] x_s;
	delete[] y_s;
	
	// apply downsampling
	if (downsample_rate > 1) {
		double *X2, *Y2, *J2, *Z2;
		unsigned int downsampled_size = num_rows / downsample_rate;
		X2 = new double[downsampled_size * emb_dim];
		Y2 = new double[downsampled_size * emb_dim];
		J2 = new double[downsampled_size * emb_dim];
		Z2 = new double[downsampled_size * emb_dim];
		
		#pragma omp parallel for collapse(2)
		for (int i=0; i<downsampled_size; i++) {
			for (int j=0; j<emb_dim; j++) {
				X2[i*emb_dim + j] = X[(i*downsample_rate)*emb_dim + j];
				Y2[i*emb_dim + j] = Y[(i*downsample_rate)*emb_dim + j];
				J2[i*emb_dim + j] = J[(i*downsample_rate)*emb_dim + j];
				Z2[i*emb_dim + j] = Z[(i*downsample_rate)*emb_dim + j];
			}
		}
		
		delete[] X;
		delete[] Y;
		delete[] J;
		delete[] Z;
		
		X = X2;
		Y = Y2;
		J = J2;
		Z = Z2;
	}
	
	double** manifolds = new double*[4]{X, Y, J, Z};
	
	return manifolds;
}
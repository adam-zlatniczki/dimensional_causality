#include <iostream>
#include <fstream>
#include "causality.h"
#include "embedding.h"
#include "dimensions.h"
#include "trimming.h"
#include "probabilities.h"

using namespace std;

double* infer_causality(double* x, double* y, unsigned int n, unsigned int emb_dim, unsigned int tau, unsigned int* k_range, unsigned int len_range, double eps/*=0.05*/, double c/*=3.0*/, double bins/*=20.0*/, unsigned int downsample_rate/*=1*/){
	// embed manifolds
	double** manifolds = get_manifolds(x, y, n, emb_dim, tau, downsample_rate);
	double* X = manifolds[0];
	double* Y = manifolds[1];
	double* J = manifolds[2];
	double* Z = manifolds[3];
	
	// update n
	n -= (emb_dim - 1) * tau;
	if (downsample_rate > 1) n = n / downsample_rate;
	
	// calculate kNN distances for the whole k-range
	double** X_nn_distances = knn_distances(X, n, emb_dim, k_range[len_range-1]);
	double** Y_nn_distances = knn_distances(Y, n, emb_dim, k_range[len_range-1]);
	double** J_nn_distances = knn_distances(J, n, emb_dim, k_range[len_range-1]);
	double** Z_nn_distances = knn_distances(Z, n, emb_dim, k_range[len_range-1]);
	
	double** range_probabilities = new double*[len_range];
	
	// explore k-range
	unsigned int k;
	unsigned int trimmed_size;
	
	
	for (int i=0; i<len_range; i++) {
		k = k_range[i];
		
		// estimate local dimensions
		double* x_dims = local_dims(X_nn_distances, n, k);
		double* y_dims = local_dims(Y_nn_distances, n, k);
		double* j_dims = local_dims(J_nn_distances, n, k);
		double* z_dims = local_dims(Z_nn_distances, n, k);
		
		// trim dimension estimates
		double** trimmed_data = trim_data(x_dims, y_dims, j_dims, z_dims, n, trimmed_size, eps);
		
		// calculate case probabilities
		double eff_sample_size = 2 * k;
		range_probabilities[i] = get_probabilities(trimmed_data, trimmed_size, eff_sample_size);
		
		// clear trimmed data
		for (int j=0; j<4; j++) delete[] trimmed_data[j];
		delete[] trimmed_data;
		
		// clear dimension estimates
		delete[] x_dims;
		delete[] y_dims;
		delete[] j_dims;
		delete[] z_dims;
	}
	
	// aggregate probabilities
	double* final_probs = new double[5]{0.0, 0.0, 0.0, 0.0, 0.0};
	
	for (int i=0; i<5; i++) {
		for (int j=0; j<len_range; j++) {
			final_probs[i] += range_probabilities[j][i];
		}
		final_probs[i] /= len_range;
	}
	
	// free memory
	for (int i=0; i<len_range; i++) delete[] range_probabilities[i];
	delete[] range_probabilities;
	
	for (int i=0; i<n; i++) {
		delete[] X_nn_distances[i];
		delete[] Y_nn_distances[i];
		delete[] J_nn_distances[i];
		delete[] Z_nn_distances[i];
	}
	
	delete[] X_nn_distances;
	delete[] Y_nn_distances;
	delete[] J_nn_distances;
	delete[] Z_nn_distances;
	
	delete[] X;
	delete[] Y;
	delete[] J;
	delete[] Z;
	delete[] manifolds;
	
	return final_probs;
}
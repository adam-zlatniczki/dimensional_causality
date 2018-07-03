#include <parallel/algorithm>
#include <iostream>
using namespace std;


#pragma omp declare simd
bool* single_trim_mask(double* local_dims, const int n, const double eps=0.05){
	double* sorted_local_dims = new double[n];
	
	#pragma omp parallel for simd
	for (int i=0; i<n; i++){
		sorted_local_dims[i] = local_dims[i];
	}

	__gnu_parallel::sort(sorted_local_dims, sorted_local_dims+n);
	
	int num_remove = eps * n;
	
	double lb = sorted_local_dims[num_remove];
	double ub = sorted_local_dims[n-1-num_remove];
	
	bool* mask = new bool[n];
	
	#pragma omp parallel for simd
	for (int i=0; i<n; i++){
		mask[i] = (lb < local_dims[i]) && (local_dims[i] < ub);
	}
	
	delete[] sorted_local_dims;
	
	return mask;
}

#pragma omp declare simd
bool* merge_masks(bool* x_mask, bool* y_mask, bool* j_mask, bool* z_mask, const int n){
	bool* mask = new bool[n];
	
	#pragma omp parallel for simd
	for (int i=0; i<n; i++){
		mask[i] = x_mask[i] && y_mask[i] && j_mask[i] && z_mask[i];
	}
	
	return mask;
}

bool* joint_trim_mask(double* x_dims, double* y_dims, double* j_dims, double* z_dims, const int n, const double eps=0.05){
	bool* x_mask;
	bool* y_mask;
	bool* j_mask;
	bool* z_mask;
	
	#pragma omp parallel sections
	{
		#pragma omp section
		x_mask = single_trim_mask(x_dims, n, eps);

		#pragma omp section
		y_mask = single_trim_mask(y_dims, n, eps);

		#pragma omp section
		j_mask = single_trim_mask(j_dims, n, eps);

		#pragma omp section
		z_mask = single_trim_mask(z_dims, n, eps);
	}
	
	bool* mask = merge_masks(x_mask, y_mask, j_mask, z_mask, n);
	
	delete[] x_mask;
	delete[] y_mask;
	delete[] j_mask;
	delete[] z_mask;
	
	return mask;
}

double** trim_data(double* x_dims, double* y_dims, double* j_dims, double* z_dims, const int n, unsigned int& trimmed_size, const double eps=0.05){
	bool* mask = joint_trim_mask(x_dims, y_dims, j_dims, z_dims, n, eps);
	
	// find trimmed size
	trimmed_size = 0;
	
	#pragma omp parallel for reduction(+:trimmed_size)
	for (int i=0; i<n; i++)
		if (mask[i] == true)
			trimmed_size += 1;
	
	//
	double** trimmed_data = new double*[4];
	for (int i=0; i<4; i++) trimmed_data[i] = new double[trimmed_size];
	
	// store non-trimmed elements
	unsigned int cntr = 0;
	
	for (int i=0; i<n; i++) {
		if (mask[i] == true) {
			trimmed_data[0][cntr] = x_dims[i];
			trimmed_data[1][cntr] = y_dims[i];
			trimmed_data[2][cntr] = j_dims[i];
			trimmed_data[3][cntr] = z_dims[i];
			cntr++;
		}
	}
	
	return trimmed_data;
}
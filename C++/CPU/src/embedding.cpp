#include <cmath>
#include <algorithm>
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


#pragma omp declare simd
double* diophantine_sum(double* x, double* y,  const unsigned int n,  const unsigned int emb_dim, const unsigned int tau) {
	double* j = new double[n];
	const double a = sqrt(2.0);

	#pragma omp parallel for simd
	for (int i = 0; i < n; i++) {
		j[i] = a*x[i] - y[i];
	}

	return embed(j, n, emb_dim, tau);
}

#pragma omp declare simd
double* shuffled_diophantine_sum(double* x, double* y, const unsigned int n, const unsigned int emb_dim, const unsigned int tau) {
	double* x_s = new double[n];
	double* y_s = new double[n];
	
	#pragma omp parallel for simd
	for (int i=0; i < n; i++) {
		x_s[i] = x[i];
		y_s[i] = y[i];
	}
	
	random_shuffle(x_s, x_s+n);
	random_shuffle(y_s, y_s+n);
	
	double* z = new double[n];
	const double a = sqrt(2.0);

	#pragma omp parallel for simd
	for (int i = 0; i < n; i++) {
		z[i] = a*x_s[i] - y_s[i];
	}
	
	delete[] x_s;
	delete[] y_s;
	
	return embed(z, n, emb_dim, tau);
}
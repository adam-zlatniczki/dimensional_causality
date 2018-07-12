#include <iostream>

using namespace std;


double exp_val(double* x_dims, const unsigned int n){
	double exp_val = 0.0;
	
	#pragma omp parallel for simd reduction(+: exp_val)
	for (int i=0; i<n; i++) {
		exp_val += x_dims[i];
	}
	
	return exp_val / n;
}

void cov(double** matr, unsigned int n, double* cov_m, unsigned int i, unsigned int j){
	double sum_1 = 0.0;
	double sum_2 = 0.0;
	double sqr_sum = 0.0;

	#pragma parallel for simd reduction(+:sum_1, sum_2, sqr_sum)
	for (int k=0; k<n; k++) {
		sum_1 += matr[i][k];
		sum_2 += matr[j][k];
		sqr_sum += matr[i][k] * matr[j][k];
	}
	
	double cov = (sqr_sum / n) - (sum_1 * sum_2 / (n * n));

	cov_m[i*4+j] = cov;
	cov_m[j*4+i] = cov;
}

double* cov_matr(double** dims, const unsigned int n){
	double* cov_m = new double[16];
	
	#pragma omp parallel sections
	{
		#pragma omp section
		cov(dims, n, cov_m, 0, 0);
		
		#pragma omp section
		cov(dims, n, cov_m, 0, 1);
		
		#pragma omp section
		cov(dims, n, cov_m, 0, 2);
		
		#pragma omp section
		cov(dims, n, cov_m, 0, 3);
		
		#pragma omp section
		cov(dims, n, cov_m, 1, 1);
		
		#pragma omp section
		cov(dims, n, cov_m, 1, 2);
		
		#pragma omp section
		cov(dims, n, cov_m, 1, 3);
		
		#pragma omp section
		cov(dims, n, cov_m, 2, 2);
		
		#pragma omp section
		cov(dims, n, cov_m, 2, 3);
		
		#pragma omp section
		cov(dims, n, cov_m, 3, 3);
	}
	
	return cov_m;
}

double* inv_cov_m_4x4(double* m, double& detOut){
    double inv[16];
    int i;

    inv[0] = m[5]  * m[10] * m[15] - 
             m[5]  * m[11] * m[14] - 
             m[9]  * m[6]  * m[15] + 
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] - 
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] + 
              m[4]  * m[11] * m[14] + 
              m[8]  * m[6]  * m[15] - 
              m[8]  * m[7]  * m[14] - 
              m[12] * m[6]  * m[11] + 
              m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] - 
             m[4]  * m[11] * m[13] - 
             m[8]  * m[5] * m[15] + 
             m[8]  * m[7] * m[13] + 
             m[12] * m[5] * m[11] - 
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] + 
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] - 
               m[8]  * m[6] * m[13] - 
               m[12] * m[5] * m[10] + 
               m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] + 
              m[1]  * m[11] * m[14] + 
              m[9]  * m[2] * m[15] - 
              m[9]  * m[3] * m[14] - 
              m[13] * m[2] * m[11] + 
              m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] - 
             m[0]  * m[11] * m[14] - 
             m[8]  * m[2] * m[15] + 
             m[8]  * m[3] * m[14] + 
             m[12] * m[2] * m[11] - 
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] + 
              m[0]  * m[11] * m[13] + 
              m[8]  * m[1] * m[15] - 
              m[8]  * m[3] * m[13] - 
              m[12] * m[1] * m[11] + 
              m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] - 
              m[0]  * m[10] * m[13] - 
              m[8]  * m[1] * m[14] + 
              m[8]  * m[2] * m[13] + 
              m[12] * m[1] * m[10] - 
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] - 
             m[1]  * m[7] * m[14] - 
             m[5]  * m[2] * m[15] + 
             m[5]  * m[3] * m[14] + 
             m[13] * m[2] * m[7] - 
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] + 
              m[0]  * m[7] * m[14] + 
              m[4]  * m[2] * m[15] - 
              m[4]  * m[3] * m[14] - 
              m[12] * m[2] * m[7] + 
              m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] - 
              m[0]  * m[7] * m[13] - 
              m[4]  * m[1] * m[15] + 
              m[4]  * m[3] * m[13] + 
              m[12] * m[1] * m[7] - 
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] + 
               m[0]  * m[6] * m[13] + 
               m[4]  * m[1] * m[14] - 
               m[4]  * m[2] * m[13] - 
               m[12] * m[1] * m[6] + 
               m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + 
              m[1] * m[7] * m[10] + 
              m[5] * m[2] * m[11] - 
              m[5] * m[3] * m[10] - 
              m[9] * m[2] * m[7] + 
              m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - 
             m[0] * m[7] * m[10] - 
             m[4] * m[2] * m[11] + 
             m[4] * m[3] * m[10] + 
             m[8] * m[2] * m[7] - 
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + 
               m[0] * m[7] * m[9] + 
               m[4] * m[1] * m[11] - 
               m[4] * m[3] * m[9] - 
               m[8] * m[1] * m[7] + 
               m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - 
              m[0] * m[6] * m[9] - 
              m[4] * m[1] * m[10] + 
              m[4] * m[2] * m[9] + 
              m[8] * m[1] * m[6] - 
              m[8] * m[2] * m[5];

    detOut = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (detOut == 0.0)
        return NULL;

    detOut = 1.0 / detOut;

	double* invOut = new double[16];
	
    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * detOut;

    return invOut;
}

double* inv_cov_m_3x3(double* m, double& detOut){
	double* inv = new double[9];
	double a = m[4]*m[8] - m[5]*m[7];
	double b = m[5]*m[6] - m[3]*m[8];
	double c = m[3]*m[7] - m[4]*m[6];
	
	detOut = m[0]*a + m[1]*b + m[2]*c;
	
	if (detOut == 0.0) {
		return NULL;
	}
	
	detOut = 1.0 / detOut;
	
	inv[0] = a * detOut;
	inv[1] = (m[2]*m[7] - m[1]*m[8]) * detOut;
	inv[2] = (m[1]*m[5] - m[2]*m[4]) * detOut;
	
	inv[3] = b * detOut;
	inv[4] = (m[0]*m[8] - m[2]*m[6]) * detOut;
	inv[5] = (m[2]*m[3] - m[0]*m[5]) * detOut;
	
	inv[6] = c * detOut;
	inv[7] = (m[1]*m[6] - m[0]*m[7]) * detOut;
	inv[8] = (m[0]*m[4] - m[1]*m[3]) * detOut;
	
	return inv;
}

double* inv_cov_m_2x2(double* m, double& detOut){
	double* inv = new double[4];
	
	detOut = m[0]*m[3] - m[1]*m[2];
	
	if (detOut == 0.0) {
		return NULL;
	}
	
	detOut = 1.0 / detOut;
	
	inv[0] = m[3] * detOut;
	inv[1] = -1.0 * m[1] * detOut;
	inv[2] = -1.0 * m[2] * detOut;
	inv[3] = m[0] * detOut;
	
	return inv;
}
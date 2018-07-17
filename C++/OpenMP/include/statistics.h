#ifndef STATISTICS_H
#define STATISTICS_H

double exp_val(double* x, const unsigned int n);
void cov(double** matr, double* expvs, unsigned int n, double** cov_m, unsigned int i, unsigned int j);
double* cov_matr(double** dims, double* expvs, const unsigned int n);
double* inv_cov_m_4x4(double* m, double& detOut);
double* inv_cov_m_3x3(double* m, double& detOut);
double* inv_cov_m_2x2(double* m, double& detOut);

#endif
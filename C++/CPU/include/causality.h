#ifndef CAUSALITY_H
#define CAUSALITY_H

extern "C" {
	double* __declspec(dllexport) infer_causality(double* x, double* y, unsigned int n, unsigned int emb_dim, unsigned int tau, unsigned int* k_range, unsigned int len_range, double eps=0.05, double c=3.0, double bins=20.0);
}

#endif
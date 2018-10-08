#ifndef CAUSALITY_H
#define CAUSALITY_H

extern "C" {
	double* infer_causality_from_manifolds(double* X, double* Y, double* J, double* Z, unsigned int n, unsigned int emb_dim, unsigned int* k_range, unsigned int len_range, double eps=0.05, double c=3.0, double bins=20.0, double** export_dims_p=NULL, double** export_stdevs_p=NULL);
	double* infer_causality(double* x, double* y, unsigned int n, unsigned int emb_dim, unsigned int tau, unsigned int* k_range, unsigned int len_range, double eps=0.05, double c=3.0, double bins=20.0, unsigned int downsample_rate=1, double** export_dims_p=NULL, double** export_stdevs_p=NULL);
	void infer_causality_R(double* probs_return, double* x, double* y, unsigned int* n, unsigned int* emb_dim, unsigned int* tau, unsigned int* k_range, unsigned int* len_range, double* eps, double* c, double* bins, unsigned int* downsample_rate, double* export_dims=NULL, double* export_stdevs=NULL);
}

#endif
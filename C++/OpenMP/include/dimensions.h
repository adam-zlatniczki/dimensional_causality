#ifndef DIMENSIONS_H
#define DIMENSIONS_H


double** knn_distances(const double* X, unsigned int n, unsigned int d, unsigned int k);

#pragma omp declare simd
double* local_dims(double** dist, const unsigned int n, unsigned int k);


#endif
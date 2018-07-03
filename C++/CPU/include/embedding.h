#ifndef EMBEDDING_H
#define EMBEDDING_H

// Takens time-delay embedding
double* embed(double* x, const unsigned int n, const unsigned int emb_dim, const unsigned int tau);

#pragma omp declare simd
double* diophantine_sum(double* x, double* y, const unsigned int n, const unsigned int emb_dim, const unsigned int tau);

#pragma omp declare simd
double* shuffled_diophantine_sum(double* x, double* y, const unsigned int n, const unsigned int emb_dim, const unsigned int tau);

#endif
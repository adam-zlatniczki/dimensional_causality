#ifndef EMBEDDING_H
#define EMBEDDING_H

// Takens time-delay embedding
double* embed(double* x, const unsigned int n, const unsigned int emb_dim, const unsigned int tau);

double** get_manifolds(double* x, double* y, const unsigned int n, const unsigned int emb_dim, const unsigned int tau, unsigned int downsample_rate=1);

#endif
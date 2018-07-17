#ifndef TRIMMING_H
#define TRIMMING_H

#pragma omp declare simd
bool* single_trim_mask(double* local_dims, const int n, const double eps=0.05);

bool* merge_masks(bool* x_mask, bool* y_mask, bool* j_mask, bool* z_mask, const int n);

bool* joint_trim_mask(double* x_dims, double* y_dims, double* j_dims, double* z_dims, const int n, const double eps=0.05);

double** trim_data(double* x_dims, double* y_dims, double* j_dims, double* z_dims, const int n, unsigned int& trimmed_size, const double eps=0.05);

#endif
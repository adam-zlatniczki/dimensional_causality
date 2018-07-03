#ifndef PROBABILITIES_H
#define PROBABILITIES_H

class Gauss {
	public:
		void set(double* expv, double* cov_m, unsigned int dim);
		double pdf(double* point);
		unsigned int get_dim(){ return dim; };
		double* get_expv(){ return expv; };
		double* get_inv_cov_m(){ return inv_cov_m; };
		double get_det(){ return det; };
		double get_norm_term(){ return norm_term; };
	private:
		unsigned int dim;
		double* expv;
		double* inv_cov_m;
		double det;
		double norm_term;
};

double prob_A11(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);
double prob_A12(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);
double prob_A13(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);
double prob_A21(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);
double prob_A22(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);
double prob_A23(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);
double prob_A31(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);
double prob_A32(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);
double prob_A33(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);

double* get_probabilities(double** dims, unsigned int n, double eff_sample_size);

#endif
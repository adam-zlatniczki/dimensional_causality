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

void check_feasibility(double min_x, double min_y, double min_j, double max_j, double max_z);

// P(A11)
double P_Y_eq_J(double* expv, double* cov_m, double max_y, double min_y, double dy);
double P_X_less_Y_eq_J(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_y, double min_y, double dy);
double P_Y_eq_J_less_Z(double* expv, double* cov_m, double max_y, double min_y, double dy, double max_z, double min_z, double dz);
double P_X_less_Y_eq_J_less_Z(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_y, double min_y, double dy, double max_z, double min_z, double dz);

// P(A12)
double P_X_less_Y(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_y, double min_y, double dy);
double P_Y_less_J(double* expv, double* cov_m, double max_y, double min_y, double dy, double max_j, double min_j, double dj);
double P_X_less_Y_less_J(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_y, double min_y, double dy, double max_j, double min_j, double dj);
double P_J_less_Z(double* expv, double* cov_m, double max_j, double min_j, double dj, double max_z, double min_z, double dz);
double P_X_less_Y_J_less_Z(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_y, double min_y, double dy, double max_j, double min_j, double dj, double max_z, double min_z, double dz);
double P_Y_less_J_less_Z(double* expv, double* cov_m, double max_y, double min_y, double dy, double max_j, double min_j, double dj, double max_z, double min_z, double dz);
double P_X_less_Y_less_J_less_Z(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_y, double min_y, double dy, double max_j, double min_j, double dj, double max_z, double min_z, double dz);

// P(A13)
double P_J_eq_Z(double* expv, double* cov_m, double max_j, double min_j, double dj);
double P_X_less_Y_J_eq_Z(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_y, double min_y, double dy, double max_j, double min_j, double dj);
double P_Y_less_J_eq_Z(double* expv, double* cov_m, double max_y, double min_y, double dy, double max_j, double min_j, double dj);
double P_X_less_Y_less_J_eq_Z(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_y, double min_y, double dy, double max_j, double min_j, double dj);

// P(A21)
double P_X_eq_J(double* expv, double* cov_m, double max_x, double min_x, double dx);
double P_Y_less_X_eq_J(double* expv, double* cov_m, double max_y, double min_y, double dy, double max_x, double min_x, double dx);
double P_X_eq_J_less_Z(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_z, double min_z, double dz);
double P_Y_less_X_eq_J_less_Z(double* expv, double* cov_m, double max_y, double min_y, double dy, double max_x, double min_x, double dx, double max_z, double min_z, double dz);

// P(A22)
double P_Y_less_X(double* expv, double* cov_m, double max_y, double min_y, double dy, double max_x, double min_x, double dx);
double P_X_less_J(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_j, double min_j, double dj);
double P_Y_less_X_less_J(double* expv, double* cov_m, double max_y, double min_y, double dy, double max_x, double min_x, double dx, double max_j, double min_j, double dj);
double P_Y_less_X_J_less_Z(double* expv, double* cov_m, double max_y, double min_y, double dy, double max_x, double min_x, double dx, double max_j, double min_j, double dj, double max_z, double min_z, double dz);
double P_X_less_J_less_Z(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_j, double min_j, double dj, double max_z, double min_z, double dz);
double P_Y_less_X_less_J_less_Z(double* expv, double* cov_m, double max_y, double min_y, double dy, double max_x, double min_x, double dx, double max_j, double min_j, double dj, double max_z, double min_z, double dz);

// P(A23)
double P_Y_less_X_J_eq_Z(double* expv, double* cov_m, double max_y, double min_y, double dy, double max_x, double min_x, double dx, double max_j, double min_j, double dj);
double P_X_less_J_eq_Z(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_j, double min_j, double dj);
double P_Y_less_X_less_J_eq_Z(double* expv, double* cov_m, double max_y, double min_y, double dy, double max_x, double min_x, double dx, double max_j, double min_j, double dj);

// P(A31)
double P_X_eq_Y_eq_J(double* expv, double* cov_m, double max_x, double min_x, double dx);
double P_X_eq_Y_eq_J_less_Z(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_z, double min_z, double dz);

// P(A32)
double P_X_eq_Y(double* expv, double* cov_m, double max_x, double min_x, double dx);
double P_X_eq_Y_less_J(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_j, double min_j, double dj);
double P_X_eq_Y_J_less_Z(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_j, double min_j, double dj, double max_z, double min_z, double dz);
double P_X_eq_Y_less_J_less_Z(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_j, double min_j, double dj, double max_z, double min_z, double dz);

// P(A33)
double P_X_eq_Y_J_eq_Z(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_j, double min_j, double dj);
double P_X_eq_Y_less_J_eq_Z(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_j, double min_j, double dj);

// calculate case probabilities
double prob_A11(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);
double prob_A12(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);
double prob_A13(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);
double prob_A21(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);
double prob_A22(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);
double prob_A23(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);
double prob_A31(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);
double prob_A32(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);
double prob_A33(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz);

void fit_gauss(double** dims, unsigned int n, double eff_sample_size, double** expv_p, double** cov_m_P);
double* get_probabilities(double* expv, double* cov_m, double c=3.0, double bins=20.0);

#endif
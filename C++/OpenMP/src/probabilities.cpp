#include <iostream>
#include <cmath>
#include "statistics.h"

#include <stdexcept>

using namespace std;

# define PI 3.14159265358979323846

class Gauss {
	public:
		void set(double* expv, double* cov_m, unsigned int dim);
		double pdf(double* point);
		unsigned int get_dim(){ return dim; };
		double* get_expv(){ return expv; };
		double* get_inv_cov_m(){ return inv_cov_m; };
		double get_det(){ return det; };
		double get_norm_term(){ return norm_term; };
		~Gauss();
	private:
		unsigned int dim = 0;
		double* expv = NULL;
		double* inv_cov_m = NULL;
		double det = 0.0;
		double norm_term = 0.0;
};

Gauss::~Gauss(){
	if (expv != NULL) delete[] expv;
	if (inv_cov_m != NULL) delete[] inv_cov_m;
}

void Gauss::set(double* expv, double* cov_m, unsigned int dim){
	this->dim = dim;
	
	this->expv = new double[dim];
	for (int i=0; i<dim; i++) this->expv[i] = expv[i];
	
	switch (dim) {
		case 4:
			inv_cov_m = inv_cov_m_4x4(cov_m, det);
			break;
		case 3:
			inv_cov_m = inv_cov_m_3x3(cov_m, det);
			break;
		case 2:
			inv_cov_m = inv_cov_m_2x2(cov_m, det);
			break;
	}
	
	det = 1.0 / det;
	
	double a = 2.0 * PI;
	norm_term = a;
	
	for (int i=0; i<dim-1; i++) {
		norm_term *= a;
	}
	
	norm_term = 1.0 / sqrt(norm_term * det);
}

double Gauss::pdf(double* point){
	double likelihood = 0.0;
	double* diff = new double[4];
	
	for (int i=0; i<dim; i++) {
		diff[i] = point[i] - expv[i];
	}
	
	for (int i=0; i<dim; i++) {
		for (int j=0; j<dim; j++) {
			likelihood += diff[i] * diff[j] * inv_cov_m[i*dim + j];
		}
	}
	
	likelihood = exp(-0.5 * likelihood);
	
	delete[] diff;
	
	return norm_term * likelihood;
}


bool check_feasibility(double min_x, double min_y, double min_j, double max_j, double max_z){
	bool feasible = true;
	
	if (min_x > max_j) feasible = false;
	if (min_y > max_j) feasible = false; 
	if (min_x > max_z) feasible = false;
	if (min_y > max_z) feasible = false;
	if (min_j > max_z) feasible = false;
	
	return feasible;
}

double P_Y_eq_J(double* expv, double* cov_m, double max_y, double min_y, double dy){
	// calculate P(Y = J)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[2] = {expv[1], expv[2]};
	double projected_cov_m[4] = {cov_m[5], cov_m[6], cov_m[9], cov_m[10]};
	gauss.set(projected_expv, projected_cov_m, 2);
	
	unsigned int y_steps = (max_y - min_y) / dy;
	double y;
		
	for (int i=0; i<=y_steps; i++) {
		y = min_y + i*dy;
		double point[2] = {y, y};
		likelihood += gauss.pdf(point);
	}
	likelihood = likelihood * dy;
	return likelihood;
}

double P_X_less_Y_eq_J(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_y, double min_y, double dy){
	// calculate P(X < Y = J)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[3] = {expv[0], expv[1], expv[2]};
	double projected_cov_m[9] = {cov_m[0], cov_m[1], cov_m[2],
								 cov_m[4], cov_m[5], cov_m[6],
								 cov_m[8], cov_m[9], cov_m[10]};
	gauss.set(projected_expv, projected_cov_m, 3);
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int y_steps;
	double x, y, y_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		y_lb = min( max(x, min_y), max_y );
		y_steps = (max_y - y_lb) / dy;
		
		int k = 0;
		if (y_lb == x) k = 1;
		
		for (k; k<=y_steps; k++) {
			y = y_lb + k*dy;
			double point[3] = {x, y, y};
			likelihood += gauss.pdf(point);
		}
	}
	likelihood = likelihood * dy * dx;
	return likelihood;
}

double P_Y_eq_J_less_Z(double* expv, double* cov_m, double max_y, double min_y, double dy, double max_z, double min_z, double dz){
	// calculate P(Y = J < Z)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[3] = {expv[1], expv[2], expv[3]};
	double projected_cov_m[9] = {cov_m[5], cov_m[6], cov_m[7],
								 cov_m[9], cov_m[10], cov_m[11],
								 cov_m[13], cov_m[14], cov_m[15]};
	gauss.set(projected_expv, projected_cov_m, 3);
	
	unsigned int y_steps = (max_y - min_y) / dy;
	unsigned int z_steps;
	double y, z, z_lb;
	
	for (int i=0; i<=y_steps; i++) {
		y = min_y + i*dy;
		z_lb = min( max(y, min_z), max_z );
		z_steps = (max_z - z_lb) / dz;
		
		int k = 0;
		if (z_lb == y) k = 1;
		
		for (k; k<=z_steps; k++){
			z = z_lb + k*dz;
			double point[3] = {y, y, z};
			likelihood += gauss.pdf(point);
		}
	}
	likelihood = likelihood * dz * dy;
	return likelihood;
}

double P_X_less_Y_eq_J_less_Z(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_y, double min_y, double dy, double max_z, double min_z, double dz){
	// calculate P(X < Y = J < Z)
	Gauss gauss;
	double likelihood = 0.0;
	
	gauss.set(expv, cov_m, 4);
			
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int y_steps, z_steps;
	double x, y, z, y_lb, z_lb;
	int k, l;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		y_lb = min( max(x, min_y), max_y );
		y_steps = (max_y - y_lb) / dy;
		
		k = 0;
		if (y_lb == x) k = 1;
		
		for (k; k<=y_steps; k++) {
			y = y_lb + k*dy;
			z_lb = min( max(y, min_z), max_z );
			z_steps = (max_z - z_lb) / dz;
			
			l = 0;
			if (z_lb == y) l = 1;
			
			for (l; l<=z_steps; l++) {
				z = z_lb + l*dz;
				double point[4] = {x, y, y, z};
				likelihood += gauss.pdf(point);
			}
		}
	}
	likelihood = likelihood * dz * dy * dx;
	return likelihood;
}

double P_X_less_Y(double* expv, double* cov_m, double max_x, double min_x, double dx, double max_y, double min_y, double dy){
	// calculate P(X<Y)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[2] = {expv[0], expv[1]};
	double projected_cov_m[4] = {cov_m[0], cov_m[1], cov_m[4], cov_m[5]};
	gauss.set(projected_expv, projected_cov_m, 2);
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int y_steps;
	double x, y, y_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		y_lb = min( max(x, min_y), max_y );
		y_steps = (max_y - y_lb) / dy;
		
		int k = 0;
		if (y_lb == x) k = 1;
		
		for (k; k<=y_steps; k++) {
			y = y_lb + k*dy;
			double point[2] = {x, y};
			likelihood += gauss.pdf(point);
		}
	}
	likelihood = likelihood * dy * dx;
	return likelihood;
}


double P_Y_less_J(double* expv, double* cov_m, double max_y, double min_y, double dy, double max_j, double min_j, double dj){
	// calculate P(Y < J)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[2] = {expv[1], expv[2]};
	double projected_cov_m[4] = {cov_m[5], cov_m[6], cov_m[9], cov_m[10]};
	gauss.set(projected_expv, projected_cov_m, 2);
	
	unsigned int y_steps = (max_y - min_y) / dy;
	unsigned int j_steps;
	double y, j, j_lb;
	
	for (int i=0; i<=y_steps; i++) {
		y = min_y + i*dy;
		j_lb = min( max(y, min_j), max_j);
		j_steps = (max_j - j_lb) / dj;
		
		int k = 0;
		if (j_lb == y) k = 1;
		
		for (k; k<=j_steps; k++) {
			j = j_lb + k*dj;
			double point[2] = {y, j};
			likelihood += gauss.pdf(point);
		}
	}
	likelihood = likelihood * dj * dy;
	return likelihood;
}

double P_X_less_Y_less_J(double* expv, double* cov_m,
                         double max_x, double min_x, double dx,
						 double max_y, double min_y, double dy,
						 double max_j, double min_j, double dj){
	// calculate P(X < Y < J)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[3] = {expv[0], expv[1], expv[2]};
	double projected_cov_m[9] = {cov_m[0], cov_m[1], cov_m[2],
	                             cov_m[4], cov_m[5], cov_m[6],
								 cov_m[8], cov_m[9], cov_m[10]};
	gauss.set(projected_expv, projected_cov_m, 3);
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int y_steps, j_steps;
	double x, y, j, y_lb, j_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		y_lb = min( max(x, min_y), max_y );
		y_steps = (max_y - y_lb) / dy;
		
		int k = 0;
		if (y_lb == x) k = 1;
		
		for (k; k<=y_steps; k++) {
			y = y_lb + k*dy;
			j_lb = min( max(y, min_j), max_j);
			j_steps = (max_j - j_lb) / dj;
			
			int l = 0;
			if (j_lb == y) l = 1;
			
			for (l; l<=j_steps; l++) {
				j = j_lb + l*dj;
				double point[3] = {x, y, j};
				likelihood += gauss.pdf(point);
			}
		}
	}
	likelihood = likelihood * dj * dy * dx;
	return likelihood;
}

double P_J_less_Z(double* expv, double* cov_m, double max_j, double min_j, double dj, double max_z, double min_z, double dz){
	// calculate P(J < Z)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[2] = {expv[2], expv[3]};
	double projected_cov_m[4] = {cov_m[10], cov_m[11], cov_m[14], cov_m[15]};
	gauss.set(projected_expv, projected_cov_m, 2);
	
	unsigned int j_steps = (max_j - min_j) / dj;
	unsigned int z_steps;
	double j, z, z_lb;
	
	for (int i=0; i<=j_steps; i++) {
		j = min_j + i*dj;
		z_lb = min( max(j, min_z), max_z);
		z_steps = (max_z - z_lb) / dz;
		
		int k = 0;
		if (z_lb == j) k = 1;
		
		for (k; k<=z_steps; k++) {
			z = z_lb + k*dz;
			double point[2] = {j, z};
			likelihood += gauss.pdf(point);
		}
	}
	likelihood = likelihood * dz * dj;
	return likelihood;
}

double P_X_less_Y_J_less_Z(double* expv, double* cov_m,
                           double max_x, double min_x, double dx,
						   double max_y, double min_y, double dy,
						   double max_j, double min_j, double dj,
						   double max_z, double min_z, double dz){
	// calculate P(X<Y, J<Z)
	// j must go from at least y
	Gauss gauss;
	gauss.set(expv, cov_m, 4);
	
	double likelihood = 0.0;
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int y_steps, j_steps, z_steps;
	double x, y, y_lb, j, j_lb, z, z_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		y_lb = min( max(x, min_y), max_y);
		y_steps = (max_y - y_lb) / dy;
		
		int k = 0;
		if (y_lb == x) k = 1;
		
		for (k; k<=y_steps; k++) {
			y = y_lb + k*dy;
			j_lb = min( max(y, min_j), max_j);
			j_steps = (max_j - j_lb) / dj;
			
			int l = 0;
			if (j_lb == y) l = 1;
			
			for (l; l<=j_steps; l++) {
				j = j_lb + l*dj;
				z_lb = min( max(j, min_z), max_z);
				z_steps = (max_z - z_lb) / dz;
				
				int m = 0;
				if (z_lb == j) m = 1;
				
				for (m; m<=z_steps; m++) {
					z = z_lb + m*dz;
					double point[4] = {x, y, j, z};
					likelihood += gauss.pdf(point);
				}
			}
			
		}
	}
	likelihood = likelihood * dz * dj * dy * dx;
	return likelihood;
}

double P_Y_less_J_less_Z(double* expv, double* cov_m,
                         double max_y, double min_y, double dy,
						 double max_j, double min_j, double dj,
						 double max_z, double min_z, double dz){
	// calculate P(Y < J < Z)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[3] = {expv[1], expv[2], expv[3]};
	double projected_cov_m[9] = {cov_m[5], cov_m[6], cov_m[7],
	                             cov_m[9], cov_m[10], cov_m[11],
								 cov_m[13], cov_m[14], cov_m[15]};
	gauss.set(projected_expv, projected_cov_m, 3);
	
	unsigned int y_steps = (max_y - min_y) / dy;
	unsigned int j_steps, z_steps;
	double y, j, j_lb, z, z_lb;
	
	for (int i=0; i<=y_steps; i++) {
		y = min_y + i*dy;
		j_lb = min( max(y, min_j), max_j);
		j_steps = (max_j - j_lb) / dj;
		
		int k = 0;
		if (j_lb == y) k = 1;
		
		for (k; k<=j_steps; k++) {
			j = j_lb + k*dj;
			z_lb = min( max(j, min_z), max_z);
			z_steps = (max_z - z_lb) / dz;
			
			int l = 0;
			if (z_lb == j) l = 1;
			
			for (l; l<=z_steps; l++) {
				z = z_lb + l*dz;
				double point[3] = {y, j, z};
				likelihood += gauss.pdf(point);
			}
		}
	}
	likelihood = likelihood * dz * dj * dy;
	return likelihood;
}

double P_X_less_Y_less_J_less_Z(double* expv, double* cov_m,
                                double max_x, double min_x, double dx,
								double max_y, double min_y, double dy,
								double max_j, double min_j, double dj,
								double max_z, double min_z, double dz){
	// calculate P(X < Y < J < Z)
	Gauss gauss;
	gauss.set(expv, cov_m, 4);
	
	double likelihood = 0.0;
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int y_steps, j_steps, z_steps;
	double x, y, y_lb, j, j_lb, z, z_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		y_lb = min( max(x, min_y), max_y);
		y_steps = (max_y - y_lb) / dy;
		
		int k = 0;
		if (y_lb == x) k = 1;
		
		for (k; k<=y_steps; k++) {
			y = y_lb + k*dy;
			j_lb = min( max(y, min_j), max_j);
			j_steps = (max_j - j_lb) / dj;
			
			int l = 0;
			if (j_lb == y) l = 1;
			
			for (l; l<=j_steps; l++) {
				j = j_lb + l*dj;
				z_lb = min( max(j, min_z), max_z);
				z_steps = (max_z - z_lb) / dz;
				
				int m = 0;
				if (z_lb == j) m = 1;
				
				for (m; m<=z_steps; m++) {
					z = z_lb + m*dz;
					double point[4] = {x, y, j, z};
					likelihood += gauss.pdf(point);
				}
			}
		}
	}
	likelihood = likelihood * dz * dj * dy * dx;
	return likelihood;
}

double P_J_eq_Z(double* expv, double* cov_m, double max_j, double min_j, double dj){
	// calculate P(J = Z)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[2] = {expv[2], expv[3]};
	double projected_cov_m[4] = {cov_m[10], cov_m[11], cov_m[14], cov_m[15]};
	gauss.set(projected_expv, projected_cov_m, 2);
	
	unsigned int j_steps = (max_j - min_j) / dj;
	double j;
	
	for (int i=0; i<=j_steps; i++) {
		j = min_j + i*dj;
		double point[2] = {j, j};
		likelihood += gauss.pdf(point);
	}
	
	likelihood = likelihood * dj;
	return likelihood;
}

double P_X_less_Y_J_eq_Z(double* expv, double* cov_m,
                         double max_x, double min_x, double dx,
						 double max_y, double min_y, double dy,
						 double max_j, double min_j, double dj){
	// P(X < Y, J = Z)
	// j starts from y
	Gauss gauss;
	gauss.set(expv, cov_m, 4);
	
	double likelihood = 0.0;
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int y_steps, j_steps;
	double x, y, y_lb, j, j_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		y_lb = min( max(x, min_y), max_y);
		y_steps = (max_y - y_lb) / dy;
		
		int k = 0;
		if (y_lb == x) k = 1;
		
		for (k; k<=y_steps; k++) {
			y = y_lb + k*dy;
			j_lb = min( max(y, min_j), max_j);
			j_steps = (max_j - j_lb) / dj;
			
			int l = 0;
			if (j_lb == y) l = 1;
			
			for (l; l<=j_steps; l++) {
				j = j_lb + l*dj;
				double point[4] = {x, y, j, j};
				likelihood += gauss.pdf(point);
			}
		}
	}
	likelihood *= dj * dy * dx;
	return likelihood;
}

double P_Y_less_J_eq_Z(double* expv, double* cov_m, double max_y, double min_y, double dy, double max_j, double min_j, double dj){
	// P(Y < J = Z)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[3] = {expv[1], expv[2], expv[3]};
	double projected_cov_m[9] = {cov_m[5], cov_m[6], cov_m[7],
	                             cov_m[9], cov_m[10], cov_m[11],
								 cov_m[13], cov_m[14], cov_m[15]};
	gauss.set(projected_expv, projected_cov_m, 3);
	
	unsigned int y_steps = (max_y - min_y) / dy;
	unsigned int j_steps;
	double y, j, j_lb;
	
	for (int i=0; i<=y_steps; i++) {
		y = min_y + i*dy;
		j_lb = min( max(y, min_j), max_j);
		j_steps = (max_j - j_lb) / dj;
		
		int k = 0;
		if (j_lb == y) k = 1;
		
		for (k; k<=j_steps; k++) {
			j = j_lb + k*dj;
			double point[3] = {y, j, j};
			likelihood += gauss.pdf(point);
		}
	}
	likelihood *= dj * dy;
	return likelihood;
}

double P_X_less_Y_less_J_eq_Z(double* expv, double* cov_m,
                              double max_x, double min_x, double dx,
							  double max_y, double min_y, double dy,
							  double max_j, double min_j, double dj){
	// P(X < Y < J = Z)
	Gauss gauss;
	gauss.set(expv, cov_m, 4);
	
	double likelihood = 0.0;
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int y_steps, j_steps;
	double x, y, y_lb, j, j_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		y_lb = min( max(x, min_y), max_y);
		y_steps = (max_y - y_lb) / dy;
		
		int k = 0;
		if (y_lb == x) k = 1;
		
		for (k; k<=y_steps; k++) {
			y = y_lb + k*dy;
			j_lb = min( max(y, min_j), max_j);
			j_steps = (max_j - j_lb) / dj;
			
			int l = 0;
			if (j_lb == y) l = 1;
			
			for (l; l<=j_steps; l++) {
				j = j_lb + l*dj;
				double point[4] = {x, y, j, j};
				likelihood += gauss.pdf(point);
			}
		}
	}
	likelihood *= dj * dy * dx;
	return likelihood;
}
	
double P_X_eq_J(double* expv, double* cov_m, double max_x, double min_x, double dx){
	// P(X = J)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[2] = {expv[0], expv[2]};
	double projected_cov_m[4] = {cov_m[0], cov_m[2], cov_m[8], cov_m[10]};
	gauss.set(projected_expv, projected_cov_m, 2);
	
	unsigned int x_steps = (max_x - min_x) / dx;
	double x;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		double point[2] = {x, x};
		likelihood += gauss.pdf(point);
	}
	
	likelihood *= dx;
	return likelihood;
}

double P_Y_less_X_eq_J(double* expv, double* cov_m,
                       double max_y, double min_y, double dy,
					   double max_x, double min_x, double dx){
	// P(Y < X = J)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[3] = {expv[0], expv[1], expv[2]};
	double projected_cov_m[9] = {cov_m[0], cov_m[1], cov_m[2],
	                             cov_m[4], cov_m[5], cov_m[6],
								 cov_m[8], cov_m[9], cov_m[10]};
	gauss.set(projected_expv, projected_cov_m, 3);
	
	unsigned int y_steps = (max_y - min_y) / dy;
	unsigned int x_steps;
	double y, x, x_lb;
	
	for (int i=0; i<=y_steps; i++) {
		y = min_y + i*dy;
		x_lb = min( max(y, min_x), max_x);
		x_steps = (max_x - x_lb) / dx;
		
		int k = 0;
		if (x_lb == y) k = 1;
		
		for (k; k<=x_steps; k++) {
			x = x_lb + k*dx;
			double point[3] = {x, y, x};
			likelihood += gauss.pdf(point);
		}
	}
	
	likelihood *= dx * dy;
	return likelihood;
}

double P_X_eq_J_less_Z(double* expv, double* cov_m,
                       double max_x, double min_x, double dx,
					   double max_z, double min_z, double dz){
	// P(X = J < Z)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[3] = {expv[0], expv[2], expv[3]};
	double projected_cov_m[9] = {cov_m[0], cov_m[2], cov_m[3],
	                             cov_m[8], cov_m[10], cov_m[11],
								 cov_m[12], cov_m[14], cov_m[15]};
	gauss.set(projected_expv, projected_cov_m, 3);
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int z_steps;
	double x, z, z_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		z_lb = min( max(x, min_z), max_z);
		z_steps = (max_z - z_lb) / dz;
		
		int k = 0;
		if (z_lb == x) k = 1;
		
		for (k; k<=z_steps; k++) {
			z = z_lb + k*dz;
			double point[3] = {x, x, z};
			likelihood += gauss.pdf(point);
		}
	}
	
	likelihood *= dz * dx;
	return likelihood;
}

double P_Y_less_X_eq_J_less_Z(double* expv, double* cov_m,
                              double max_y, double min_y, double dy,
							  double max_x, double min_x, double dx,
							  double max_z, double min_z, double dz){
	// P(Y < X = J < Z)
	Gauss gauss;
	gauss.set(expv, cov_m, 4);
	
	double likelihood = 0.0;
	
	unsigned int y_steps = (max_y - min_y) / dy;
	unsigned int x_steps, z_steps;
	double y, x, x_lb, z, z_lb;
	
	for (int i=0; i<=y_steps; i++) {
		y = min_y + i*dy;
		x_lb = min( max(y, min_x), max_x);
		x_steps = (max_x - x_lb) / dx;
		
		int k = 0;
		if (x_lb == y) k = 1;
		
		for (k; k<=x_steps; k++) {
			x = x_lb + k*dx;
			z_lb = min( max(x, min_z), max_z);
			z_steps = (max_z - z_lb) / dz;
			
			int l = 0;
			if (z_lb == x) l = 1;
			
			for (l; l<=z_steps; l++) {
				z = z_lb + l*dz;
				double point[4] = {x, y, x, z};
				likelihood += gauss.pdf(point);
			}
		}
	}
	
	likelihood *= dz * dx * dy;
	return likelihood;
}

double P_Y_less_X(double* expv, double* cov_m,
                  double max_y, double min_y, double dy,
				  double max_x, double min_x, double dx){
	// P(Y < X)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[2] = {expv[0], expv[1]};
	double projected_cov_m[4] = {cov_m[0], cov_m[1], cov_m[4], cov_m[5]};
	gauss.set(projected_expv, projected_cov_m, 2);
	
	unsigned int y_steps = (max_y - min_y) / dy;
	unsigned int x_steps;
	double y, x, x_lb;
	
	for (int i=0; i<=y_steps; i++) {
		y = min_y + i*dy;
		x_lb = min( max(y, min_x), max_x);
		x_steps = (max_x - x_lb) / dx;
		
		int k = 0;
		if (x_lb == y) k = 1;
		
		for (k; k<=x_steps; k++) {
			x = x_lb + k*dx;
			double point[2] = {x, y};
			likelihood += gauss.pdf(point);
		}
	}
	
	likelihood *= dx * dy;
	return likelihood;
}

double P_X_less_J(double* expv, double* cov_m,
                  double max_x, double min_x, double dx,
				  double max_j, double min_j, double dj){
	// P(X < J)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[2] = {expv[0], expv[2]};
	double projected_cov_m[4] = {cov_m[0], cov_m[2], cov_m[8], cov_m[10]};
	gauss.set(projected_expv, projected_cov_m, 2);
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int j_steps;
	double x, j, j_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		j_lb = min( max(x, min_j), max_j);
		j_steps = (max_j - j_lb) / dj;
		
		int k = 0;
		if (j_lb == x) k = 1;
		
		for (k; k<=x_steps; k++) {
			j = j_lb + k*dj;
			double point[2] = {x, j};
			likelihood += gauss.pdf(point);
		}
	}
	
	likelihood *= dj * dx;
	return likelihood;
}

double P_Y_less_X_less_J(double* expv, double* cov_m,
                         double max_y, double min_y, double dy,
						 double max_x, double min_x, double dx,
						 double max_j, double min_j, double dj){
	// P(Y < X < J)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[3] = {expv[0], expv[1], expv[2]};
	double projected_cov_m[9] = {cov_m[0], cov_m[1], cov_m[2],
	                             cov_m[4], cov_m[5], cov_m[6],
								 cov_m[8], cov_m[9], cov_m[10]};
	gauss.set(projected_expv, projected_cov_m, 3);
	
	unsigned int y_steps = (max_y - min_y) / dy;
	unsigned int x_steps, j_steps;
	double y, x, x_lb, j, j_lb;
	
	for (int i=0; i<=y_steps; i++) {
		y = min_y + i*dy;
		x_lb = min( max(y, min_x), max_x);
		x_steps = (max_x - x_lb) / dx;
		
		int k = 0;
		if (x_lb == y) k = 1;
		
		for (k; k<=x_steps; k++) {
			x = x_lb + k*dx;
			j_lb = min( max(x, min_j), max_j);
			j_steps = (max_j - j_lb) / dj;
			
			int l = 0;
			if (j_lb == x) l = 1;
			
			for (l; l<=j_steps; l++) {
				j = j_lb + l*dj;
				double point[3] = {x, y, j};
				likelihood += gauss.pdf(point);
			}
		}
	}
	
	likelihood *= dj * dx * dy;
	return likelihood;
}

double P_Y_less_X_J_less_Z(double* expv, double* cov_m,
                           double max_y, double min_y, double dy,
						   double max_x, double min_x, double dx,
						   double max_j, double min_j, double dj,
						   double max_z, double min_z, double dz){
	// P(Y < X, J < Z)
	Gauss gauss;
	gauss.set(expv, cov_m, 4);
	
	double likelihood = 0.0;
	
	unsigned int y_steps = (max_y - min_y) / dy;
	unsigned int x_steps, j_steps, z_steps;
	double y, x, x_lb, j, j_lb, z, z_lb;
	
	for (int i=0; i<=y_steps; i++) {
		y = min_y + i*dy;
		x_lb = min( max(y, min_x), max_x);
		x_steps = (max_x - x_lb) / dx;
		
		int k = 0;
		if (x_lb == y) k = 1;
		
		for (k; k<=x_steps; k++) {
			x = x_lb + k*dx;
			j_lb = min( max(x, min_j), max_j);
			j_steps = (max_j - j_lb) / dj;
			
			int l = 0;
			if (j_lb == x) l = 1;
			
			for (l; l<=j_steps; l++) {
				j = j_lb + l*dj;
				z_lb = min( max(j, min_z), max_z);
				z_steps = (max_z - z_lb) / dz;
				
				int m = 0;
				if (z_lb == j) m = 1;
				
				for (m; m<=z_steps; m++) {
					z = z_lb + m*dz;
					double point[4] = {x, y, j, z};
					likelihood += gauss.pdf(point);
				}
			}
		}
	}
	
	likelihood *= dz * dj * dx * dy;
	return likelihood;
}

double P_X_less_J_less_Z(double* expv, double* cov_m,
                         double max_x, double min_x, double dx,
						 double max_j, double min_j, double dj,
						 double max_z, double min_z, double dz){
	// P(X < J < Z)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[3] = {expv[0], expv[2], expv[3]};
	double projected_cov_m[9] = {cov_m[0], cov_m[2], cov_m[3],
	                             cov_m[8], cov_m[10], cov_m[11],
								 cov_m[12], cov_m[14], cov_m[15]};
	gauss.set(projected_expv, projected_cov_m, 3);
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int j_steps, z_steps;
	double x, j, j_lb, z, z_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		j_lb = min( max(x, min_j), max_j);
		j_steps = (max_j - j_lb) / dj;
		
		int k = 0;
		if (j_lb == x) k = 1;
		
		for (k; k<=j_steps; k++) {
			j = j_lb + k*dj;
			z_lb = min( max(j, min_z), max_z);
			z_steps = (max_z - z_lb) / dz;
			
			int l = 0;
			if (z_lb == j) l = 1;
			
			for (l; l<=z_steps; l++) {
				z = z_lb + l*dz;
				double point[3] = {x, j, z};
				likelihood += gauss.pdf(point);
			}
		}
	}
	
	likelihood *= dz * dj * dx;
	return likelihood;
}

double P_Y_less_X_less_J_less_Z(double* expv, double* cov_m,
                                double max_y, double min_y, double dy,
								double max_x, double min_x, double dx,
								double max_j, double min_j, double dj,
								double max_z, double min_z, double dz){
	// P(Y < X < J < Z)
	Gauss gauss;
	gauss.set(expv, cov_m, 4);
	
	double likelihood = 0.0;
	
	unsigned int y_steps = (max_y - min_y) / dy;
	unsigned int x_steps, j_steps, z_steps;
	double y, x, x_lb, j, j_lb, z, z_lb;
	
	for (int i=0; i<=y_steps; i++) {
		y = min_y + i*dy;
		x_lb = min( max(y, min_x), max_x);
		x_steps = (max_x - x_lb) / dx;
		
		int k = 0;
		if (x_lb == y) k = 1;
		
		for (k; k<=x_steps; k++) {
			x = x_lb + k*dx;
			j_lb = min( max(x, min_j), max_j);
			j_steps = (max_j - j_lb) / dj;
			
			int l = 0;
			if (j_lb == x) l = 1;
			
			for (l; l<=j_steps; l++) {
				j = j_lb + l*dj;
				z_lb = min( max(j, min_z), max_z);
				z_steps = (max_z - z_lb) / dz;
				
				int m = 0;
				if (z_lb == j) m = 1;
				
				for (m; m<=z_steps; m++) {
					z = z_lb + m*dz;
					double point[4] = {x, y, j, z};
					likelihood += gauss.pdf(point);
				}
			}
		}
	}
	
	likelihood *= dz * dj * dx * dy;
	return likelihood;
}

double P_Y_less_X_J_eq_Z(double* expv, double* cov_m,
                         double max_y, double min_y, double dy,
						 double max_x, double min_x, double dx,
						 double max_j, double min_j, double dj){
	// P(Y < X, J = Z)
	Gauss gauss;
	gauss.set(expv, cov_m, 4);
	
	double likelihood = 0.0;
	
	unsigned int y_steps = (max_y - min_y) / dy;
	unsigned int x_steps, j_steps;
	double y, x, x_lb, j, j_lb;
	
	for (int i=0; i<=y_steps; i++) {
		y = min_y + i*dy;
		x_lb = min( max(y, min_x), max_x);
		x_steps = (max_x - x_lb) / dx;
		
		int k = 0;
		if (x_lb == y) k = 1;
		
		for (k; k<=x_steps; k++) {
			x = x_lb + k*dx;
			j_lb = min( max(x, min_j), max_j);
			j_steps = (max_j - j_lb) / dj;
			
			int l = 0;
			if (j_lb == x) l = 1;
			
			for (l; l<=j_steps; l++) {
				j = j_lb + l*dj;
				double point[4] = {x, y, j, j};
				likelihood += gauss.pdf(point);
			}
		}
	}
	
	likelihood *= dj * dx * dy;
	return likelihood;
}

double P_X_less_J_eq_Z(double* expv, double* cov_m,
                       double max_x, double min_x, double dx,
					   double max_j, double min_j, double dj){
	// P(X < J = Z)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[3] = {expv[0], expv[2], expv[3]};
	double projected_cov_m[9] = {cov_m[0], cov_m[2], cov_m[3],
	                             cov_m[8], cov_m[10], cov_m[11],
								 cov_m[12], cov_m[14], cov_m[15]};
	gauss.set(projected_expv, projected_cov_m, 3);
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int j_steps;
	double x, j, j_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		j_lb = min( max(x, min_j), max_j);
		j_steps = (max_j - j_lb) / dj;
		
		int k = 0;
		if (j_lb == x) k = 1;
		
		for (k; k<=j_steps; k++) {
			j = j_lb + k*dj;
			double point[3] = {x, j, j};
			likelihood += gauss.pdf(point);
		}
	}
	
	likelihood *= dj * dx;
	return likelihood;
}

double P_Y_less_X_less_J_eq_Z(double* expv, double* cov_m,
                              double max_y, double min_y, double dy,
						      double max_x, double min_x, double dx,
						      double max_j, double min_j, double dj){
	// P(Y < X < J = Z)
	Gauss gauss;
	gauss.set(expv, cov_m, 4);
	
	double likelihood = 0.0;
	
	unsigned int y_steps = (max_y - min_y) / dy;
	unsigned int x_steps, j_steps;
	double y, x, x_lb, j, j_lb;
	
	for (int i=0; i<=y_steps; i++) {
		y = min_y + i*dy;
		x_lb = min( max(y, min_x), max_x);
		x_steps = (max_x - x_lb) / dx;
		
		int k = 0;
		if (x_lb == y) k = 1;
		
		for (k; k<=x_steps; k++) {
			x = x_lb + k*dx;
			j_lb = min( max(x, min_j), max_j);
			j_steps = (max_j - j_lb) / dj;
			
			int l = 0;
			if (j_lb == x) l = 1;
			
			for (l; l<=j_steps; l++) {
				j = j_lb + l*dj;
				double point[4] = {x, y, j, j};
				likelihood += gauss.pdf(point);
			}
		}
	}
	
	likelihood *= dj * dx * dy;
	return likelihood;
}

double P_X_eq_Y_eq_J(double* expv, double* cov_m, double max_x, double min_x, double dx){
	// P(X = Y = J)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[3] = {expv[0], expv[1], expv[2]};
	double projected_cov_m[9] = {cov_m[0], cov_m[1], cov_m[2],
	                             cov_m[4], cov_m[5], cov_m[6],
								 cov_m[8], cov_m[9], cov_m[10],};
	gauss.set(projected_expv, projected_cov_m, 3);
	
	unsigned int x_steps = (max_x - min_x) / dx;
	double x;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		double point[3] = {x, x, x};
		likelihood += gauss.pdf(point);
	}
	
	likelihood *= dx;
	return likelihood;
}

double P_X_eq_Y_eq_J_less_Z(double* expv, double* cov_m,
                            double max_x, double min_x, double dx,
							double max_z, double min_z, double dz){
	// P(X = Y = J < Z)
	Gauss gauss;
	gauss.set(expv, cov_m, 4);
	
	double likelihood = 0.0;
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int z_steps;
	double x, z, z_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		z_lb = min( max(x, min_z), max_z);
		z_steps = (max_z - z_lb) / dz;
		
		int k = 0;
		if (z_lb == x) k = 1;
		
		for (k; k<=z_steps; k++) {
			z = z_lb + k*dz;
			double point[4] = {x, x, x, z};
			likelihood += gauss.pdf(point);
		}
	}
	
	likelihood *= dz * dx;
	return likelihood;
}

double P_X_eq_Y(double* expv, double* cov_m, double max_x, double min_x, double dx){
	// P(X = Y)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[2] = {expv[0], expv[1]};
	double projected_cov_m[4] = {cov_m[0], cov_m[1], cov_m[4], cov_m[5]};
	gauss.set(projected_expv, projected_cov_m, 2);
	
	unsigned int x_steps = (max_x - min_x) / dx;
	double x;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		double point[2] = {x, x};
		likelihood += gauss.pdf(point);
	}
	
	likelihood *= dx;
	return likelihood;
}

double P_X_eq_Y_less_J(double* expv, double* cov_m,
                       double max_x, double min_x, double dx,
					   double max_j, double min_j, double dj){
	// P(X = Y < J)
	Gauss gauss;
	double likelihood = 0.0;
	
	double projected_expv[3] = {expv[0], expv[1], expv[2]};
	double projected_cov_m[9] = {cov_m[0], cov_m[1], cov_m[2],
	                             cov_m[4], cov_m[5], cov_m[6],
								 cov_m[8], cov_m[9], cov_m[10]};
	gauss.set(projected_expv, projected_cov_m, 3);
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int j_steps;
	double x, j, j_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		j_lb = min( max(x, min_j), max_j);
		j_steps = (max_j - j_lb) / dj;
		
		int k = 0;
		if (j_lb == x) k = 1;
		
		for (k; k<=j_steps; k++) {
			j = j_lb + k*dj;
			double point[3] = {x, x, j};
			likelihood += gauss.pdf(point);
		}
	}
	
	likelihood *= dj * dx;
	return likelihood;
}

double P_X_eq_Y_J_less_Z(double* expv, double* cov_m,
                         double max_x, double min_x, double dx,
						 double max_j, double min_j, double dj,
						 double max_z, double min_z, double dz){
	// P(X = Y, J < Z)
	Gauss gauss;
	gauss.set(expv, cov_m, 4);
	
	double likelihood = 0.0;
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int j_steps, z_steps;
	double x, j, j_lb, z, z_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		j_lb = min( max(x, min_j), max_j);
		j_steps = (max_j - j_lb) / dj;
		
		int k = 0;
		if (j_lb == x) k = 1;
		
		for (k; k<=j_steps; k++) {
			j = j_lb + k*dj;
			z_lb = min( max(j, min_z), max_z);
			z_steps = (max_z - z_lb) / dz;
			
			int l = 0;
			if (z_lb == j) l = 1;
			
			for (l; l<=z_steps; l++) {
				z = z_lb + l*dz;
				double point[4] = {x, x, j, z};
				likelihood += gauss.pdf(point);
			}
		}
	}
	
	likelihood *= dz * dj * dx;
	return likelihood;
}

double P_X_eq_Y_less_J_less_Z(double* expv, double* cov_m,
                         double max_x, double min_x, double dx,
						 double max_j, double min_j, double dj,
						 double max_z, double min_z, double dz){
	// P(X = Y < J < Z)
	Gauss gauss;
	gauss.set(expv, cov_m, 4);
	
	double likelihood = 0.0;
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int j_steps, z_steps;
	double x, j, j_lb, z, z_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		j_lb = min( max(x, min_j), max_j);
		j_steps = (max_j - j_lb) / dj;
		
		int k = 0;
		if (j_lb == x) k = 1;
		
		for (k; k<=j_steps; k++) {
			j = j_lb + k*dj;
			z_lb = min( max(j, min_z), max_z);
			z_steps = (max_z - z_lb) / dz;
			
			int l = 0;
			if (z_lb == j) l = 1;
			
			for (l; l<=z_steps; l++) {
				z = z_lb + l*dz;
				double point[4] = {x, x, j, z};
				likelihood += gauss.pdf(point);
			}
		}
	}
	
	likelihood *= dz * dj * dx;
	return likelihood;
}

double P_X_eq_Y_J_eq_Z(double* expv, double* cov_m,
                       double max_x, double min_x, double dx,
					   double max_j, double min_j, double dj){
	// calculate P(X = Y, J = Z)
	Gauss gauss;
	gauss.set(expv, cov_m, 4);
	
	double likelihood = 0.0;
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int j_steps;
	double x, j, j_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		j_lb = min( max(x, min_j), max_j);
		j_steps = (max_j - j_lb) / dj;
		
		int k = 0;
		if (j_lb == x) k = 1;
		
		for (k; k<=j_steps; k++) {
			j = j_lb + k*dj;
			double point[4] = {x, x, j, j};
			likelihood += gauss.pdf(point);
		}
	}
	
	likelihood *= dj * dx;
	return likelihood;
}

double P_X_eq_Y_less_J_eq_Z(double* expv, double* cov_m,
                            double max_x, double min_x, double dx,
					        double max_j, double min_j, double dj){
	// calculate P(X = Y < J = Z)
	Gauss gauss;
	gauss.set(expv, cov_m, 4);
	
	double likelihood = 0.0;
	
	unsigned int x_steps = (max_x - min_x) / dx;
	unsigned int j_steps;
	double x, j, j_lb;
	
	for (int i=0; i<=x_steps; i++) {
		x = min_x + i*dx;
		j_lb = min( max(x, min_j), max_j);
		j_steps = (max_j - j_lb) / dj;
		
		int k = 0;
		if (j_lb == x) k = 1;
		
		for (k; k<=j_steps; k++) {
			j = j_lb + k*dj;
			double point[4] = {x, x, j, j};
			likelihood += gauss.pdf(point);
		}
	}
	
	likelihood *= dj * dx;
	return likelihood;
}

double prob_A11(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
	// calculate P(X < Y = J < Z)
	// check for certificate of case impossibility
	if ( (max_y < min_x) || (max_y < min_j) )
		return 0.0;
	
	double likelihood = 0.0;
	
	if (max_y < min_z) {
		// P(X<Y=J<Z) = P(X<Y=J)
		if (max_x < min_y) {
			// P(X<Y=J) = P(Y=J)
			likelihood = P_Y_eq_J(expv, cov_m, max_y, min_y, dy);
		} else {
			// P(X<Y=J)
			likelihood = P_X_less_Y_eq_J(expv, cov_m, max_x, min_x, dx, max_y, min_y, dy);
		}
	} else {
		if (max_x < min_y) {
			// P(Y=J<Z)
			likelihood = P_Y_eq_J_less_Z(expv, cov_m, max_y, min_y, dy, max_z, min_z, dz);
		} else {
			// P(X<Y=J<Z)
			likelihood = P_X_less_Y_eq_J_less_Z(expv, cov_m, max_x, min_x, dx, max_y, min_y, dy, max_z, min_z, dz);
		}
	}
	
	return likelihood;
}

double prob_A12(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
	// calculate P(X < Y < J < Z)
	
	// check for certificate of case impossibility
	if ( (max_y < min_x) || (max_j < min_y) ){
		return 0.0;
	}
	
	double likelihood = 0.0;
	
	if (max_j < min_z) {
		// P(X<Y<J<Z) = P(X<Y<J)
		if (max_y < min_j) {
			// P(X<Y<J) = P(X<Y)
			if (max_x < min_y) {
				// P(X<Y) = 1
				likelihood = 1.0 / (dx * dy * dj * dz);
			} else {
				// P(X<Y)
				likelihood = P_X_less_Y(expv, cov_m, max_x, min_x, dx, max_y, min_y, dy);
			}
		} else {
			// P(X<Y<J)
			if (max_x < min_y) {
				// P(X<Y<J) = P(Y<J)
				likelihood = P_Y_less_J(expv, cov_m, max_y, min_y, dy, max_j, min_j, dj);
			} else {
				// P(X<Y<J)
				likelihood = P_X_less_Y_less_J(expv, cov_m, max_x, min_x, dx, max_y, min_y, dy, max_j, min_j, dj);
			}
		}
	} else {
		// P(X<Y<J<Z)
		if (max_y < min_j) {
			// P(X<Y<J<Z) = P(X<Y, X<J, J<Z)
			if (max_x < min_y) {
				// P(X<Y, J<Z) = P(J<Z)
				likelihood = P_J_less_Z(expv, cov_m, max_j, min_j, dj, max_z, min_z, dz);
			} else {
				// P(X<Y, J<Z)
				// j must go from at least x
				likelihood = P_X_less_Y_J_less_Z(expv, cov_m, max_x, min_x, dx, max_y, min_y, dy, max_j, min_j, dj, max_z, min_z, dz);
			}
		} else {
			// P(X<Y<J<Z)
			if (max_x < min_y) {
				// P(X<Y<J<Z) = P(Y<J<Z)
				likelihood = P_Y_less_J_less_Z(expv, cov_m, max_y, min_y, dy, max_j, min_j, dj, max_z, min_z, dz);
			} else {
				// P(X<Y<J<Z)
				likelihood = P_X_less_Y_less_J_less_Z(expv, cov_m, max_x, min_x, dx, max_y, min_y, dy, max_j, min_j, dj, max_z, min_z, dz);
			}
		}
	}
	
	return likelihood;
}

double prob_A13(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
	// calculate P(X < Y < J = Z)
    double likelihood = 0.0;
	
	// check for certificate of impossibility
	if (max_y < min_x) return 0.0;
	
	if (max_j < min_z) {
		return 0.0;
	} else {
		// P(X < Y < J = Z)
		if (max_y < min_j) {
			// P(X < Y < J = Z) = P(X < Y, J = Z)
			if (max_x < min_y) {
				// P(J = Z)
				likelihood = P_J_eq_Z(expv, cov_m, max_j, min_j, dj);
			} else {
				// P(X < Y, J = Z)
				likelihood = P_X_less_Y_J_eq_Z(expv, cov_m, max_x, min_x, dx, max_y, min_y, dy, max_j, min_j, dj);
			}
		} else {
			// P(X < Y < J = Z)
			if (max_x < min_y) {
				// P(Y < J = Z)
				likelihood = P_Y_less_J_eq_Z(expv, cov_m, max_y, min_y, dy, max_j, min_j, dj);
			} else {
				// P(X < Y < J = Z)
				likelihood = P_X_less_Y_less_J_eq_Z(expv, cov_m, max_x, min_x, dx, max_y, min_y, dy, max_j, min_j, dj);
			}
		}
	}

    return likelihood;
}

double prob_A21(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
	// calculate P(Y < X = J < Z)
    double likelihood = 0.0;

	// check for certificate of impossibility
	if ((max_x < min_y) || (max_x < min_j)) return 0.0;
	
	if (max_j < min_z) {
		// P(Y < X = J)
		if (max_y < min_x) {
			// P(X = J)
			likelihood = P_X_eq_J(expv, cov_m, max_x, min_x, dx);
		} else {
			// P(Y < X = J)
			likelihood = P_Y_less_X_eq_J(expv, cov_m, max_y, min_y, dy, max_x, min_x, dx);
		}
	} else {
		// P(Y < X = J < Z)
		if (max_y < min_x) {
			// P(X = J < Z)
			likelihood = P_X_eq_J_less_Z(expv, cov_m, max_x, min_x, dx, max_z, min_z, dz);
		} else {
			// P(Y < X = J < Z)
			likelihood = P_Y_less_X_eq_J_less_Z(expv, cov_m, max_y, min_y, dy, max_x, min_x, dx, max_z, min_z, dz);
		}
	}
	
    return likelihood;
}

double prob_A22(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
	// calculate P(Y < X < J < Z)
    double likelihood = 0.0;
	
	// check for certificate of impossibility
	if (max_x < min_y) return 0.0;

	if (max_j < min_z) {
		// P(Y < X < J)
		if (max_x < min_j) {
			// P(Y < X)
			if (max_y < min_x) {
				likelihood = 1.0 / (dx * dy * dj * dz);
			} else {
				// P(Y < X)
				likelihood = P_Y_less_X(expv, cov_m, max_y, min_y, dy, max_x, min_x, dx);
			}
		} else {
			// P(Y < X < J)
			if (max_y < min_x) {
				// P(X < J)
				likelihood = P_X_less_J(expv, cov_m, max_x, min_x, dx, max_j, min_j, dj);
			} else {
				// P(Y < X < J)
				likelihood = P_Y_less_X_less_J(expv, cov_m, max_y, min_y, dy, max_x, min_x, dx, max_j, min_j, dj);
			}
		}
	} else {
		// P(Y < X < J < Z)
		if (max_x < min_j) {
			// P(Y < X, J < Z)
			if (max_y < min_x) {
				// P(J < Z)
				likelihood = P_J_less_Z(expv, cov_m, max_j, min_j, dj, max_z, min_z, dz);
			} else {
				// P(Y < X, J < Z)
				likelihood = P_Y_less_X_J_less_Z(expv, cov_m, max_y, min_y, dy, max_x, min_x, dx, max_j, min_j, dj, max_z, min_z, dz);
			}
		} else {
			// P(Y < X < J < Z)
			if (max_y < min_x) {
				// P(X < J < Z)
				likelihood = P_X_less_J_less_Z(expv, cov_m, max_x, min_x, dx, max_j, min_j, dj, max_z, min_z, dz);
			} else {
				// P(Y < X < J < Z)
				likelihood = P_Y_less_X_less_J_less_Z(expv, cov_m, max_y, min_y, dy, max_x, min_x, dx, max_j, min_j, dj, max_z, min_z, dz);
			}
		}
	}
    
    return likelihood;
}

double prob_A23(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
	// calculate P(Y < X < J = Z)
    double likelihood = 0.0;
	
	// check for certificate of impossibility
	if ((max_x < min_y) || (max_j < min_z)) return 0.0;

	if (max_j < min_z) {
		likelihood = 0.0;
	} else {
		// P(Y < X < J = Z)
		if (max_x < min_j) {
			// P(Y < X, J = Z)
			if (max_y < min_x) {
				// P(J = Z)
				likelihood = P_J_eq_Z(expv, cov_m, max_j, min_j, dj);
			} else {
				// P(Y < X, J = Z)
				likelihood = P_Y_less_X_J_eq_Z(expv, cov_m, max_y, min_y, dy, max_x, min_x, dx, max_j, min_j, dj);
			}
		} else {
			// P(Y < X < J = Z)
			if (max_y < min_x) {
				// P(X < J = Z)
				likelihood = P_X_less_J_eq_Z(expv, cov_m, max_x, min_x, dx, max_j, min_j, dj);
			} else {
				// P(Y < X < J = Z)
				likelihood = P_Y_less_X_less_J_eq_Z(expv, cov_m, max_y, min_y, dy, max_x, min_x, dx, max_j, min_j, dj);
			}
		}
	}
   
    return likelihood;
}

double prob_A31(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
	// calculate P(X = Y = J < Z)
    double likelihood = 0.0;
	
	if (max_j < min_z) {
		// P(X = Y = J)
		if (max_y < min_j) {
			likelihood = 0.0;
		} else {
			// P(X = Y = J)
			if (max_x < min_y) {
				likelihood = 0.0;
			} else {
				// P(X = Y = J)
				likelihood = P_X_eq_Y_eq_J(expv, cov_m, max_x, min_x, dx);
			}
		}
	} else {
		// P(X = Y = J < Z)
		if (max_y < min_j) {
			likelihood = 0.0;
		} else {
			// P(X = Y = J < Z)
			if (max_x < min_y) {
				likelihood = 0.0;
			} else {
				// P(X = Y = J < Z)
				likelihood = P_X_eq_Y_eq_J_less_Z(expv, cov_m, max_x, min_x, dx, max_z, min_z, dz);
			}
		}
	}
    
    return likelihood;	
}

double prob_A32(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
	// calculate P(X = Y < J < Z)
    double likelihood = 0.0;

	if (max_j < min_z) {
		// P(X = Y < J)
		if (max_y < min_j) {
			// P(X = Y)
			if (max_x < min_y) {
				likelihood = 0.0;
			} else {
				// P(X = Y)
				likelihood = P_X_eq_Y(expv, cov_m, max_x, min_x, dx);
			}
		} else {
			// P(X = Y < J)
			if (max_x < min_y) {
				likelihood = 0.0;
			} else {
				// P(X = Y < J)
				likelihood = P_X_eq_Y_less_J(expv, cov_m, max_x, min_x, dx, max_j, min_j, dj);
			}
		}
	} else {
		// P(X = Y < J < Z)
		if (max_y < min_j) {
			// P(X = Y, J < Z)
			if (max_x < min_y) {
				likelihood = 0.0;
			} else {
				// P(X = Y, J < Z)
				likelihood = P_X_eq_Y_J_less_Z(expv, cov_m, max_x, min_x, dx, max_j, min_j, dj, max_z, min_z, dz);
			}
		} else {
			// P(X = Y < J < Z)
			if (max_x < min_y) {
				likelihood = 0.0;
			} else {
				// P(X = Y < J < Z)
				likelihood = P_X_eq_Y_less_J_less_Z(expv, cov_m, max_x, min_x, dx, max_j, min_j, dj, max_z, min_z, dz);
			}
		}
	}
    
    return likelihood;
}

double prob_A33(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
	// calculate P(X = Y < J = Z)
    double likelihood = 0.0;

	if (max_j < min_z) {
		likelihood = 0.0;
	} else {
		// calculate P(X = Y < J = Z)
		if (max_y < min_j) {
			// calculate P(X = Y, J = Z)
			if (max_x < min_y) {
				likelihood = 0.0;
			} else {
				// calculate P(X = Y, J = Z)
				likelihood = P_X_eq_Y_J_eq_Z(expv, cov_m, max_x, min_x, dx, max_j, min_j, dj);
			}
		} else {
			// calculate P(X = Y < J = Z)
			if (max_x < min_y) {
				likelihood = 0.0;
			} else {
				// calculate P(X = Y < J = Z)
				likelihood = P_X_eq_Y_less_J_eq_Z(expv, cov_m, max_x, min_x, dx, max_j, min_j, dj);
			}
		}
	}
    
    return  likelihood;
}

void fit_gauss(double** dims, unsigned int n, double eff_sample_size, double** expv_p, double** cov_m_p){
	*expv_p = new double[4];
	double* expv = *expv_p;
	
	#pragma omp parallel sections
	{
		#pragma omp section
		expv[0] = exp_val(dims[0], n);
		
		#pragma omp section
		expv[1] = exp_val(dims[1], n);
		
		#pragma omp section
		expv[2] = exp_val(dims[2], n);
		
		#pragma omp section
		expv[3] = exp_val(dims[3], n);
	}
	
	*cov_m_p = cov_matr(dims, expv, n);
	double* cov_m = *cov_m_p;
	
	for (int i=0; i<16; i++) cov_m[i] *= eff_sample_size / n;
}

double* get_probabilities(double* expv, double* cov_m, double c, double bins){
	
	double min_x = max( expv[0] - c * sqrt(cov_m[0]) , 0.0);
	double max_x = expv[0] + c * sqrt(cov_m[0]);
    double dx = (max_x - min_x) / bins;

    double min_y = max( expv[1] - c * sqrt(cov_m[5]), 0.0);
	double max_y = expv[1] + c * sqrt(cov_m[5]);
    double dy = (max_y - min_y) / bins;
	
	double min_j = max( expv[2] - c * sqrt(cov_m[10]), 0.0);
	double max_j = expv[2] + c * sqrt(cov_m[10]);
    double dj = (max_j - min_j) / bins;
	
	double min_z = max( expv[3] - c * sqrt(cov_m[15]), 0.0);
	double max_z = expv[3] + c * sqrt(cov_m[15]);
    double dz = (max_z - min_z) / bins;
	
	// check if the dimensions satisfy the feasibility conditions
	// return zeros to indicate that probabilities can't be calculated
	if (!check_feasibility(min_x, min_y, min_j, max_j, max_z)){
		delete[] cov_m;
		double* case_probs = new double[5]{0.0, 0.0, 0.0, 0.0, 0.0};
		return case_probs;
	}
	
	double* likelihoods = new double[9]{0, 0, 0, 0, 0, 0, 0, 0, 0};
	
	#pragma omp parallel sections
	{
		#pragma omp section
		likelihoods[0] = prob_A11(expv, cov_m, min_x, max_x, min_y, max_y, min_j, max_j, min_z, max_z, dx, dy, dj, dz);
		
		#pragma omp section
		likelihoods[1] = prob_A12(expv, cov_m, min_x, max_x, min_y, max_y, min_j, max_j, min_z, max_z, dx, dy, dj, dz);
		
		#pragma omp section
		likelihoods[2] = prob_A13(expv, cov_m, min_x, max_x, min_y, max_y, min_j, max_j, min_z, max_z, dx, dy, dj, dz);
		
		#pragma omp section
		likelihoods[3] = prob_A21(expv, cov_m, min_x, max_x, min_y, max_y, min_j, max_j, min_z, max_z, dx, dy, dj, dz);
		
		#pragma omp section
		likelihoods[4] = prob_A22(expv, cov_m, min_x, max_x, min_y, max_y, min_j, max_j, min_z, max_z, dx, dy, dj, dz);
		
		#pragma omp section
		likelihoods[5] = prob_A23(expv, cov_m, min_x, max_x, min_y, max_y, min_j, max_j, min_z, max_z, dx, dy, dj, dz);
		
		#pragma omp section
		likelihoods[6] = prob_A31(expv, cov_m, min_x, max_x, min_y, max_y, min_j, max_j, min_z, max_z, dx, dy, dj, dz);
		
		#pragma omp section
		likelihoods[7] = prob_A32(expv, cov_m, min_x, max_x, min_y, max_y, min_j, max_j, min_z, max_z, dx, dy, dj, dz);
		
		#pragma omp section
		likelihoods[8] = prob_A33(expv, cov_m, min_x, max_x, min_y, max_y, min_j, max_j, min_z, max_z, dx, dy, dj, dz);
	}
	
	double normalizing_term = 0.0;
	
	for (int i=0; i<9; i++) normalizing_term += likelihoods[i];	
	for (int i=0; i<9; i++) likelihoods[i] /= normalizing_term;
	
	double p_x_causes_y, p_y_causes_x, p_circular, p_common_cause, p_independence;
	p_x_causes_y = likelihoods[0];
	p_y_causes_x = likelihoods[3];
	p_circular = likelihoods[6];
	p_common_cause = likelihoods[1] + likelihoods[4] + likelihoods[7];
	p_independence = likelihoods[2] + likelihoods[5] + likelihoods[8];
	
	delete[] likelihoods;
	delete[] cov_m;
	
	double* case_probs = new double[5]{p_x_causes_y, p_circular, p_y_causes_x, p_common_cause, p_independence};

	return case_probs;
}
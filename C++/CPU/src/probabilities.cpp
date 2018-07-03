#include <iostream>
#include <cmath>
#include "statistics.h"

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
		unsigned int dim;
		double* expv = NULL;
		double* inv_cov_m = NULL;
		double det;
		double norm_term;
};

Gauss::~Gauss(){
	if (expv != NULL) delete[] expv;
	if (inv_cov_m != NULL) delete[] inv_cov_m;
}

void Gauss::set(double* expv, double* cov_m, unsigned int dim){
	this->dim = dim;
	this->expv = expv;
	
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


void check_feasibility(double min_x, double min_y, double min_j, double max_j, double max_z){
	if (min_x > max_j) throw std::invalid_argument( "D_X > D_J, which should be impossible! Verify that your data is not stochastic." );
	if (min_y > max_j) throw std::invalid_argument( "D_Y > D_J, which should be impossible! Verify that your data is not stochastic." );
	if (min_x > max_z) throw std::invalid_argument( "D_X > D_Z, which should be impossible! Verify that your data is not stochastic." );
	if (min_y > max_z) throw std::invalid_argument( "D_Y > D_J, which should be impossible! Verify that your data is not stochastic." );
	if (min_j > max_z) throw std::invalid_argument( "D_J > D_Z, which should be impossible! Verify that your data is not stochastic." );
}


double prob_A11( double* expv, double* cov_m,
				double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z,
				double dx, double dy, double dj, double dz){
	// calculate P(X < Y = J < Z)
	
	// check for certificate of case impossibility
	if ( (max_y < min_x) || (max_y < min_j) )
		return 0.0;
	
	double likelihood = 0.0;
	Gauss gauss;
	
	if (max_y < min_z) {
		// P(X<Y=J<Z) = P(X<Y=J)
		if (max_x < min_y) {
			// P(X<Y=J) = P(Y=J)
			double projected_expv[2] = {expv[1], expv[2]};
			double projected_cov_m[4] = {cov_m[5], cov_m[6], cov_m[9], cov_m[10]};
			gauss.set(projected_expv, projected_cov_m, 2);
			
			unsigned int y_steps = (max_y - min_y) / dy;
			double y;
				
			for (int i=0; i<=y_steps; i++) {
				y = min_y + i*dy;
				double point[2] = {y, y};
				likelihood += gauss.pdf(point) * dy;
			}
		} else {
			// P(X<Y=J)
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
				
				for (int k=1; k<=y_steps; k++) {
					y = y_lb + k*dy;
					double point[3] = {x, y, y};
					likelihood += gauss.pdf(point) * dy * dx;
				}
			}
		}
	} else {
		if (max_x < min_y) {
			// P(Y=J<Z)
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
				
				for (int k=1; k<=z_steps; k++){
					z = z_lb + k*dz;
					double point[3] = {y, y, z};
					likelihood += gauss.pdf(point) * dz * dy;
				}
			}
		} else {
			// P(X<Y=J<Z)
			gauss.set(expv, cov_m, 4);
			
			unsigned int x_steps = (max_x - min_x) / dx;
			unsigned int y_steps, z_steps;
			double x, y, z, y_lb, z_lb;
			
			for (int i=0; i<=x_steps; i++) {
				x = min_x + i*dx;
				y_lb = min( max(x, min_y), max_y );
				y_steps = (max_y - y_lb) / dy;
				
				for (int k=1; k<=y_steps; k++) {
					y = y_lb + k*dy;
					z_lb = min( max(y, min_z), max_z );
					z_steps = (max_z - z_lb) / dz;
					
					for (int l=1; l<=z_steps; l++) {
						z = z_lb + l*dz;
						double point[4] = {x, y, y, z};
						likelihood += gauss.pdf(point) * dz * dy * dx;
					}
				}
			}
		}
	}
	
	return likelihood;
}





double prob_A12(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
    // calculate P(X < Y < J < Z)
    double likelihood = 0.0;

    if (max_x < min_y) { // P(X < Y < J < Z) = P(Y < J < Z)
        if (max_y < min_j) { // P(Y < J < Z) = P(J < Z)
            if (max_j < min_z) {
                likelihood = 1.0 / (dz * dj * dy * dx); // TODO: validate this idea
            } else { // calculate P(J < Z)
				Gauss gauss;
				double projected_expv[2] = {expv[2], expv[3]};
				double projected_cov_m[4] = {cov_m[10], cov_m[11], cov_m[14], cov_m[15]};
				gauss.set(projected_expv, projected_cov_m, 2);
				
				unsigned int j_steps = (max_j - min_j) / dj;
				double j, z;
				
				//#pragma omp parallel for private(j) reduction(+:likelihood)
				for (int i=0; i<j_steps; i++) {
					j = min_j + i*dj;
					unsigned int z_steps = (max_z - j) / dz;
					
					//#pragma omp parallel for private(z) reduction(+:likelihood)
					for (int k=0; k<z_steps; k++) {
						z = j + k*dz;
						double point[2] = {j, z};
						likelihood += gauss.pdf(point) * dz * dj;
					}
				}
			}
        } else { // calculate P(Y < J < Z)
            if (max_j < min_z) { // P(Y < J < Z) = P(Y < J)
				Gauss gauss;
				double projected_expv[2] = {expv[1], expv[2]};
				double projected_cov_m[4] = {cov_m[1], cov_m[2], cov_m[5], cov_m[6]};
				gauss.set(projected_expv, projected_cov_m, 2);
				
				unsigned int y_steps = (max_y - min_y) / dy;				
				double y, j;
				
				//#pragma omp parallel for private(y) reduction(+:likelihood)
				for (int i=0; i<y_steps; i++) {
					y = min_y + i*dy;
					unsigned int j_steps = (max_j - y) / dj;
					
					//#pragma omp parallel for private(j) reduction(+:likelihood)
					for (int k=0; k<j_steps; k++) {
						j = y + k*dj;
						double point[2] = {y, j};
						likelihood += gauss.pdf(point) * dj * dy;
					}
				}
            } else { // calculate P(Y < J < Z)
				Gauss gauss;
				double projected_expv[3] = {expv[1], expv[2], expv[3]};
				double projected_cov_m[9] = {cov_m[5], cov_m[6], cov_m[7],
				                            cov_m[9], cov_m[10], cov_m[11],
				                            cov_m[13], cov_m[14], cov_m[15]};
				gauss.set(projected_expv, projected_cov_m, 3);
				
				unsigned int y_steps = (max_y - min_y) / dy;
				double y, j, z;
				
				//#pragma omp parallel for private(y) reduction(+:likelihood)
				for (int i=0; i<y_steps; i++) {
					y = min_y + i*dy;
					unsigned int j_steps = (max_j - y) / dy;
					
					//#pragma omp parallel for private(j) reduction(+:likelihood)
					for (int k=0; k<j_steps; k++) {
						j = y + k*dj;
						unsigned int z_steps = (max_z - j) / dz;
						
						//#pragma omp parallel for private(z) reduction(+:likelihood)
						for (int l=0; l<z_steps; l++) {
							z = j + l*dz;
							double point[3] = {y, j, z};
							likelihood += gauss.pdf(point) * dz * dj * dy;
						}
					}
				}
			}
		}
    } else { // calculate P(X < Y < J < Z)
        if (max_y < min_j) { // P(X < Y < J < Z) = P(X < Y, J < Z)
            if (max_j < min_z) { // P(X < Y, J < Z) = P(X < Y)
				Gauss gauss;
				double projected_expv[2] = {expv[0], expv[1]};
				double projected_cov_m[4] = {cov_m[0], cov_m[1], cov_m[4], cov_m[5]};
				gauss.set(projected_expv, projected_cov_m, 2);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				double x, y;
                
				//#pragma omp parallel for private(x) reduction(+:likelihood)
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					unsigned int y_steps = (max_y - x) / dy;
					
					//#pragma omp parallel for private(y) reduction(+:likelihood)
					for (int j=0; j<y_steps; j++) {
						y = x + j*dy;
						double point[2] = {x, y};
						likelihood += gauss.pdf(point) * dy * dx;
					}
				}
            } else { // calculate P(X < Y, J < Z)
				Gauss gauss;
				gauss.set(expv, cov_m, 4);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				unsigned int j_steps = (max_j - min_j) / dj;
				double x, y, j, z;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					unsigned int y_steps = (max_y - x) / dy;
					
					for (int k=0; k<y_steps; k++) {
						y = x + k*dy;
						
						for (int l=0; l<j_steps; l++) {
							j = min_j + l*dj;
							unsigned int z_steps = (max_z - j) / dz;
							
							for (int m=0; m<z_steps; m++) {
								z = j + m*dz;
								double point[4] = {x, y, j, z};
								likelihood += gauss.pdf(point) * dz * dj * dy * dx;
							}
						}
					}
				}
			}
        } else { // calculate P(X < Y < J < Z)
            if (max_j < min_z) { // P(X < Y < J < Z) = P(X < Y < J)
				Gauss gauss;
				double projected_expv[3] = {expv[0], expv[1], expv[2]};
				double projected_cov_m[9] = {cov_m[0], cov_m[1], cov_m[2],
				                            cov_m[4], cov_m[5], cov_m[6],
											cov_m[8], cov_m[9], cov_m[11]};
				gauss.set(projected_expv, projected_cov_m, 3);
			
				unsigned int x_steps = (max_x - min_x) / dx;
				double x, y, j;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					unsigned int y_steps = (max_y - x) / dy;
					
					for (int k=0; k<y_steps; k++) {
						y = x + k*dy;
						unsigned int j_steps = (max_j - y) / dj;
						
						for (int l=0; l<j_steps; l++) {
							j = y + l*dj;
							double point[3] = {x, y, j};
							likelihood += gauss.pdf(point) * dj * dy * dx;
						}
					}
				}
            } else { // calculate P(X < Y < J < Z)
				Gauss gauss;
				gauss.set(expv, cov_m, 4);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				double x, y, j, z;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					unsigned int y_steps = (max_y - x) / dy;
					
					for (int k=0; k<y_steps; k++) {
						y = x + k*dy;
						unsigned int j_steps = (max_j - y) / dj;
						
						for (int l=0; l<j_steps; l++) {
							j = y + l*dj;
							unsigned int z_steps = (max_z - j) / dz;
							
							for (int m=0; m<z_steps; m++) {
								z = j + m*dz;
								double point[4] = {x, y, j, z};
								likelihood += gauss.pdf(point) * dz * dj * dy * dx;
							}
						}
					}
				}
			}
		}
	}
    return likelihood;
}

double prob_A13(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
	// calculate P(X < Y < J = Z)
    double likelihood = 0.0;

    if (max_x < min_y) { // P(X < Y < J = Z) = P(Y < J = Z)
        if (max_y < min_j) { // P(Y < J = Z) = P(J = Z)
            if (max_j < min_z) { // P(J = Z) = 0
                likelihood = 0.0;
            } else { // calculate P(J = Z)
				Gauss gauss;
				double projected_expv[2] = {expv[2], expv[3]};
				double projected_cov_m[4] = {cov_m[10], cov_m[11], cov_m[14], cov_m[15]};
				gauss.set(projected_expv, projected_cov_m, 2);
				
				unsigned int j_steps = (max_j - min_j) / dj;
				double j;
				
				for (int i=0; i<j_steps; i++) {
					j = min_j + i*dj;
					double point[2] = {j, j};
					likelihood += gauss.pdf(point) * dj;
				}
			}
        } else { // calculate P(Y < J = Z)
            if (max_j < min_z) { // P(Y < J = Z) = 0
                likelihood = 0.0;
            } else { // calculate P(Y < J = Z)
				Gauss gauss;
				double projected_expv[3] = {expv[1], expv[2], expv[3]};
				double projected_cov_m[9] = {cov_m[5], cov_m[6], cov_m[7],
											cov_m[9], cov_m[10], cov_m[11],
											cov_m[13], cov_m[14], cov_m[15]};
				gauss.set(projected_expv, projected_cov_m, 3);
				
				unsigned int y_steps = (max_y - min_y) / dy;
				double y, j, z;
				
				for (int i=0; i<y_steps; i++) {
					y = min_y + i*dy;
					unsigned int j_steps = (max_j - y) / dj;
					
					for (int k=0; k<j_steps; k++) {
						j = y + k*dj;
						double point[3] = {y, j, j};
						likelihood += gauss.pdf(point) * dj * dy;
					}
				}
			}
		}
    } else { // calculate P(X < Y < J = Z)
        if (max_y < min_j) { // P(X < Y < J = Z) = P(X < Y, J = Z)
            if (max_j < min_z) { // P(X < Y, J = Z) = 0
                likelihood = 0.0;
            } else { // calculate P(X < Y, J = Z)
				Gauss gauss;
				gauss.set(expv, cov_m, 4);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				unsigned int j_steps = (max_j - min_j) / dj;
				double x, y, j;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					unsigned int y_steps = (max_y - x) / dy;
					
					for (int k=0; k<y_steps; k++){
						y = x + k*dy;
						
						for (int l=0; l<j_steps; l++) {
							j = min_j + l*dj;
							
							double point[4] = {x, y, j, j};
							likelihood += gauss.pdf(point) * dj * dy * dx;
						}
					}
				}
			}
        } else { // calculate P(X < Y < J = Z)
            if (max_j < min_z) { // P(X < Y < J = Z) = 0
                likelihood = 0.0;
            } else { // calculate P(X < Y < J = Z)
				Gauss gauss;
				gauss.set(expv, cov_m, 4);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				double x, y, j;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					unsigned int y_steps = (max_y - x) / dy;
					
					for (int k=0; k<y_steps; k++) {
						y = x + k*dy;
						unsigned int j_steps = (max_j - y) / dj;
						
						for (int l=0; l<j_steps; l++) {
							j = y + l*dj;
							double point[4] = {x, y, j, j};
							likelihood += gauss.pdf(point) * dj * dy * dx;
						}
					}
				}
			}
		}
	}
    return likelihood;
}

double prob_A21(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
	// calculate P(Y < X = J < Z)
    double likelihood = 0.0;

    if (max_y < min_x) { // P(Y < X = J < Z) = P(X = J < Z)
        if (max_x < min_j) { // P(X = J < Z) = 0
            likelihood = 0.0;
        } else { // calculate P(X = J < Z)
            if (max_j < min_z) { // P(X = J < Z) = P(X = J)
				Gauss gauss;
				double projected_expv[2] = {expv[0], expv[2]};
				double projected_cov_m[4] = {cov_m[0], cov_m[2], cov_m[8], cov_m[10]};
				gauss.set(projected_expv, projected_cov_m, 2);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				double x;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					double point[2] = {x, x};
					likelihood += gauss.pdf(point) * dx;
				}
            } else { // calculate P(X = J < Z)
				Gauss gauss;
				double projected_expv[3] = {expv[0], expv[2], expv[3]};
				double projected_cov_m[9] = {cov_m[0], cov_m[2], cov_m[3],
				                            cov_m[8], cov_m[10], cov_m[11],
											cov_m[12], cov_m[14], cov_m[15]};
				gauss.set(projected_expv, projected_cov_m, 3);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				double x, z;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					unsigned int z_steps = (max_z - x) / dz;
					
					for (int j=0; j<z_steps; j++) {
						z = x + j*dz;
						double point[3] = {x, x, z};
						likelihood += gauss.pdf(point) * dz * dx;
					}
				}
			}
		}
    } else { // calculate P(Y < X = J < Z)
        if (max_x < min_j) { // P(Y < X = J < Z) = 0
            likelihood = 0.0;
        } else { // calculate P(Y < X = J < Z)
            if (max_j < min_z) { // P(Y < X = J < Z) = P(Y < X = J)
				Gauss gauss;
				double projected_expv[3] = {expv[0], expv[1], expv[2]};
				double projected_cov_m[9] = {cov_m[0], cov_m[1], cov_m[2],
				                            cov_m[4], cov_m[5], cov_m[6],
			                                cov_m[8], cov_m[9], cov_m[10]};
				gauss.set(projected_expv, projected_cov_m, 3);
				
				unsigned int y_steps = (max_y - min_y) / dy;
				double y, x;
				
				for (int i=0; i<y_steps; i++) {
					y = min_y + i*dy;
					unsigned int x_steps = (max_x - y) / dx;
					
					for (int k=0; k<x_steps; k++) {
						x = y + k*dx;
						double point[3] = {x, y, x};
						likelihood += gauss.pdf(point) * dx * dy;
					}
				}
            } else { // calculate P(Y < X = J < Z)
				Gauss gauss;
				gauss.set(expv, cov_m, 4);
				
				unsigned int y_steps = (max_y - min_y) / dy;
				double y, x, z;
				
				for (int i=0; i<y_steps; i++) {
					y = min_y + i*dy;
					unsigned int x_steps = (max_x - y) / dx;
					
					for (int k=0; k<x_steps; k++) {
						x = y + k*dx;
						unsigned int z_steps = (max_z - x) / dz;
						
						for (int l=0; l<z_steps; l++) {
							z = x + l*dz;
							double point[4] = {x, y, x, z};
							likelihood += gauss.pdf(point) * dz * dx * dy;
						}
					}
				}
			}
		}
	}
    return likelihood;
}

double prob_A22(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
	// calculate P(Y < X < J < Z)
    double likelihood = 0.0;

    if (max_y < min_x) { // P(Y < X < J < Z) = P(X < J < Z)
        if (max_x < min_j) { // P(X < J < Z) = P(J < Z)
            if (max_j < min_z) { // P(J < Z) = 1
                likelihood = 1.0 / (dx * dy * dj * dz);  // TODO: validate approach
            } else { // calculate # P(J < Z)
				Gauss gauss;
				double projected_expv[2] = {expv[2], expv[3]};
				double projected_cov_m[4] = {cov_m[10], cov_m[11], cov_m[14], cov_m[15]};
				gauss.set(projected_expv, projected_cov_m, 2);
				
				unsigned int j_steps = (max_j - min_j) / dj;
				double j, z;
				
				for (int i=0; i<j_steps; i++) {
					j = min_j + i*dj;
					unsigned int z_steps = (max_z - j) / dz;
					
					for (int k=0; k<z_steps; k++) {
						z = j + k*dz;
						double point[2] = {j, z};
						likelihood += gauss.pdf(point) * dz * dj;
					}
				}
			}
        } else { // P(X < J < Z)
            if (max_j < min_z) { // P(X < J < Z) = P(X < J)
				Gauss gauss;
				double projected_expv[2] = {expv[0], expv[2]};
				double projected_cov_m[4] = {cov_m[0], cov_m[2], cov_m[8], cov_m[10]};
				
				unsigned int x_steps = (max_x - min_x) / dx;
				double x, j;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					unsigned int j_steps = (max_j - x) / dj;
					
					for (int k=0; k<j_steps; k++) {
						j = x + k*dj;
						double point[2] = {x, j};
						likelihood += gauss.pdf(point) * dj * dx;
					}
				}
            } else { // P(X < J < Z)
				Gauss gauss;
				double projected_expv[3] = {expv[0], expv[2], expv[3]};
				double projected_cov_m[9] = {cov_m[0], cov_m[2], cov_m[3],
				                            cov_m[8], cov_m[10], cov_m[11],
											cov_m[12], cov_m[14], cov_m[15]};
				gauss.set(projected_expv, projected_cov_m, 3);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				double x, j, z;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					unsigned int j_steps = (max_j - x) / dj;
					
					for (int k=0; k<j_steps; k++) {
						j = x + k*dj;
						unsigned int z_steps = (max_z - j) / dz;
						
						for (int l=0; l<z_steps; l++) {
							z = j + l*dz;
							double point[3] = {x, j, z};
							likelihood += gauss.pdf(point) * dz * dj * dx;
						}
					}
				}
			}
		}
    } else { // calculate P(Y < X < J < Z)
        if (max_x < min_j) { // P(Y < X < J < Z) = P(Y < X, J < Z)
            if (max_j < min_z) { // P(Y < X, J < Z) = P(Y < X)
				Gauss gauss;
				double projected_expv[2] = {expv[0], expv[1]};
				double projected_cov_m[4] = {cov_m[0], cov_m[1], cov_m[4], cov_m[5]};
				gauss.set(projected_expv, projected_cov_m, 2);
				
				unsigned int y_steps = (max_y - min_y) / dy;
				double y, x;
				
				for (int i=0; i<y_steps; i++) {
					y = min_y + i*dy;
					unsigned int x_steps = (max_x - y) / dx;
					
					for (int j=0; j<x_steps; j++) {
						x = y + j*dx;
						double point[2] = {x, y};
						likelihood += gauss.pdf(point) * dx * dy;
					}
				}
            } else { // calculate P(Y < X, J < Z)
				Gauss gauss;
				gauss.set(expv, cov_m, 4);
				
				unsigned int y_steps = (max_y - min_y) / dy;
				unsigned int j_steps = (max_j - min_j) / dj;
				double y, x, j, z;
				
				for (int i=0; i<y_steps; i++) {
					y = min_y + i*dy;
					unsigned int x_steps = (max_x - y) / dx;
					
					for (int k=0; k<x_steps; k++) {
						x = y + k*dx;
						
						for (int l=0; l<j_steps; l++) {
							j = min_j + l*dj;
							unsigned int z_steps = (max_z - j) / dz;
							
							for (int m=0; m<z_steps; m++) {
								z = j + m*dz;
								double point[4] = {x, y, j, z};
								likelihood += gauss.pdf(point) * dz * dj * dx * dy;
							}
						}
					}
				}
			}
        } else { // calculate P(Y < X < J < Z)
            if (max_j < min_z) { // P(Y < X < J < Z) = P(Y < X < J)
				Gauss gauss;
				double projected_expv[3] = {expv[0], expv[1], expv[2]};
				double projected_cov_m[9] = {cov_m[0], cov_m[1], cov_m[2],
				                            cov_m[4], cov_m[5], cov_m[6],
											cov_m[8], cov_m[9], cov_m[10]};
				
				unsigned int y_steps = (max_y - min_y) / dy;
				double y, x, j;
				
				for (int i=0; i<y_steps; i++) {
					y = min_y + i*dy;
					unsigned int x_steps = (max_x - y) / dx;
					
					for (int k=0; k<x_steps; k++) {
						x = y + k*dx;
						unsigned int j_steps = (max_j - x) / dj;
						
						for (int l=0; l<j_steps; l++) {
							j = x + l*dj;
							double point[3] = {x, y, j};
							likelihood += gauss.pdf(point) * dj * dx * dy;
						}
					}
				}
            } else { // calculate P(Y < X < J < Z)
				Gauss gauss;
				gauss.set(expv, cov_m, 4);
				
				unsigned int y_steps = (max_y - min_y) / dy;
				double y, x, j, z;
				
				for (int i=0; i<y_steps; i++) {
					y = min_y + i*dy;
					unsigned int x_steps = (max_x - y) / dx;
					
					for (int k=0; k<x_steps; k++) {
						x = y + k*dx;
						unsigned int j_steps = (max_j - x) / dj;
						
						for (int l=0; l<j_steps; l++) {
							j = x + l*dj;
							unsigned int z_steps = (max_z - j) / dz;
							
							for (int m=0; m<z_steps; m++) {
								z = j + m*dz;
								double point[4] = {x, y, j, z};
								likelihood += gauss.pdf(point) * dz * dj * dx * dy;
							}
						}
					}
				}
			}
		}
	}
    return likelihood;
}

double prob_A23(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
	// calculate P(Y < X < J = Z)
    double likelihood = 0.0;

    if (max_y < min_x) { // P(Y < X < J = Z) = P(X < J = Z)
        if (max_x < min_j) { // P(X < J = Z) = P(J = Z)
            if (max_j < min_z) { // P(J = Z) = 0
                likelihood = 0.0;
            } else { // calculate P(J = Z)
				Gauss gauss;
				double projected_expv[2] = {expv[2], expv[3]};
				double projected_cov_m[4] = {cov_m[10], cov_m[11], cov_m[14], cov_m[15]};
				gauss.set(projected_expv, projected_cov_m, 2);
				
				unsigned int j_steps = (max_j - min_j) / dj;
				double j;
				
				for (int i=0; i<j_steps; i++) {
					j = min_j + i*dj;
					double point[2] = {j, j};
					likelihood += gauss.pdf(point) * dj;
				}
			}
        } else { // calculate P(X < J = Z)
            if (max_j < min_z) { // P(X < J = Z) = 0
                likelihood = 0.0;
            } else { // calculate P(X < J = Z)
				Gauss gauss;
				double projected_expv[3] = {expv[0], expv[2], expv[3]};
				double projected_cov_m[9] = {cov_m[0], cov_m[2], cov_m[3],
				                            cov_m[8], cov_m[10], cov_m[11],
											cov_m[12], cov_m[14], cov_m[15]};
				gauss.set(projected_expv, projected_cov_m, 3);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				double x, j;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					unsigned int j_steps = (max_j - x) / dj;
					
					for (int k=0; k<j_steps; k++) {
						j = x + k*dj;
						double point[3] = {x, j, j};
						likelihood += gauss.pdf(point) * dj * dx;
					}
				}
			}
		}
    } else { // calculate P(Y < X < J = Z)
        if (max_x < min_j) { // P(Y < X < J = Z) = P(Y < X, J = Z)
            if (max_j < min_z) { // P(Y < X, J = Z) = 0
                likelihood = 0.0;
            } else { // calculate P(Y < X, J = Z)
				Gauss gauss;
				gauss.set(expv, cov_m, 4);
				
				unsigned int y_steps = (max_y - min_y) / dy;
				unsigned int j_steps = (max_j - min_j) / dj;
				double y, x, j;
				
				for (int i=0; i<y_steps; i++) {
					y = min_y + i*dy;
					unsigned int x_steps = (max_x - y) / dx;
					
					for (int k=0; k<x_steps; k++) {
						x = y + k*dx;
						
						for (int l=0; l<j_steps; l++) {
							j = min_j + l*dj;
							double point[4] = {x, y, j, j};
							likelihood += gauss.pdf(point) * dj * dx * dy;
						}
					}
				}
			}
        } else { // calculate P(Y < X < J = Z)
            if (max_j < min_z) { // P(Y < X < J = Z) = 0
                likelihood = 0.0;
            } else { // calculate P(Y < X < J = Z)
				Gauss gauss;
				gauss.set(expv, cov_m, 4);
				
				unsigned int y_steps = (max_y - min_y) / dy;
				double y, x, j, z;
				
				for (int i=0; i<y_steps; i++) {
					y = min_y + i*dy;
					unsigned int x_steps = (max_x - y) / dx;
					
					for (int k=0; k<x_steps; k++) {
						x = y + k*dx;
						unsigned int j_steps = (max_j - x) / dj;
						
						for (int l=0; l<j_steps; l++) {
							j = x + l*dj;
							double point[4] = {x, y, j, j};
							likelihood += gauss.pdf(point) * dj * dx * dy;
						}
					}
				}
			}
		}
	}
    return likelihood;
}

double prob_A31(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
	// calculate P(X = Y = J < Z)
    double likelihood = 0.0;

    if (max_x < min_y) { // P(X = Y = J < Z) = 0
        likelihood = 0.0;
    } else { // calculate P(X = Y = J < Z)
        if (max_y < min_j) { // P(X = Y = J < Z) = 0
            likelihood = 0.0;
        } else { // calculate P(X = Y = J < Z)
            if (max_j < min_z) { // P(X = Y = J < Z) = P(X = Y = J)
				Gauss gauss;
				double projected_expv[3] = {expv[0], expv[1], expv[2]};
				double projected_cov_m[9] = {cov_m[0], cov_m[1], cov_m[2],
				                            cov_m[4], cov_m[5], cov_m[6],
											cov_m[8], cov_m[9], cov_m[10]};
				gauss.set(projected_expv, projected_cov_m, 3);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				double x;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					double point[3] = {x, x, x};
					likelihood += gauss.pdf(point) * dx;
				}
            } else { // calculate P(X = Y = J < Z)
				Gauss gauss;
				gauss.set(expv, cov_m, 4);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				double x, z;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					unsigned int z_steps = (max_z - x) / dz;
					
					for (int k=0; k<z_steps; k++) {
						z = x + k*dz;
						double point[4] = {x, x, x, z};
						likelihood += gauss.pdf(point) * dz * dx;
					}
				}
			}
		}
	}
    return likelihood;	
}

double prob_A32(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
	// calculate P(X = Y < J < Z)
    double likelihood = 0.0;

    if (max_x < min_y) { // P(X = Y < J < Z) = 0
        likelihood = 0.0;
    } else { // calculate P(X = Y < J < Z)
        if (max_y < min_j) { // P(X = Y < J < Z) = P(X = Y, J < Z)
            if (max_j < min_z) { // P(X = Y, J < Z) = P(X = Y)
				Gauss gauss;
				double projected_expv[2] = {expv[0], expv[1]};
				double projected_cov_m[4] = {cov_m[0], cov_m[1], cov_m[4], cov_m[5]};
				gauss.set(projected_expv, projected_cov_m, 2);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				double x;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dj;
					double point[2] = {x, x};
					likelihood += gauss.pdf(point) * dx;
				}
            } else { // calculate P(X = Y, J < Z)
				Gauss gauss;
				gauss.set(expv, cov_m, 4);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				unsigned int j_steps = (max_j - min_j) / dj;
				double x, j, z;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					
					for (int k=0; k<j_steps; k++) {
						j = min_j + k*dj;
						unsigned int z_steps = (max_z - j) / dz;
						
						for (int l=0; l<z_steps; l++) {
							z = j + l*dz;
							double point[4] = {x, x, j, z};
							likelihood += gauss.pdf(point) * dz * dj * dx;
						}
					}
				}
			}
        } else { // calculate P(X = Y < J < Z)
            if (max_j < min_z) { // P(X = Y < J < Z) = P(X = Y < J)
				Gauss gauss;
				double projected_expv[3] = {expv[0], expv[1], expv[2]};
				double projected_cov_m[9] = {cov_m[0], cov_m[1], cov_m[2],
				                            cov_m[4], cov_m[5], cov_m[6],
											cov_m[8], cov_m[9], cov_m[10]};
				gauss.set(projected_expv, projected_cov_m, 3);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				double x, j;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					unsigned int j_steps = (max_j - x) / dj;
					
					for (int k=0; k<j_steps; k++) {
						j = x + k*dj;
						double point[3] = {x, x, j};
						likelihood += gauss.pdf(point) * dj * dx;
					}
				}
            } else { // calculate P(X = Y < J < Z)
				Gauss gauss;
				gauss.set(expv, cov_m, 4);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				double x, j, z;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					unsigned int j_steps = (max_j - x) / dj;
					
					for (int k=0; k<j_steps; k++) {
						j = x + k*dj;
						unsigned int z_steps = (max_z - j) / dz;
						
						for (int l=0; l<z_steps; l++) {
							z = j + l*dz;
							double point[4] = {x, x, j, z};
							likelihood += gauss.pdf(point) * dz * dj * dx;
						}
					}
				}
			}
		}
	}
    return likelihood;
}

double prob_A33(double* expv, double* cov_m, double min_x, double max_x, double min_y, double max_y, double min_j, double max_j, double min_z, double max_z, double dx, double dy, double dj, double dz){
	// calculate P(X = Y < J = Z)
    double likelihood = 0.0;

    if (max_x < min_y) { // P(X = Y < J = Z) = 0
        likelihood = 0.0;
    } else { // calculate P(X = Y < J = Z)
        if (max_y < min_j) { // P(X = Y < J = Z) = P(X = Y, J = Z)
            if (max_j < min_z) { // P(X = Y, J = Z) = 0
                likelihood = 0.0;
            } else { // calculate P(X = Y, J = Z)
				Gauss gauss;
				gauss.set(expv, cov_m, 4);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				unsigned int j_steps = (max_j - min_j) / dj;
				double x, j;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					
					for (int k=0; k<j_steps; k++) {
						j = min_j + k*dj;
						double point[4] = {x, x, j, j};
						likelihood += gauss.pdf(point) * dj * dx;
					}
				}
			}
        } else { // calculate P(X = Y < J = Z)
            if (max_j < min_z) { // P(X = Y < J = Z) = 0
                likelihood = 0.0;
            } else { // calculate P(X = Y < J = Z)
				Gauss gauss;
				gauss.set(expv, cov_m, 4);
				
				unsigned int x_steps = (max_x - min_x) / dx;
				double x, j;
				
				for (int i=0; i<x_steps; i++) {
					x = min_x + i*dx;
					unsigned int j_steps = (max_j - x) / dj;
					
					for (int k=0; k<j_steps; k++) {
						j = x + k*dj;
						double point[4] = {x, x, j, j};
						likelihood += gauss.pdf(point) * dj * dx;
					}
				}
			}
		}
	}
    return  likelihood;
}

double* get_probabilities(double** dims, unsigned int n, double eff_sample_size){
	double* expv = new double[4];
	
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
	
	double* cov_m = cov_matr(dims, n);

	for (int i=0; i<16; i++) cov_m[i] *= eff_sample_size / n;
	
	double c = 3.0; // how many stdevs from the mean should be included in the computations
    double bins = 20.0;
	
	double min_x = expv[0] - c * sqrt(cov_m[0]);
	double max_x = expv[0] + c * sqrt(cov_m[0]);
    double dx = (max_x - min_x) / bins;

    double min_y = expv[1] - c * sqrt(cov_m[5]);
	double max_y = expv[1] + c * sqrt(cov_m[5]);
    double dy = (max_y - min_y) / bins;
	
	double min_j = expv[2] - c * sqrt(cov_m[10]);
	double max_j = expv[2] + c * sqrt(cov_m[10]);
    double dj = (max_j - min_j) / bins;
	
	double min_z = expv[3] - c * sqrt(cov_m[15]);
	double max_z = expv[3] + c * sqrt(cov_m[15]);
    double dz = (max_z - min_z) / bins;
	
	check_feasibility(min_x, min_y, min_j, max_j, max_z);
	
	double* likelihoods = new double[9]{0, 0, 0, 0, 0, 0, 0, 0, 0};
	
	cout << "X: epxv " << expv[0] << ", var " << cov_m[0] << ", dx " << dx << endl;
	cout << "Y: epxv " << expv[1] << ", var " << cov_m[5] << ", dy " << dy << endl;
	cout << "J: epxv " << expv[2] << ", var " << cov_m[10] << ", dj " << dj << endl;
	cout << "Z: epxv " << expv[3] << ", var " << cov_m[15] << ", dz " << dz << endl;
	
	cout << "calculating Aij likelihood" << endl;
	
	#pragma omp parallel sections
	{
		#pragma omp section
		likelihoods[0] = prob_A11(expv, cov_m, min_x, max_x, min_y, max_y, min_j, max_j, min_z, max_z, dx, dy, dj, dz);
		
		/*
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
		*/
	}
	cout << "Finished calculating Aij likelihood" << endl;
	
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
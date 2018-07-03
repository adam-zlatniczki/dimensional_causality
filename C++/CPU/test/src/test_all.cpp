#include <iostream>
#include <vector>
#include <cmath>
#include "embedding.h"
#include "dimensions.h"
#include "trimming.h"
#include "statistics.h"
#include "probabilities.h"
#include "causality.h"

using namespace std;


bool doubles_equal(double a, double b, double epsilon = 0.000001)
{
    return fabs(a - b) < epsilon;
}

int test_embedding(){
	double x[10] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
	double* X_test = embed(x, 10, 3, 2);
	double X_expected[18] =   {5.0, 3.0, 1.0,
							  6.0, 4.0, 2.0,
							  7.0, 5.0, 3.0,
							  8.0, 6.0, 4.0,
							  9.0, 7.0, 5.0,
							  10.0, 8.0, 6.0};
	bool match = true;
	for (int i = 0; i < 18; i++) {
		if (X_test[i] != X_expected[i]) {
			match = false;
			//break;
		}
	}

	delete[] X_test;

	if(match){
		return 0;
	}else{
		return 1;
	}
}

int test_diophantine_sum(){
	double x[10] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
	double y[10] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };

	const double a = sqrt(2.0) - 1;
	
	double* J_test = diophantine_sum(x, y, 10, 3, 2);
	double J_expected[18] = {
		5.0*a, 3.0*a, 1.0*a,
		6.0*a, 4.0*a, 2.0*a,
		7.0*a, 5.0*a, 3.0*a,
		8.0*a, 6.0*a, 4.0*a,
		9.0*a, 7.0*a, 5.0*a,
		10.0*a, 8.0*a, 6.0*a
	};
	
	bool match = true;
	for (int i = 0; i < 18; i++) {
		if (!doubles_equal(J_test[i], J_expected[i])) {
			match = false;
			break;
		}
	}
	
	delete[] J_test;
	
	if(match){
		return 0;
	}else{
		return 1;
	}
}

int test_shuffled_diophantine_sum(){
	srand(0);
	
	double x[10] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
	double y[10] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
	
	const double a = sqrt(2.0);
	
	double* Z_test = shuffled_diophantine_sum(x, y, 10, 3, 2);
	double Z_expected[18] = {
		a*4.0 - 2.0, a*9.0 - 5.0, a*10.0 - 4.0,
		a*8.0 - 10.0, a*2.0 - 3.0, a*1.0 - 8.0,
		a*5.0 - 7.0, a*4.0 - 2.0, a*9.0 - 5.0,
		a*6.0 - 1.0, a*8.0 - 10.0, a*2.0 - 3.0,
		a*7.0 - 9.0, a*5.0 - 7.0, a*4.0 - 2.0,
		a*3.0 - 6.0, a*6.0 - 1.0, a*8.0 - 10.0
	};

	bool match = true;
	for (int i = 0; i < 18; i++) {
		if (!doubles_equal(Z_test[i], Z_expected[i])) {
			match = false;
			break;
		}
	}
	
	delete[] Z_test;
	
	if(match){
		return 0;
	}else{
		return 1;
	}
}

int test_knn_dist(){
	double* matr = new double[15]{2.0, 1.0, 5.0, 1.0, 5.0, 3.0, 3.0, 4.0, 4.0, 5.0, 2.0, 2.0, 4.0, 3.0, 1.0};
	
	double** distances_test = knn_distances(matr, 5, 3, 4);
	double distances_expected[5][5] = {
		{ 3.31662479,  4.35889894,  4.58257569,  4.89897949},
		{ 2.44948974,  4.12310563,  4.58257569,  5.09901951},
		{ 2.44948974,  3.31662479,  3.31662479,  3.46410162},
		{ 1.73205081,  3.46410162,  4.35889894,  5.09901951},
		{ 1.73205081,  3.31662479,  4.12310563,  4.89897949}
	};
	
	bool match = true;
	
	for (int i=0; i<5; i++) {
		for (int j=0; j<4; j++) {
			if (!doubles_equal(distances_test[i][j], distances_expected[i][j])) {
				match = false;
				break;
			}
		}
	}
	
	for (int i=0; i<5; i++) {
		delete[] distances_test[i];
	}
	delete[] distances_test;
	delete[] matr;
	
	if (match) {
		return 0;
	} else {
		return 1;
	}
}


int test_local_dims(){
	double** dists = new double*[5];
	dists[0] = new double[4]{3.31662479,  4.35889894,  4.58257569,  4.89897949};
	dists[1] = new double[4]{2.44948974,  4.12310563,  4.58257569,  5.09901951};
	dists[2] = new double[4]{2.44948974,  3.31662479,  3.31662479,  3.46410162};
	dists[3] = new double[4]{1.73205081,  3.46410162,  4.35889894,  5.09901951};
	dists[4] = new double[4]{1.73205081,  3.31662479,  4.12310563,  4.89897949};

	double* dims_test = local_dims(dists, 5, 4);
	double dims_expected[5] = {5.934102, 3.262765, 15.932334, 1.792954, 1.776939};
	
	bool match = true;
	for (int i=0; i<5; i++){
		if (!doubles_equal(dims_test[i], dims_expected[i], 0.0001)){
			match = false;
			break;
		}
	}
	
	delete[] dists;
	delete[] dims_test;

	if (match) {
		return 0;
	}else{
		return 1;
	}
}

int test_single_trim_mask(){
	double local_dims[5] = {2.0, 1.0, 3.0, 5.0, 4.0};
	bool* mask_test = single_trim_mask(local_dims, 5);
	bool mask_expected[] = {true, false, true, false, true};
	
	bool match = true;
	
	for (int i=0; i<5; i++){
		if ( mask_test[i] != mask_expected[i]){
			match = false;
			break;
		}
	}
	
	delete[] mask_test;
	
	if(match){
		return 0;
	}else{
		return 1;
	}
}

int test_merge_masks(){
	bool x_mask[5] = {false, true, true, true, true};
	bool y_mask[5] = {true, false, true, true, true};
	bool j_mask[5] = {true, true, false, true, true};
	bool z_mask[5] = {true, true, true, false, true};
	
	bool* mask_test = merge_masks(x_mask, y_mask, j_mask, z_mask, 5);
	bool mask_expected[5] = {false, false, false, false, true};
	
	bool match = true;
	for (int i=0; i<5; i++) {
		if (mask_test[i] != mask_expected[i]) {
			match = false;
			break;
		}
	}
	
	delete[] mask_test;
	
	if (match) {
		return 0;
	}else{
		return 1;
	}
}

int test_joint_trim_mask(){
	double x_dims[5] = {2.0, 1.0, 3.0, 5.0, 4.0};
	double y_dims[5] = {2.0, 1.0, 3.0, 4.0, 5.0};
	double j_dims[5] = {2.0, 1.0, 3.0, 5.0, 4.0};
	double z_dims[5] = {2.0, 1.0, 3.0, 5.0, 4.0};
	
	bool* mask_test = joint_trim_mask(x_dims, y_dims, j_dims, z_dims, 5, 0.05);
	bool mask_expected[5] = {true, false, true, false, false};
	
	bool match = true;
	for (int i=0; i<5; i++) {
		if (mask_test[i] != mask_expected[i]) {
			match = false;
			break;
		}
	}
	
	delete[] mask_test;
	
	if (match) {
		return 0;
	}else{
		return 1;
	}
}

int test_trim_data(){
	double x_dims[5] = {2.0, 1.0, 3.0, 5.0, 4.0};
	double y_dims[5] = {2.0, 1.0, 3.0, 4.0, 5.0};
	double j_dims[5] = {2.0, 1.0, 3.0, 5.0, 4.0};
	double z_dims[5] = {2.0, 1.0, 3.0, 5.0, 4.0};
	
	unsigned int trimmed_size;
	double** trimmed_data_test = trim_data(x_dims, y_dims, j_dims, z_dims, 5, trimmed_size, 0.05);
	
	double trimmed_data_expected[4][2] = { {2.0, 3.0}, {2.0, 3.0}, {2.0, 3.0}, {2.0, 3.0}};
	
	bool match = true;
	
	for (int i=0; i<4; i++) {
		for (int j=0; j<2; j++) {
			if( trimmed_data_test[i][j] != trimmed_data_expected[i][j] ) {
				match = false;
				break;
			}
		}
	}
	if (trimmed_size != 2) match = false;
	
	for (int i=0; i<4; i++) delete[] trimmed_data_test[i];
	delete[] trimmed_data_test;
	
	if (match) {
		return 0;
	} else {
		return 1;
	}
}

int test_exp_val(){
	double x[10] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
	double exp_val_test = exp_val(x, 10);
	
	if (doubles_equal(exp_val_test, 5.5)) {
		return 0;
	} else {
		return 1;
	}
}

int test_cov_m(){
	double** data = new double*[4];
	data[0] = new double[10]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
	data[1] = new double[10]{-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0};
	data[2] = new double[10]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
	data[3] = new double[10]{-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0};
	
	double* cov_m_test = cov_matr(data, 10);
	
	double cov_m_expected[16] = {
		8.25, -8.25, 8.25, -8.25, -8.25, 8.25, -8.25, 8.25, 8.25, -8.25, 8.25, -8.25, -8.25, 8.25, -8.25, 8.25
	};
	
	bool match = true;
	
	for (int i=0; i<16; i++) {
		if (!doubles_equal(cov_m_expected[i], cov_m_test[i])) {
			match = false;
			break;
		}
	}
	
	delete[] data[0];
	delete[] data[1];
	delete[] data[2];
	delete[] data[3];
	delete[] data;
	
	if (match) {
		return 0;
	} else {
		return 1;
	}
}

int test_inv_cov_m_4x4(){
	double cov_m[16] = {
		1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 2.0
	};
	
	double det = 0.0;
	
	double* inv_cov_m_test = inv_cov_m_4x4(cov_m, det);
	
	double inv_cov_m_expected[16] = {
		1.0 ,0.0, 0.0, 0.0, 0.0, 3.0, -2.0, 0.0, 0.0, -1.0, 1.0, 0.000, 0.0, 0.0, 0.0, 0.5
	};
	
	bool match = true;
	
	for (int i=0; i<16; i++) {
		if (!doubles_equal(inv_cov_m_test[i], inv_cov_m_expected[i])) {
			match = false;
			break;
		}
	}
	if (det != 0.5) {
		match = false;
	}
	
	if (match) {
		return 0;
	} else {
		return 1;
	}
}

int test_inv_cov_m_3x3(){
	double cov_m[9] = {
		1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 1.0
	};
	
	double det = 0.0;
	
	double* inv_cov_m_test = inv_cov_m_3x3(cov_m, det);
	
	double inv_cov_m_expected[9] = {
		0.5 , -0.5 ,  0.5 , 0.25,  0.25, -0.25, -0.5 ,  0.5 ,  0.5
	};
	
	bool match = true;
	
	for (int i=0; i<9; i++) {
		if (!doubles_equal(inv_cov_m_test[i], inv_cov_m_expected[i])) {
			match = false;
			break;
		}
	}
	if (det != 0.25) {
		match = false;
	}
	
	if (match) {
		return 0;
	} else {
		return 1;
	}
}

int test_inv_cov_m_2x2(){
	double cov_m[4] = {
		1.0, 2.0, 1.0, 4.0
	};
	
	double det = 0.0;
	
	double* inv_cov_m_test = inv_cov_m_2x2(cov_m, det);
	
	double inv_cov_m_expected[4] = {
		2.0, -1.0, -0.5, 0.5
	};
	
	bool match = true;
	
	for (int i=0; i<4; i++) {
		if (!doubles_equal(inv_cov_m_test[i], inv_cov_m_expected[i])) {
			match = false;
			break;
		}
	}
	if (det != 0.5) {
		match = false;
	}
	
	if (match) {
		return 0;
	} else {
		return 1;
	}
}


int test_gauss_set(){
	double expv[2] = {1.0, 2.0};
	double cov_m[4] = {1.0, 2.0, 1.0, 4.0};
	
	Gauss gauss;
	gauss.set(expv, cov_m, 2);
	
	bool match = true;
	
	if (gauss.get_expv()[0] != 1.0) match = false;
	if (gauss.get_expv()[1] != 2.0) match = false;
	
	if (gauss.get_inv_cov_m()[0] != 2.0) match = false;
	if (gauss.get_inv_cov_m()[1] != -1.0) match = false;
	if (gauss.get_inv_cov_m()[2] != -0.5) match = false;
	if (gauss.get_inv_cov_m()[3] != 0.5) match = false;
	
	if (gauss.get_dim() != 2) match = false;
	
	if (gauss.get_det() != 2.0) match = false;
	
	if (match) {
		return 0;
	} else {
		return 1;
	}
}


int test_gauss_pdf(){
	double expv[4] = {1.0, 2.0, 3.0, 4.0};
	double cov_m[16] = {	1.0, 0.0, 0.0, 0.0,
						0.0, 5.0, 8.0, 0.0,
						0.0, 8.0, 13.0, 0.0,
						0.0, 0.0, 0.0, 4.0};

	Gauss gauss;
	gauss.set(expv, cov_m, 4);
	
	double point[4] = {0.5, 1.5, 2.5, 3.5};
	
	double l_test = gauss.pdf(point);
	
	if (doubles_equal(l_test, 0.00843680)) {
		return 0;
	} else {
		return 1;
	}
}


int test_infer_causality(){
	double* x = new double[100];
	double* y = new double[100];
	
	srand(0);
	for (int i=0; i<100; i++) {
		x[i] = rand();
		y[i] = rand();
	}
	
	unsigned int* k_range = new unsigned int[3]{4, 6, 8};
	
	cout << "Starting inference" << endl;
	double* probs = infer_causality(x, y, 100, 3, 1, k_range, 2, 0.05);
	cout << "Finished inference" << endl;
	
	cout << "P(X->Y) = " << probs[0] << endl;
	cout << "P(X<->Y) = " << probs[1] << endl;
	cout << "P(Y->X) = " << probs[2] << endl;
	cout << "P(X cc Y) = " << probs[3] << endl;
	cout << "P(X | Y) = " << probs[4] << endl;
	
	delete[] probs;
	delete[] k_range;
	
	return 0;
}




int main(){
	cout << "Running unit tests..." << endl;
	cout << "=====================" << endl;

	cout << "Test time-delay embedding failed: " << test_embedding() << endl;
	cout << "Test Diophantine sum failed: " << test_diophantine_sum() << endl;
	cout << "Test shuffled Diophantine sum failed: " << test_shuffled_diophantine_sum() << endl;
	
	cout << "Test kNN distances (ALGLIB) failed: " << test_knn_dist() << endl;
	
	cout << "Test local dimension estimation failed: " << test_local_dims() << endl;
	
	cout << "Test mask generation for trimming failed: " << test_single_trim_mask() << endl;
	cout << "Test merge trimming masks failed: " << test_merge_masks() << endl;
	cout << "Test joint trimming mask calculation failed: " << test_joint_trim_mask() << endl;
	cout << "Test trim data failed: " << test_trim_data() << endl;
	
	cout << "Test expected value failed: " << test_exp_val() << endl;
	cout << "Test covariance matrix failed: " << test_cov_m() << endl;
	cout << "Test 4x4 covariance matrix inversion failed: " << test_inv_cov_m_4x4() << endl;
	cout << "Test 3x3 covariance matrix inversion failed: " << test_inv_cov_m_3x3() << endl;
	cout << "Test 2x2 covariance matrix inversion failed: " << test_inv_cov_m_2x2() << endl;
	
	cout << "Test Gauss set failed: " << test_gauss_set() << endl;
	cout << "Test Gauss pdf failed: " << test_gauss_pdf() << endl;
	
	cout << "Smoke test inferring causality " << test_infer_causality() << endl;
	
	
	
	
	cout.flush();
	return 0;
}
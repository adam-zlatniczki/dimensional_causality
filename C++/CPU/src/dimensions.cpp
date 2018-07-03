#include <cmath>
#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "alglibmisc.h"

using namespace alglib;
using namespace std;


double** knn_distances(const double* X, unsigned int n, unsigned int d, unsigned int k){
	real_2d_array a;
	a.setcontent(n, d, X);
	bool self_match = false;

	kdtree kdt;
    kdtreebuild(a, d, 0, 2, kdt);
	
	double** distances = new double*[n];
	
	#pragma omp parallel for
	for (int i=0; i<n; i++) {
		distances[i] = new double[k];
		
		real_1d_array x;
		real_1d_array r;
	
		x.setcontent(d, &X[i*d]);
		
		kdtreerequestbuffer buf;
		kdtreecreaterequestbuffer(kdt, buf);
		kdtreetsqueryknn(kdt, buf, x, k, self_match);
		kdtreetsqueryresultsdistances(kdt, buf, r);
		
		for (int j=0; j<k; j++) {
			distances[i][j] = r(j);
		}
	}
	
	return distances;
}


#pragma omp declare simd
double* local_dims(double** dist, const unsigned int n, unsigned int k){
	double* dims = new double[n];
	const double log_2 = log(2.0);
	
	#pragma omp parallel for simd
	for (int i=0; i<n; i++){
		dims[i] = log_2 / log(dist[i][k-1] / dist[i][k/2-1]);
	}
	
	return dims;
}
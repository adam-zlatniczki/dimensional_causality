all:
	g++ -g -O3 -fPIC -fopenmp -std=c++11 -I../../../C++/OpenMP/include -I../../../C++/OpenMP/lib/alglib -c ../../../C++/OpenMP/lib/alglib/alglibinternal.cpp ../../../C++/OpenMP/lib/alglib/alglibmisc.cpp ../../../C++/OpenMP/lib/alglib/ap.cpp ../../../C++/OpenMP/src/causality.cpp ../../../C++/OpenMP/src/dimensions.cpp ../../../C++/OpenMP/src/embedding.cpp ../../../C++/OpenMP/src/probabilities.cpp ../../../C++/OpenMP/src/statistics.cpp ../../../C++/OpenMP/src/trimming.cpp
	g++ -shared -fopenmp -o dimensional_causality_openmp.so alglibinternal.o alglibmisc.o ap.o causality.o dimensions.o embedding.o probabilities.o statistics.o trimming.o
	rm -rf *.o

clean:
	rm -f dimensional_causality_openmp.so
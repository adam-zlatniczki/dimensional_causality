ifeq ($(OS),Windows_NT)
	RM = del /Q
	EXT = dll
else
	RM = rm -f
	EXT = so
endif

all:
	g++ -g -O3 -fPIC -fopenmp -std=c++11 -Iinclude -Ilib/alglib -c lib/alglib/alglibinternal.cpp lib/alglib/alglibmisc.cpp lib/alglib/ap.cpp src/causality.cpp src/dimensions.cpp src/embedding.cpp src/probabilities.cpp src/statistics.cpp src/trimming.cpp
	g++ -shared -fopenmp -o bin/dimensional_causality_openmp.$(EXT) alglibinternal.o alglibmisc.o ap.o causality.o dimensions.o embedding.o probabilities.o statistics.o trimming.o
	$(RM) *.o

test_all:
	g++ -g -O3 -fPIC -fopenmp -std=c++11 -Iinclude -Ilib/alglib -c test/src/test_all.cpp lib/alglib/alglibinternal.cpp lib/alglib/alglibmisc.cpp lib/alglib/ap.cpp src/causality.cpp src/dimensions.cpp src/embedding.cpp src/probabilities.cpp src/statistics.cpp src/trimming.cpp
	g++ -fopenmp -o bin/test_all alglibinternal.o alglibmisc.o ap.o causality.o dimensions.o embedding.o probabilities.o statistics.o trimming.o test_all.o
	$(RM) *.o

clean:
	$(RM) *.o
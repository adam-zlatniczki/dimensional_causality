g++ -Iinclude -c src\causality.cpp -o build\dimensional_causality.o -O3 -fopenmp
g++ -shared -o bin\dimensional_causality.dll build\dimensional_causality.o -Wl,--out-implib,libdimensional_causality.a
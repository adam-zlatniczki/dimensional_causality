g++ -Iinclude -Ilib\alglib -o build\test_parallel -O3 -fopenmp test\src\test_all.cpp src\* lib\alglib\*
build\test_parallel.exe
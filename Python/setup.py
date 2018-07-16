#!/usr/bin/env python

import os
from setuptools.command.install import install
from wheel.bdist_wheel import bdist_wheel
from setuptools.command.test import test
import glob
from distutils.core import setup

extension = ''
if os.name == 'posix':
    extension = '.so'
else:
    extension = '.dll'


def compile_CPU_library():
    ''' Compile the CPU-only version of the source. '''
    print "Compiling and linking shared library..."
    os.system(
        'g++ -g -O3 -fPIC -fopenmp -std=c++11 -I../C++/CPU/include -I../C++/CPU/lib/alglib -c ../C++/CPU/lib/alglib/alglibinternal.cpp ../C++/CPU/lib/alglib/alglibmisc.cpp ../C++/CPU/lib/alglib/ap.cpp ../C++/CPU/src/causality.cpp ../C++/CPU/src/dimensions.cpp ../C++/CPU/src/embedding.cpp ../C++/CPU/src/probabilities.cpp ../C++/CPU/src/statistics.cpp ../C++/CPU/src/trimming.cpp')
    os.system(
        'g++ -shared -fopenmp -o dimensional_causality/dimensional_causality_cpu' + extension + ' alglibinternal.o alglibmisc.o ap.o causality.o dimensions.o embedding.o probabilities.o statistics.o trimming.o')
    tmp_files = glob.glob('*.o')
    for file in tmp_files:
        os.remove(file)


class CustomInstall(install):
    ''' Customized setuptools install command that compiles the C++ source into a shared library. '''

    def run(self):
        compile_CPU_library()
        install.run(self)


class CustomWheel(bdist_wheel):
    ''' Customized setuptools install command that compiles the C++ source into a shared library. '''

    def run(self):
        compile_CPU_library()
        bdist_wheel.run(self)


class CustomTest(test):
    ''' Customized setuptools install command that compiles the C++ source into a shared library. '''

    def run(self):
        compile_CPU_library()
        test.run(self)

setup(name='Dimensional Causality',
      version='1.0',
      description='Python version of the Dimensional Causality method',
      long_description='Python version of the Dimensional Causality method developed in Benko, Zlatniczki, Fabo, Solyom, Eross, Telcs & Somogyvari (2018) - Inference of causal relations via dimensions.',
      author='Adam Zlatniczki',
      author_email='adam.zlatniczki@cs.bme.hu',
      url='https://github.com/adam-zlatniczki/dimensional_causality',
      cmdclass={
          'install': CustomInstall,
          'bdist_wheel': CustomWheel,
          'test': CustomTest
      },
      packages=['dimensional_causality'],
      package_dir={'dimensional_causality': 'dimensional_causality'},
      package_data={'dimensional_causality': ['dimensional_causality_cpu' + extension]},
      test_suite='nose.collector',
      tests_require=['nose'],
      classifiers=['TODO'],
      license='TODO'
      )

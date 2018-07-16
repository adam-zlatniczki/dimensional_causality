# dimensional_causality

This project contains the implementation of the Dimensional Causality method proposed in Benko, Zlatniczki, Fabo, Solyom, Eross, Telcs & Somogyvari (2018) - Inference of causal relations via dimensions.
The method is available in C++, Python, R and MatLab. See the installation steps below.


1 - Installation  
----------------

	1.1 - C++  
		1.1.1 - Prerequisites
			- Windows
				- Install mingw
				- add its bin directory to your system path
			- Unix:
				- Run
					apt-get install g++
					apt-get install make
		1.1.2 - Installation
			- move to C++/CPU
			- On Windows, run
				mingw32-make
			- On Unix, run
				make
			- the built dll/so can be found in the bin directory

	1.2 - Python
		1.2.1 - Prerequisites
			- Windows
				- Install mingw
				- add its bin directory to your system path
			- Unix:
				- Run
					apt-get install g++
		1.2.2 - Installation
			- move to the Python directory
			- run the command
				pip install .

	1.3 - R
		1.3.1 - Prerequisites
			- Windows:
				- Install rtools (https://cran.r-project.org/bin/windows/Rtools/)
				- Make sure that Rtools\bin and Rtools\mingw_32\bin are added to your system path (or you can simply set this during the Rtools install)
			- Unix:
				- Run
					apt-get install g++
					apt-get install make
		1.3.2 - Installation
			- move to the R directory
			- run the command
				R CMD INSTALL dimensionalcausality

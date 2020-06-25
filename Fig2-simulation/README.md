##DR simulation Code

AA-ADMM code are not included here.
### Compiling

The Code has been tested on the following. Systems and compilers:

* Ubuntu 18.04 using gcc 7.5.0. And it needs MKL, or it may slow down the performance.
* macOS Catalina 10.15.5 with Qt 5.10.1 and Homebrew GCC 9.3.0

Follow the following steps to compile the code:

* Create a folder `build` within the root directories of the code

* Run cmake to generate the build files inside the build folder, and compile the source code:
        * On linux, run the following commands within the build folder:

    	```
    	$ cmake -DCMAKE_BUILD_TYPE=Release ..
        $ make
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-8 -DCMAKE_CXX_COMPILER=g++-8 ..
        ```
        * On macOS, the default compiler on macOS (Apple Clang from Xcode) does not support OpenMP. To achieve best performance, it is recommended to compile the code with Homebrew GCC. For example, if you create a build folder in the source code directory, then use the following command (assuming gcc-9 and g++-9 are installed by Homebrew)run the following commands within the build folder:

        ```
        $ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-9 -DCMAKE_CXX_COMPILER=g++-9 ..
        $ make
        ```

* Afterwards, there should be a folder `SGP2020` generated under `DRsimulationCode/build/samples/`, and there are few executable files in it.

	* Create new result folder, output message will be generated here:

		```
       $ mkdir ./samples/SGP2020/result
       $ cd ./samples/SGP2020
       ```

### Usage
* Some examples:

	```
        $ ./testDRSoft
        $ ./beams -it 10 -a 0
        $ ./beams -it 10 -am 3 -c 1
        $ ./beams -it 10 -am 3 -c 0
        $ ./beams_rubber -it 100 -am 3 -c 1
	```

        * -am 3: use anderson acceleration and set window size m=3
        * -a 0: ADMM
	* -c 0: use primal residual as merit function
	* -c 1: use DR envelope as merit function
        * -c 0: use combined residual as merit function
        * -it 10: set iteration number as 10, if using rubber, please set it at least 100, or it may not converged.
        * testDRSoft: a small demo. It can be used together with `testParam.txt`. Press 'space' to simulate a process.

* Simulation demo instructions:
	* Press 'p' to simulate one frame.
	* Press 'space' to start demo.

* A script for test has been included. Here are some instructions:
	* Put both `testDRSoft` and `testParam.txt` into the folder `SGP2020`.
	* Run 
	
		```
		$ cd ./samples/SGP2020/
		$ ./testDRSoft
		```
   * Press 'p' and close the window after it finishs simiulation.
   * Repeat the last step.
   * You can find results in the subfolder `result`.

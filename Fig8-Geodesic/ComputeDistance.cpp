// BSD 3-Clause License
//
// Copyright (c) 2019, Jiong Tao, Bailin Deng, Yue Peng
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include "GeodesicDistanceSolver.h"
#include "DistanceFile.h"
#include "GetRSS.h"
#include <iostream>


int main(int argc, char* argv[])
{
    // Modified features by
  if(argc < 4)
  {
    std::cerr << "Usage: GeodDistSolver PARAMETERS_FILE MESH_FILE DISTANCE_FILE" << std::endl;
    return 1;
  }

  Parameters param;
    // Modified features by
  if (argc == 4) {
	  if (!param.load(argv[1])) {
		  std::cerr << "Error: unable to load parameter file" << std::endl;
		  return 1;
	  }
  }
  else if (argc == 5) {
	  if (!param.load(argv[1], atof(argv[4]))) {
		  std::cerr << "Error: unable to load parameter file" << std::endl;
		  return 1;
	  }
  }
  else if (argc == 6) {
      if (!param.load(argv[1], atof(argv[4]), atoi(argv[5]))) {
          std::cerr << "Error: unable to load parameter file" << std::endl;
          return 1;
      }
      omp_set_num_threads(param.number_of_threads);
  }
  param.output_options();

	GeodesicDistanceSolver solver;
	if(!solver.solve(argv[2],param)){
		std::cerr << "Error in solving geodesic distance" << std::endl;
		return 1;
	}

	if(!DistanceFile::save(argv[3], solver.get_distance_values())){
		std::cerr << "Error in saving distance file" << std::endl;
		return 1;
	}

	size_t peak_mem = getPeakRSS();
	std::cout << "Peak memory usage in bytes: " << peak_mem << std::endl;

    // Modified features by
#ifdef CERR_TIME_RAM
	std::cerr << peak_mem << std::endl;
#endif
	return 0;
}






#ifndef MCL_MINIMIZER_H
#define MCL_MINIMIZER_H

#include "Problem.hpp"

namespace mcl {
namespace optlib {

template<typename Scalar, int DIM>
class Minimizer {
public:
    typedef Eigen::Matrix<Scalar,DIM,1> VectorX;
	static const int FAILURE = -1; // returned by minimize if an error is encountered

	virtual void set_max_iters( int iters ) = 0;
	virtual void set_verbose( int v ) = 0;
	virtual int minimize(Problem<Scalar,DIM> &problem, VectorX &x) = 0;
};

} // ns optlib
} // ns mcl

#endif


#ifndef MCL_NEWTON_H
#define MCL_NEWTON_H

#include "Armijo.hpp"
#include "MoreThuente.hpp"
#include "Minimizer.hpp"

namespace mcl {
namespace optlib {

template<typename Scalar, int DIM>
class Newton : public Minimizer<Scalar,DIM> {
private:
	typedef Eigen::Matrix<Scalar,DIM,1> VectorX;
	typedef Eigen::Matrix<Scalar,DIM,DIM> MatrixX;

public:
	int max_iters;
	Newton() : max_iters(20) {}
	void set_max_iters( int iters ){ max_iters = iters; }
	void set_verbose( int v ){ (void)(v); } // TODO

	int minimize(Problem<Scalar,DIM> &problem, VectorX &x){

		VectorX grad, delta_x, x_last;
		if( DIM  == Eigen::Dynamic ){
			int dim = x.rows();
			x_last.resize(dim);
			grad.resize(dim);
			delta_x.resize(dim);
		}

		int iter = 0;
		for( ; iter < max_iters; ++iter ){

			problem.gradient(x,grad);
			problem.solve_hessian(x,grad,delta_x);
			Scalar rate = Armijo<Scalar, DIM, decltype(problem)>::linesearch(x, delta_x, problem, 1);

			if( rate <= 0 ){
				printf("Newton::minimize: Failure in linesearch\n");
				return Minimizer<Scalar,DIM>::FAILURE;
			}

			x_last = x;
			x += rate * delta_x;
			if( problem.converged(x_last,x,grad) ){ break; }
		}

		return iter;
	}

};

} // ns optlib
} // ns mcl

#endif

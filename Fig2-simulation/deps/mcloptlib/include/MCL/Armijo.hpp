
#ifndef MCL_ARMIJO_H
#define MCL_ARMIJO_H

#include "Problem.hpp"

namespace mcl {
namespace optlib {

// Backtracking-Armijo
template<typename Scalar, int DIM, typename P>
class Armijo {
public:
	typedef Eigen::Matrix<Scalar,DIM,1> VectorX;
	typedef Eigen::Matrix<Scalar,DIM,DIM> MatrixX;

	static Scalar linesearch(const VectorX &x, const VectorX &p, P &problem, Scalar alpha_init) {

		const Scalar tau = 0.7;
		const Scalar beta = 0.3;
		const int max_iter = 1000000;
		Scalar alpha = std::abs(alpha_init);
		VectorX grad;
		if( DIM == Eigen::Dynamic ){ grad = VectorX::Zero(x.rows()); }
		Scalar f_x = problem.gradient(x, grad);
		Scalar bgdp = beta*grad.dot(p);

		int iter = 0;
		for( iter=0; iter < max_iter; ++iter ){
			Scalar f_xap = problem.value(x + alpha*p);
			Scalar f_x_a = f_x + alpha*bgdp; // Armijo condition
			if( f_xap <= f_x_a ){ break; }
			alpha *= tau;
		}

		if( iter == max_iter ){
			printf("Armijo::linesearch Error: Reached max_iters\n");
			return -1;
		}

		return alpha;
	}
};

}
}

#endif

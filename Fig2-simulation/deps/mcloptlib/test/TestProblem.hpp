
#include "MCL/Problem.hpp"

// min |Ax-b|
class DynProblem : public mcl::optlib::Problem<double,Eigen::Dynamic> {
public:
	typedef Eigen::Matrix<double,Eigen::Dynamic,1> VectorX;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixX;

	MatrixX A;
	VectorX b;
	DynProblem( int dim_ ){

		// Test on random SPD
		A = MatrixX::Random(dim_,dim_);
		A = A.transpose() * A;
		A = A + MatrixX::Identity(dim_,dim_);

		b = VectorX::Random(dim_);
	}

	int dim() const { return b.rows(); }
	bool converged(const VectorX &x0, const VectorX &x1, const VectorX &grad){

		// Check sizes of input
		int m_dim = dim();
		if( x0.rows() != m_dim ){
			throw std::runtime_error("Error in Problem::converged: x0 wrong dimension");
		}
		if( x1.rows() != m_dim ){
			throw std::runtime_error("Error in Problem::converged: x1 wrong dimension");
		}
		if( grad.rows() != m_dim ){
			throw std::runtime_error("Error in Problem::converged: gradient wrong dimension");
		}

		return grad.norm() < 1e-10 || (x0-x1).norm() < 1e-10;
	}

	double value(const VectorX &x){

		// Check sizes of input
		int m_dim = dim();
		if( x.rows() != m_dim ){
			throw std::runtime_error("Error in Problem::value: x wrong dimension");
		}

		return (A*x-b).norm();
	}

	double gradient(const VectorX &x, VectorX &grad){

		// Check sizes of input
		int m_dim = dim();
		if( x.rows() != m_dim ){
			throw std::runtime_error("Error in Problem::gradient: x wrong dimension");
		}
		if( grad.rows() != m_dim ){
			throw std::runtime_error("Error in Problem::gradient: gradient wrong dimension");
		}

		grad = A*x-b; return value(x);
	}
	void hessian(const VectorX &x, MatrixX &hess){

		// Check sizes of input
		int m_dim = dim();
		if( x.rows() != m_dim ){
			throw std::runtime_error("Error in Problem::hessian: x wrong dimension");
		}
		if( hess.rows() != m_dim || hess.cols() != m_dim ){
			throw std::runtime_error("Error in Problem::hessian: hessian wrong dimension");
		}
		hess = A;
	}

	void solve_hessian(const VectorX &x, const VectorX &grad, VectorX &dx){

		// Check sizes of input
		int m_dim = dim();
		if( x.rows() != m_dim ){
			throw std::runtime_error("Error in Problem::solve_hessian: x wrong dimension");
		}
		if( dx.rows() != m_dim ){
			throw std::runtime_error("Error in Problem::solve_hessian: dx wrong dimension");
		}
		if( grad.rows() != m_dim ){
			throw std::runtime_error("Error in Problem::solve_hessian: gradient wrong dimension");
		}

		// Check to make sure base class function works as expected
		Problem::solve_hessian(x,grad,dx);
	}
};

class Rosenbrock : public mcl::optlib::Problem<double,2> {
public:
	typedef Eigen::Matrix<double,2,1> VectorX;
	bool converged(const VectorX &x0, const VectorX &x1, const VectorX &grad){
		(void)(x1); (void)(x0);
		return grad.norm() < 1e-10;
	}
	double value(const VectorX &x){
		double a = 1.0 - x[0];
		double b = x[1] - x[0]*x[0];
		return a*a + b*b*100.0;
	}
	// Test finite diff as well I guess
};


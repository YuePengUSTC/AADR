//  BSD 3-Clause License
//
//  Copyright (c) 2019, Bailin Deng
//  Modified work Copyright 2020, Wenqing Ouyang, Yue Peng
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
//  * Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ALMGEOMETRYSOLVER_H
#define ALMGEOMETRYSOLVER_H

#include "SolverCommon.h"
#include "Constraint.h"
#include "SPDSolver.h"
#include "LinearRegularization.h"
#include "OMPHelper.h"
#include "AndersonAcceleration.h"
#include "Parameters.h"
#include <vector>
#include <memory>
#include <iostream>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <limits>
#include <deque>
#include <iomanip>
#include <set>

template<unsigned int N>
class ALMGeometrySolver {
protected:
    typedef MatrixT<N, 1> VectorN;
    typedef MatrixT<N, Eigen::Dynamic> MatrixNX;
    typedef MatrixT<Eigen::Dynamic, N> MatrixXN;

public:
    ALMGeometrySolver()
        : n_hard_constraints_(0),
          n_soft_constraints_(0),
          penalty_parameter_(1.0),
          solver_initialized_(false) {
    }

    ~ALMGeometrySolver() {
        for (int i = 0; i < static_cast<int>(hard_constraints_.size()); ++i) {
            if (hard_constraints_[i]) {
                delete hard_constraints_[i];
            }
        }

        for (int i = 0; i < static_cast<int>(soft_constraints_.size()); ++i) {
            if (soft_constraints_[i]) {
                delete soft_constraints_[i];
            }
        }
    }

    bool setup_ADMM(int n_points, Scalar penalty_param,
                    SPDSolverType spd_solver_type = LDLT_SOLVER) {
        Timer timer;
        Timer::EventID t_begin = timer.get_time();

        penalty_parameter_ = penalty_param;
        n_hard_constraints_ = static_cast<int>(hard_constraints_.size());
        n_soft_constraints_ = static_cast<int>(soft_constraints_.size());

        assert(n_points != 0);
        assert((n_hard_constraints_ + n_soft_constraints_) != 0);

        current_x_.setZero(N, n_points);
        new_x_.setZero(N, n_points);

        std::vector<Triplet> triplets;
        int idO = 0;
        for (int i = 0; i < n_hard_constraints_; ++i) {
            hard_constraints_[i]->add_constraint(false, triplets, idO);
        }

        // Set up full global update matrix
        ColMajorSparseMatrix D_hard(idO, n_points);
        D_hard.setFromTriplets(triplets.begin(), triplets.end());
        D_hard.makeCompressed();
        rho_D_hard_t_ = D_hard.transpose() * penalty_parameter_;
        ColMajorSparseMatrix global_mat = rho_D_hard_t_ * D_hard;
        int z_cols = D_hard.rows();
        std::cout << "z_cols = " << z_cols*N << std::endl;

        z_hard_.setZero(N, z_cols);
        //   new_z_.setZero(N, z_cols);

        Dx_hard_.setZero(N, z_cols);

        current_proj_.setZero(N, z_cols);
        soft_energy_.setZero(n_soft_constraints_);

        idO = 0;
        triplets.clear();
        for (int i = 0; i < n_soft_constraints_; ++i) {
            soft_constraints_[i]->add_constraint(true, triplets, idO);
        }
        std::cout << "n_soft_constraints_ = " << n_soft_constraints_ << std::endl;

        ColMajorSparseMatrix D_soft(idO, n_points);
        D_soft.setFromTriplets(triplets.begin(), triplets.end());
        D_soft.makeCompressed();
        D_soft_t_ = D_soft.transpose();
        rho_D_soft_t_ = penalty_parameter_ * D_soft_t_;
        global_mat += rho_D_soft_t_ * D_soft;
        Dx_soft_.setZero(N, D_soft.rows());
        P_Dx_soft_.setZero(N, D_soft.rows());
        z_soft_.setZero(N, D_soft.rows());
        
        current_z_.setZero(N, z_cols + D_soft.rows());
        new_z_.setZero(N, z_cols + D_soft.rows());
        current_u_.setZero(N, z_cols+ D_soft.rows());
        new_u_.setZero(N, z_cols+ D_soft.rows());
        current_s_.setZero(N, z_cols + D_soft.rows());
        s_hard_.setZero(N, z_cols);
        s_soft_.setZero(N, D_soft.rows());
        x_temp_.setZero(N, z_cols + D_soft.rows());
        x_temp_hard_.setZero(N, z_cols );
        x_temp_soft_.setZero(N, D_soft.rows());
        current_v_.setZero(N, z_cols + D_soft.rows());
        current_g_.setZero(N, z_cols + D_soft.rows());
        // Set up regularization terms
        rhs_fixed_.setZero(n_points, N);
        x_system_rhs_.setZero(n_points, N);
        MatrixXN regularization_rhs;
        ColMajorSparseMatrix L;
        if (regularization_.get_regularization_system(n_points, L,
                                                      regularization_rhs)) {
            ColMajorSparseMatrix Lt = L.transpose();
            global_mat += Lt * L;
            rhs_fixed_ = Lt * regularization_rhs;

            //Add by PY
            L_ = L;
            L_.makeCompressed();
            regularization_rhs_ = regularization_rhs;
        }

        if (spd_solver_type == LDLT_SOLVER) {
            SPD_solver_ = std::make_shared<SimplicialLDLTSolver>();
        } else {
            SPD_solver_ = std::make_shared<SimplicialLLTSolver>();
        }

        SPD_solver_->initialize(global_mat);
        if (SPD_solver_->info() != Eigen::Success) {
            std::cerr << "Error: SPD solver initialization failed" << std::endl;
            return false;
        }

        std::cout << "predecomposition time = "
                  << timer.elapsed_time(t_begin, timer.get_time()) << std::endl;

        solver_initialized_ = true;

        return true;
    }

    void solve_ADMM_zxu(const MatrixNX& init_x, Scalar rel_residual_eps, int max_iter,
                        int Anderson_m) {
        if (!solver_initialized_) {
            std::cerr << "Error: solver not initialized yet" << std::endl;
        }

        bool accel = Anderson_m > 0;

        init_variables(init_x);
        OMP_PARALLEL
        {
            ADMM_compute_Dx_hard(current_x_);
        }
        current_z_.block(0, 0, N, z_hard_.cols()) = Dx_hard_;
        z_hard_ = Dx_hard_;
        int iter_count = 0;
        Scalar residual_eps = rel_residual_eps * rel_residual_eps * z_hard_.cols() * z_hard_.cols() * 2;
        Scalar prev_residual = std::numeric_limits<Scalar>::max();
        Scalar current_residual = 0;
        bool accept = false, reset = false;
        bool end_iteration = false;


        Scalar constraint_err_;
        OMP_PARALLEL
        {
            ADMM_compute_Dx_soft(current_x_);

        }
        soft_constraints_[0]->project(Dx_soft_, P_Dx_soft_, true, &constraint_err_);
        current_z_.block(0, z_hard_.cols(), N, z_soft_.cols()) = Dx_soft_;


        AndersonAcceleration* aa = NULL;

        if (accel) {
            aa = new AndersonAcceleration(Anderson_m, current_x_.size() + current_u_.size(),
                                          current_x_.size() + current_u_.size());
            aa->init(current_u_, current_x_);
        }

        Timer timer;
        Timer::EventID t_begin = timer.get_time();

        while (!end_iteration) {
            OMP_PARALLEL
            {
                ADMM_compute_Dx_hard(current_x_);//new_x -> Dx_hard
                ADMM_compute_Dx_soft(current_x_);//new_x -> Dx_soft
                ADMM_z_update();
                
                ADMM_x_update();
                OMP_SINGLE
                {
                    prev_Dx_hard_ = Dx_hard_;
                    prev_Dx_soft_ = Dx_soft_;
                }
                ADMM_compute_Dx_hard(new_x_);//new_x -> Dx_hard
                ADMM_compute_Dx_soft(new_x_);//new_x -> Dx_soft
                ADMM_u_update();

                OMP_SINGLE
                {
                    //   current_residual = (Dx_hard_ - new_z_).squaredNorm() + (new_z_ - z_hard_).squaredNorm();
                    current_residual = (Dx_hard_ - current_z_.block(0,0,N,z_hard_.cols())).squaredNorm() + (Dx_soft_ - current_z_.block(0,z_hard_.cols(),N,z_soft_.cols())).squaredNorm()
                            + (prev_Dx_hard_ - Dx_hard_).squaredNorm() + (prev_Dx_soft_ - Dx_soft_).squaredNorm();

                    accept = (!accel) || reset || current_residual < prev_residual;

                    if (accept) {
                        default_x_ = new_x_;
                        default_z_ = current_z_;
                        default_u_ = new_u_;
                        iter_count++;
                        std::cout << "Iteration " << iter_count << ":   ";
                        std::cout << "combined residual: " << current_residual;

                        if(reset){
                            std::cout << ", AA was reset";
                        }
                        std::cout << std::endl;

                        function_values_.push_back(current_residual);
                        step_comb_residual_.push_back(current_residual);
                        Timer::EventID t_iter = timer.get_time();
                        elapsed_time_.push_back(timer.elapsed_time(t_begin, t_iter));

                        prev_residual = current_residual;
                        reset = false;

                        if (accel) {
                            aa->compute(new_u_, new_x_, current_u_, current_x_);
                        }
                        else {
                            current_x_ = new_x_;
                            current_u_ = new_u_;
                        }
                    }
                    else {
                        current_x_ = default_x_;
                        current_u_ = default_u_;
                        reset = true;

                        if (accel) {
                            aa->reset(current_u_, current_x_);
                        }
                    }

                    //           if(iter_count >= max_iter || current_residual < residual_eps){
                    //             end_iteration = true;
                    //           }
                    if (iter_count >= max_iter) {
                        end_iteration = true;
                    }


                }
            }
        }
        //Scalar constraint_err_;

        OMP_PARALLEL
        {
            ADMM_compute_Dx_soft(current_x_);
        }

        soft_constraints_[0]->project(Dx_soft_, z_soft_, true, &constraint_err_);

        if (aa) {
            delete aa;
        }
    }

    void solve_DR_zxu(const MatrixNX& init_x, Scalar rel_residual_eps, int max_iter,
                      int Anderson_m) {
        if (!solver_initialized_) {
            std::cerr << "Error: solver not initialized yet" << std::endl;
        }

        bool accel = Anderson_m > 0;

        init_variables(init_x);
        int iter_count = 0;
        //Scalar residual_eps = rel_residual_eps * rel_residual_eps *  z_hard_.cols() * z_hard_.cols() * 2;
        Scalar prev_energy = std::numeric_limits<Scalar>::max();
        Scalar current_residual = 0, current_energy = 0;
        bool accept = false;//, reset = false;
        bool end_iteration = false;
        AndersonAcceleration* aa = NULL;

        MatrixNX energy_calc_Vec1, energy_calc_Vec2;
        VectorN energy_V;

        Scalar constraint_err_;
        OMP_PARALLEL
        {
            DR_compute_Dx_soft(current_x_);
            DR_compute_Dx_hard(current_x_);//x -> Dx_hard
        }
        soft_constraints_[0]->project(Dx_soft_, P_Dx_soft_, true, &constraint_err_);

        MatrixNX dual_residual_Mat = current_u_, prim_residual_Mat = current_v_;

        //Initialization
        z_hard_ = Dx_hard_;//Ax-z=c
        current_s_.block(0, 0, N, z_hard_.cols()) = Dx_hard_ - current_u_.block(0, 0, N, z_hard_.cols());//Here current_u_ = 0, y0 initialization?
        current_s_.block(0, z_hard_.cols(), N, z_soft_.cols()) = Dx_soft_ - current_u_.block(0, z_hard_.cols(), N, z_soft_.cols());

        OMP_PARALLEL
        {
            //Update z
            DR_x_update();
            DR_compute_Dx_soft(current_x_);
            DR_compute_Dx_hard(current_x_);//x -> Dx_hard
            OMP_SINGLE
            {
                //Update u
                current_u_.block(0,0,N,z_hard_.cols()) = Dx_hard_;
                current_u_.block(0, z_hard_.cols(), N, z_soft_.cols()) = Dx_soft_;
            }

            //Update x
            DR_z_update();

            //Update v
            DR_compute_v();

            OMP_SINGLE
            {
                //Update G(s)
                current_g_ = current_s_ + current_v_ - current_u_;

                aa = new AndersonAcceleration(Anderson_m, current_s_.size(), current_s_.size());
                aa->init(current_s_);
            }
        }
        //        }

        int reset_num = 0;
        double total_time = 0.0;
        Timer timer;

        while (!end_iteration) {
            Timer::EventID t_begin = timer.get_time();
            OMP_PARALLEL
            {
                OMP_SINGLE
                {
                    default_g_ = current_g_;
                    aa->compute(current_g_, current_s_);
                }

                //Update z
                DR_x_update();
                DR_compute_Dx_soft(current_x_);
                DR_compute_Dx_hard(current_x_);//x -> Dx_hard
                OMP_SINGLE
                {
                    //Update u
                    current_u_.block(0,0,N,z_hard_.cols()) = Dx_hard_;
                    current_u_.block(0, z_hard_.cols(), N, z_soft_.cols()) = Dx_soft_;
                }

                //Update x
                DR_z_update();

                //Update v
                DR_compute_v();

                OMP_SINGLE
                {
                    //Update v and g
                    energy_calc_Vec2 = current_v_ - current_u_;
                    current_energy = energy_calc_Vec2.norm();

                    accept = (current_energy < prev_energy);// || (!accel)
                }
            }

            if (!accept)
            {
                OMP_PARALLEL
                {
                    OMP_SINGLE
                    {
                        reset_num++;
                        current_s_ = default_g_;
                        aa->reset(current_s_);
                    }

                    //Update z
                    DR_x_update();
                    DR_compute_Dx_soft(current_x_);
                    DR_compute_Dx_hard(current_x_);//x -> Dx_hard
                    OMP_SINGLE
                    {
                        //Update u
                        current_u_.block(0,0,N,z_hard_.cols()) = Dx_hard_;
                        current_u_.block(0, z_hard_.cols(), N, z_soft_.cols()) = Dx_soft_;
                    }

                    //Update x
                    DR_z_update();

                    //Update v
                    DR_compute_v();

                    OMP_SINGLE
                    {
                        //Update v and g
                        energy_calc_Vec2 = current_v_ - current_u_;
                        current_energy = energy_calc_Vec2.norm();
                    }
                }
            }
            prev_energy = current_energy;
            current_g_ = current_s_ + energy_calc_Vec2;

            Timer::EventID t_iter = timer.get_time();
            total_time += timer.elapsed_time(t_begin, t_iter);
            elapsed_time_.push_back(total_time);
            function_values_.push_back(current_energy);

            //To get the combined residual, unnecessary in application.
            OMP_PARALLEL
            {
                //Update z
                DR_x_update();
                DR_compute_Dx_soft(current_x_);
                DR_compute_Dx_hard(current_x_);//x -> Dx_hard
                OMP_SINGLE
                {
                    //Update u
                    new_u_.block(0,0,N,z_hard_.cols()) = Dx_hard_;
                    new_u_.block(0, z_hard_.cols(), N, z_soft_.cols()) = Dx_soft_;
                }

            }

            //Get residual, unnecessay in application, Dx_soft_ = Ai*current_x_
            dual_residual_Mat = new_u_ - current_u_;
            prim_residual_Mat = current_v_ - new_u_;
            current_residual = dual_residual_Mat.squaredNorm() + prim_residual_Mat.squaredNorm();//

            //Save step combined residual
            step_comb_residual_.push_back(current_residual);

            std::cout << "Iteration " << iter_count << ":   ";
            std::cout << "combined residual: " << current_residual;
            std::cout << std::endl;

            iter_count++;
            if (iter_count >= max_iter) {
                end_iteration = true;
            }
        }
        //Scalar constraint_err_;
        default_x_ = current_x_;

        //For Wiremesh design
        soft_constraints_[0]->project(Dx_soft_, z_soft_, true, &constraint_err_);
        std::cout << "reset number = " << reset_num << std::endl;

        if (aa) {
            delete aa;
        }
    }

    const MatrixNX& get_solution() {
        return default_x_;
    }

    void add_hard_constraint(Constraint<N>* constraint) {
        hard_constraints_.push_back(constraint);
    }

    void add_soft_constraint(Constraint<N>* constraint) {
        soft_constraints_.push_back(constraint);
    }

    void add_closeness(int idx, Scalar weight, const VectorN &target_pt) {
        regularization_.add_closeness(idx, weight, target_pt);
    }

    void add_uniform_laplacian(const std::vector<int> &indices, Scalar weight) {
        regularization_.add_uniform_laplacian(indices, weight);
    }

    void add_laplacian(const std::vector<int> &indices,
                       const std::vector<Scalar> coefs, Scalar weight) {
        regularization_.add_laplacian(indices, coefs, weight);
    }

    void add_relative_uniform_laplacian(const std::vector<int> &indices,
                                        Scalar weight,
                                        const MatrixNX &ref_points) {
        regularization_.add_relative_uniform_laplacian(indices, weight, ref_points);
    }

    void add_relative_laplacian(const std::vector<int> &indices,
                                const std::vector<Scalar> coefs, Scalar weight,
                                const MatrixNX &ref_points) {
        regularization_.add_relative_laplacian(indices, coefs, weight, ref_points);
    }

    void output_iteration_history(SolverType solver_type) {
        if (solver_type == AA_SOLVER)
            assert(function_values_.size() == Anderson_reset_.size());

        assert(function_values_.size() == elapsed_time_.size());
        int n_iter = function_values_.size();
        for (int i = 0; i < n_iter; ++i) {
            std::cout << "Iteration " << i << ": ";
            std::cout << std::setprecision(6) << elapsed_time_[i] << " secs, ";
            std::cout << " target value " << std::setprecision(16)
                      << function_values_[i];
            std::cout << " combined residual " << step_comb_residual_[i];
            if ((solver_type == AA_SOLVER) && (Anderson_reset_[i])) {
                std::cout << " (reject accelerator)";
            }

            std::cout << std::endl;
        }

        std::cout << std::endl;
    }

    void save(int AccType, int Anderson_m)
    {
        std::string file;
        if (AccType == 0)
            file = "./result/residual-no.txt";
        else if (AccType == 1)
            file = "./result/residual-AA.txt";
        else {
            file = "./result/residual-DR.txt";
        }
        //        if (Anderson_m > 0)
        //            file = "./result/residual-AA"+std::to_string(Anderson_m)+".txt";
        //        else
        //            file = "./result/residual-no.txt";

        std::ofstream ofs;
        ofs.open(file, std::ios::out | std::ios::ate);
        if (!ofs.is_open())
        {
            std::cout << "Cannot open: " << file << std::endl;
        }

        ofs << std::setprecision(16);
        for (size_t i = 0; i < elapsed_time_.size(); i++)
        {
            ofs << elapsed_time_[i] << '\t' << step_comb_residual_[i] << '\t' << function_values_[i] << std::endl;
        }

        ofs.close();
    }

protected:
    MatrixNX current_x_, current_u_, current_z_, current_s_, current_v_, current_g_, current_proj_, new_x_, new_u_, new_z_,s_hard_,s_soft_,x_temp_, x_temp_hard_, x_temp_soft_;
    MatrixNX default_x_, default_u_, default_g_, default_z_;
    MatrixNX Dx_hard_, Dx_soft_, z_hard_, z_soft_, P_Dx_soft_;
    MatrixNX prev_Dx_hard_, prev_z_hard_, prev_Dx_soft_,prev_z_soft_;

    Scalar merit_err_;
    VectorX soft_energy_;

    // Constraints and regularization terms
    std::vector<Constraint<N>*> soft_constraints_;
    std::vector<Constraint<N>*> hard_constraints_;
    int n_hard_constraints_, n_soft_constraints_;

    Scalar penalty_parameter_;

    LinearRegularization<N> regularization_;

    // Data structures used for direct solve update
    std::shared_ptr<SPDSolver> SPD_solver_;

    MatrixXN x_system_rhs_;
    ColMajorSparseMatrix rho_D_hard_t_, D_soft_t_, L_, rho_D_soft_t_;
    MatrixXN rhs_fixed_, regularization_rhs_;

public:
    // History of iterations
    std::deque<bool> Anderson_reset_;
    std::vector<double> function_values_;
    std::vector<double> elapsed_time_;
    std::vector<double> step_comb_residual_;

private:
    bool solver_initialized_;

    void clear_iteration_history() {
        Anderson_reset_.clear();
        function_values_.clear();
        elapsed_time_.clear();
        step_comb_residual_.clear();
    }

    void init_variables(const MatrixNX &init_x) {
        current_x_ = init_x;
        default_x_ = init_x;
        default_g_.setZero();
        current_u_.setZero();
        default_u_.setZero();
    }

    void ADMM_compute_Dx_hard(const MatrixNX &x) {
        OMP_FOR
                for (int i = 0; i < n_hard_constraints_; ++i) {
            hard_constraints_[i]->apply_transform(x, Dx_hard_);
        }
    }

    void ADMM_compute_Dx_soft(const MatrixNX &x){
        OMP_FOR
                for (int i = 0; i < n_soft_constraints_; ++i) {
            soft_constraints_[i]->apply_transform(x, Dx_soft_);
        }
    }

    void ADMM_z_update() {
        OMP_SINGLE
        {
            current_proj_ = Dx_hard_ + current_u_.block(0,0,N,z_hard_.cols());
        }

        OMP_FOR
                for (int i = 0; i < n_hard_constraints_; ++i) {
            hard_constraints_[i]->project(current_proj_, z_hard_);
        }
        OMP_SINGLE
        {
            current_z_.block(0, 0, N, z_hard_.cols()) = z_hard_;
        }

        OMP_SINGLE
        {
            P_Dx_soft_ = Dx_soft_ + current_u_.block(0,z_hard_.cols(),N,z_soft_.cols());
        }
        OMP_FOR
                for (int i = 0; i < n_soft_constraints_; ++i) {
            soft_constraints_[i]->project(P_Dx_soft_, z_soft_, true);
        }
        OMP_SINGLE
        {
            current_z_.block(0,z_hard_.cols(), N,z_soft_.cols()) = (penalty_parameter_ * P_Dx_soft_ + z_soft_) / (1 + penalty_parameter_);
        }
    }

    void ADMM_x_update() {
        OMP_FOR
                for (int i = 0; i < int(N); ++i) {
            //x_system_rhs_.col(i) = rhs_fixed_.col(i)
            //        + rho_D_hard_t_ * (z_hard_.row(i) - current_u_.row(i)).transpose()
            //        + D_soft_t_ * z_soft_.row(i).transpose();
            x_system_rhs_.col(i) = rhs_fixed_.col(i)
                    + rho_D_hard_t_ * (current_z_.block(i, 0, 1, z_hard_.cols()) - current_u_.block(i, 0, 1, z_hard_.cols())).transpose()
                    + rho_D_soft_t_ * (current_z_.block(i, z_hard_.cols(), 1, z_soft_.cols()) - current_u_.block(i, z_hard_.cols(), 1, z_soft_.cols())).transpose();
            new_x_.row(i) = SPD_solver_->solve(x_system_rhs_.col(i)).transpose();
        }
    }

    void ADMM_u_update() {
        OMP_SINGLE
        {
            new_u_.block(0, 0, N, z_hard_.cols()) = current_u_.block(0, 0, N, z_hard_.cols()) + Dx_hard_ - current_z_.block(0,0,N,z_hard_.cols());
            new_u_.block(0, z_hard_.cols(), N, z_soft_.cols()) = current_u_.block(0, z_hard_.cols(), N, z_soft_.cols()) + Dx_soft_ - current_z_.block(0, z_hard_.cols(), N, z_soft_.cols());
        }
    }

    void ADMM_xzu_x_update() {
        OMP_FOR
                for (int i = 0; i < int(N); ++i) {
            //x_system_rhs_.col(i) = rhs_fixed_.col(i)
            //        + rho_D_hard_t_ * (z_hard_.row(i) - current_u_.row(i)).transpose()
            //        + D_soft_t_ * z_soft_.row(i).transpose();
            x_system_rhs_.col(i) = rhs_fixed_.col(i)
                    + rho_D_hard_t_ * (current_z_.block(i,0,1, z_hard_.cols()) - current_u_.block(i, 0, 1, z_hard_.cols())).transpose()
                    + rho_D_soft_t_ * (current_z_.block(i, z_hard_.cols(), 1, z_soft_.cols()) - current_u_.block(i, z_hard_.cols(), 1, z_soft_.cols())).transpose();
            current_x_.row(i) = SPD_solver_->solve(x_system_rhs_.col(i)).transpose();
        }
    }

    void ADMM_xzu_z_update() {
        OMP_SINGLE
        {
            current_proj_ = Dx_hard_ + current_u_.block(0,0,N,z_hard_.cols());
        }

        OMP_FOR
                for (int i = 0; i < n_hard_constraints_; ++i) {
            hard_constraints_[i]->project(current_proj_,z_hard_);
        }
        OMP_SINGLE
        {
            new_z_.block(0, 0, N, z_hard_.cols()) = z_hard_;
        }

        OMP_SINGLE
        {
            P_Dx_soft_ = Dx_soft_ + current_u_.block(0,z_hard_.cols(),N,z_soft_.cols());
        }
        OMP_FOR
                for (int i = 0; i < n_soft_constraints_; ++i) {
            soft_constraints_[i]->project(P_Dx_soft_, z_soft_, true);
        }
        OMP_SINGLE
        {
            new_z_.block(0,z_hard_.cols(), N,z_soft_.cols()) = (penalty_parameter_ * P_Dx_soft_ + z_soft_) / (1 + penalty_parameter_);
        }

        
    }

    void ADMM_xzu_u_update() {
        OMP_SINGLE
        {
            new_u_.block(0, 0, N, z_hard_.cols()) = current_u_.block(0, 0, N, z_hard_.cols()) + Dx_hard_ - new_z_.block(0,0,N,z_hard_.cols());
            new_u_.block(0, z_hard_.cols(), N, z_soft_.cols()) = current_u_.block(0, z_hard_.cols(), N, z_soft_.cols()) + Dx_soft_ - new_z_.block(0, z_hard_.cols(), N, z_soft_.cols());
        }
    }

    Scalar get_combined_residual() {
        return (Dx_hard_ - z_hard_).squaredNorm() + (Dx_hard_ - prev_Dx_hard_).squaredNorm();
    }

    void DR_compute_Dx_hard(const MatrixNX &x) {
        OMP_FOR
                for (int i = 0; i < n_hard_constraints_; ++i) {
            hard_constraints_[i]->apply_transform(x, Dx_hard_);
        }
    }

    void DR_compute_Dx_soft(const MatrixNX &x){
        OMP_FOR
                for (int i = 0; i < n_soft_constraints_; ++i) {
            soft_constraints_[i]->apply_transform(x, Dx_soft_);
        }
    }

    void DR_compute_u(const MatrixNX &x) {
        OMP_FOR
                for (int i = 0; i < n_hard_constraints_; ++i) {
            hard_constraints_[i]->apply_transform(x, current_u_);
        }
    }

    void DR_z_update() {
        OMP_SINGLE{
            auto tem = 2 * current_u_ - current_s_;
            s_hard_ = tem.block(0,0,N,z_hard_.cols());
            s_soft_ = tem.block(0, z_hard_.cols(), N, z_soft_.cols());
        }

        OMP_FOR
                for (int i = 0; i < n_hard_constraints_; ++i) {
            hard_constraints_[i]->project(s_hard_, z_hard_);
        }

        OMP_FOR
                for (int i = 0; i < n_soft_constraints_; ++i) {
            Scalar energy = 0;
            soft_constraints_[i]->project(s_soft_, P_Dx_soft_, true, &energy);
            soft_energy_(i) = energy;
        }
        OMP_SINGLE{
            z_soft_ = (penalty_parameter_ * s_soft_ + P_Dx_soft_) / (1 + penalty_parameter_);
        }
    }

    void DR_x_update() {
        OMP_SINGLE
        {
            x_temp_ = current_s_;
            x_temp_hard_ = x_temp_.block(0, 0, N, z_hard_.cols());
            x_temp_soft_ = x_temp_.block(0, z_hard_.cols(), N, z_soft_.cols());
        }
        //  auto temp = 2.0 * current_u_ - current_s_;
        OMP_FOR
                for (int i = 0; i < int(N); ++i) {
            x_system_rhs_.col(i) = rhs_fixed_.col(i)
                    + rho_D_hard_t_ * (x_temp_hard_.row(i)).transpose()
                    + rho_D_soft_t_ * (x_temp_soft_.row(i)).transpose();
            current_x_.row(i) = SPD_solver_->solve(x_system_rhs_.col(i)).transpose();
        }
    }
    void DR_compute_v() {
        OMP_SINGLE{
            current_v_.block(0, 0, N, z_hard_.cols()) = z_hard_;
            current_v_.block(0, z_hard_.cols(), N, z_soft_.cols()) = z_soft_;
        }

    }
    void DR2_compute_v(const MatrixNX &x) {
        OMP_FOR
                for (int i = 0; i < n_hard_constraints_; ++i) {
            hard_constraints_[i]->apply_transform(x,Dx_hard_);
        }
        OMP_FOR
                for (int i = 0; i < n_soft_constraints_; ++i) {
            soft_constraints_[i]->apply_transform(x, Dx_soft_);
        }
        OMP_SINGLE{
            current_v_.block(0, 0, N, z_hard_.cols()) = Dx_hard_;
            current_v_.block(0, z_hard_.cols(), N, z_soft_.cols()) = Dx_soft_;
        }

    }

    void DR2_z_update() {
        OMP_SINGLE{
            s_hard_ = current_s_.block(0,0,N,z_hard_.cols());
            s_soft_ = current_s_.block(0, z_hard_.cols(), N, z_soft_.cols());
        }

        OMP_FOR
                for (int i = 0; i < n_hard_constraints_; ++i) {
            hard_constraints_[i]->project(s_hard_, z_hard_);
        }

        OMP_FOR
                for (int i = 0; i < n_soft_constraints_; ++i) {
            Scalar energy = 0;
            soft_constraints_[i]->project(s_soft_, P_Dx_soft_, true, &energy);
            soft_energy_(i) = energy;
        }
        OMP_SINGLE{
            z_soft_ = (penalty_parameter_ * s_soft_ + P_Dx_soft_) / (1 + penalty_parameter_);
        }

    }

    void DR2_x_update() {
        OMP_SINGLE
        {
            x_temp_ = 2.0 * current_u_ - current_s_;
            x_temp_hard_ = x_temp_.block(0, 0, N, z_hard_.cols());
            x_temp_soft_ = x_temp_.block(0, z_hard_.cols(), N, z_soft_.cols());
        }
        //  auto temp = 2.0 * current_u_ - current_s_;
        OMP_FOR
                for (int i = 0; i < int(N); ++i) {
            x_system_rhs_.col(i) = rhs_fixed_.col(i)
                    + rho_D_hard_t_ * (x_temp_hard_.row(i)).transpose()
                    + rho_D_soft_t_ * (x_temp_soft_.row(i)).transpose();
            current_x_.row(i) = SPD_solver_->solve(x_system_rhs_.col(i)).transpose();
        }
    }

};

#endif	// ALMGEOMETRYSOLVER_H

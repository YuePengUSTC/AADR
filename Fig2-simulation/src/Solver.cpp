
// Original work Copyright (c) 2017, University of Minnesota
// Modified work Copyright 2020, Yue Peng
//
// ADMM-Elastic Uses the BSD 2-Clause License (http://www.opensource.org/licenses/BSD-2-Clause)
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
//    conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list
//    of conditions and the following disclaimer in the documentation and/or other materials
//    provided with the distribution.
// THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR  A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF MINNESOTA, DULUTH OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
// IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include "Solver.hpp"
#include "MCL/MicroTimer.hpp"
#include <fstream>
#include <unordered_set>
#include <unordered_map>

using namespace admm;
using namespace Eigen;

Solver::Solver() : initialized(false) {
    m_constraints = std::make_shared<ConstraintSet>( ConstraintSet() );
}

void Solver::step()
{
    if (m_settings.acceleration_type == Settings::ANDERSON)
    {
        stepDRzux_s();
    } else
    {
        stepADMMxzu();
    }
}
/*void Solver::stepDRxuz_s(){
    if( m_settings.verbose > 0 ){
        std::cout << "\nSimulating with dt: " <<
                     m_settings.timestep_s << "s..." << std::flush;
    }

    mcl::MicroTimer t;

    // Other const-variable short names and runtime data
    const int dof = static_cast<int>(m_x.rows());
    const int n_nodes = dof/3;
    const double dt = m_settings.timestep_s;
    const int n_energyterms = static_cast<int>(energyterms.size());
    const int n_threads = std::min(n_energyterms, omp_get_max_threads());
    m_runtime = RuntimeData(); // reset

    // Take an explicit step to get predicted node positions
    // with simple forces (e.g. wind).
    for( size_t i=0; i<ext_forces.size(); ++i ){
        ext_forces[i]->project( dt, m_x, m_v, m_masses );
    }

    // Add gravity
    if( std::abs(m_settings.gravity)>0 ){
        for( int i=0; i<n_nodes; ++i ){
            if (positive_pin(i) > 0)// Free points
            {
                m_v[i*3+1] += dt*m_settings.gravity;
            }
        }
    }

    int count = 0;
    for( int i=0; i<n_nodes; ++i ){
        if (positive_pin(i) > 0)// Free points
        {
            m_x_free.segment<3>(3*count) = m_x.segment<3>(i*3) + dt * m_v.segment<3>(i*3);
            count++;
        }
    }

    // Storing some of these variables as members can reduce allocation overhead
    // and speed up run time.

    // Position without elasticity/constraints
    VecX x_bar = m_x_free;
    VecX M_xbar = m_M * x_bar;
    VecX curr_x = x_bar, new_x = curr_x; // Temperorary x used in optimization

    // Initialize ADMM vars
    VecX C_fix = m_C * m_x_pin;
    VecX A_xbar = m_D*x_bar;
    VecX curr_z = solver_W_inv * (A_xbar - C_fix), default_z = curr_z;//m_D*x_bar;
    int z_size = static_cast<int>(curr_z.rows());

    VecX curr_u = VecX::Zero( z_size ), new_u = curr_u;
    VecX solver_termB = VecX::Zero( m_x_free.rows() );

    //Anderson acceleration
    AndersonAcceleration *accelerator=NULL;

    VecX dual_residual_Vec = VecX::Zero( z_size );
    VecX prim_residual_Vec = VecX::Zero( z_size );
    double comb_residual = 0.0;
    VecX energy_sum = VecX::Zero( n_energyterms );
    double eps = 1e-20, prev_energy = 1e+20, curr_energy = 0.0, beta = m_settings.beta;

    t.reset();
    //Initialize x0 = curr_x, y0 = curr_z, eta = 0.5, s0 = Ax0 - y0.
    VecX curr_s = m_D * curr_x - curr_z, curr_v = VecX::Zero( z_size ),
            new_s = VecX::Zero( z_size ), default_g = VecX::Zero( z_size );

//    dual_residual_Vec = solver_W * (curr_z - default_z);
//    prim_residual_Vec = m_D * curr_x - solver_W *curr_z - C_fix;
//    comb_residual = dual_residual_Vec.squaredNorm() + prim_residual_Vec.squaredNorm();
//    step_comb_residual.push_back( comb_residual );
//    m_runtime.step_time.push_back(m_runtime.local_ms+m_runtime.global_ms+m_runtime.acceleration_ms);

    if(m_settings.acceleration_type == Settings::ANDERSON){
        //Initialize acceleration
        accelerator = new AndersonAcceleration(m_settings.Anderson_m,
                                               z_size,
                                               z_size);
        accelerator->init(curr_s);
        //accelerator.init(m_settings.Anderson_m, z_size, curr_s);
    }
    m_runtime.initialization_ms += t.elapsed_ms();

//    if (m_settings.acceleration_type == Settings::ANDERSON)
//    {
//        dual_residual_Vec = curr_u - default_u;
//        prim_residual_Vec = curr_v - curr_u;

//    } else{
//        dual_residual_Vec = solver_W * (curr_z - default_z);
//        prim_residual_Vec = m_D * curr_x - solver_W *curr_z - C_fix;
//    }
//    comb_residual = dual_residual_Vec.squaredNorm() + prim_residual_Vec.squaredNorm();
//    step_comb_residual.push_back( comb_residual );
//    m_runtime.step_time.push_back(m_runtime.local_ms+m_runtime.global_ms+m_runtime.acceleration_ms);


    int reset_num = 0;
    bool reset = false;
    // Run a timestep
    int s_i = 0;
    for( ; s_i < m_settings.admm_iters; ++s_i ){
        if (m_settings.acceleration_type == Settings::ANDERSON)
        {
            t.reset();

            //Update x
            solver_termB.noalias() = M_xbar + solver_Dt_Wt_W * curr_s;
            m_linsolver->solve( curr_x, solver_termB );

            //Update u
            curr_u = m_D * curr_x;

            //Update z
        #pragma omp parallel for num_threads(n_threads)
            for( size_t i=0; i<energyterms.size(); ++i ){
                energyterms[i]->update_DR_z( solver_W_inv, solver_W, curr_z, curr_s, curr_u, C_fix );
            }

            //Update v and g
            curr_v = solver_W * curr_z + C_fix;
            new_s = curr_s + curr_v - curr_u;

            prim_residual_Vec = curr_v - curr_u;
            curr_energy = prim_residual_Vec.squaredNorm();

            m_runtime.local_ms += t.elapsed_ms();

            //============================================================//

            //To get the combined residual, unnecessary in application.
            //Update new_x_ from G(.)
            solver_termB.noalias() = M_xbar + solver_Dt_Wt_W * new_s;
            m_linsolver->solve( new_x, solver_termB );

            //Update new_u_ from new_x_
            new_u = m_D * new_x;

            dual_residual_Vec = new_u - curr_u;

            comb_residual = beta*((curr_v - new_u).squaredNorm() + dual_residual_Vec.squaredNorm());

        }

        //Compute residual
        t.reset();
        if (m_settings.acceleration_type == Settings::ANDERSON)
        {
            if(m_settings.check_type == Settings::DRE)
            {
                #pragma omp parallel for num_threads(n_threads)
                for( size_t i=0; i<energyterms.size(); ++i ){
                    energy_sum(i) = energyterms[i]->get_all_energy(curr_z);
                }

                curr_energy = Calc_function(curr_x) + energy_sum.sum()
                        + beta * ((curr_s - curr_u).dot(curr_v - curr_u) + 0.5 * (curr_v - curr_u).squaredNorm());
            }

            if (curr_energy < prev_energy || reset)
            {
                reset = false;
                prev_energy = curr_energy;

                default_g = new_s;
                //accelerator.compute(curr_s, default_g);
                accelerator->compute(default_g, curr_s);

            } else
            {
                reset_num++;
                reset = true;
                curr_s = default_g;

//                accelerator.replace(curr_s);
                accelerator->replace(curr_s);
                s_i--;
            }
        }

        m_runtime.acceleration_ms += t.elapsed_ms();

        if (m_settings.acceleration_type == Settings::ANDERSON)
        {          
            if(!reset){
                step_energy.push_back( curr_energy );
                //Save step time
                step_comb_residual.push_back( comb_residual );
                m_runtime.step_time.push_back(m_runtime.local_ms+m_runtime.global_ms+m_runtime.acceleration_ms);


            }

        }

        // End iteration
//        if (comb_residual < eps)
//        {
//            break;
//        }
    } // end solver loop

    std::cout << "XUZ_s report! reset number = " << reset_num << std::endl;

    VecX actual_x = m_S_free * curr_x + m_S_fix * m_x_pin;
    m_v.noalias() = ( actual_x - m_x ) * ( 1.0 / dt );
    m_x = actual_x;

    // Output run time
    if( m_settings.verbose > 0 ){ m_runtime.print(m_settings); }

    saveDR();
}*/ // end timestep iteration

/*void Solver::stepDRxuz_us(){
    if( m_settings.verbose > 0 ){
        std::cout << "\nSimulating with dt: " <<
                     m_settings.timestep_s << "s..." << std::flush;
    }

    mcl::MicroTimer t;

    if (m_settings.acceleration_type == Settings::NOACC)
        return;

    // Other const-variable short names and runtime data
    const int dof = static_cast<int>(m_x.rows());
    const int n_nodes = dof/3;
    const double dt = m_settings.timestep_s;
    const int n_energyterms = static_cast<int>(energyterms.size());
    const int n_threads = std::min(n_energyterms, omp_get_max_threads());
    m_runtime = RuntimeData(); // reset

    // Take an explicit step to get predicted node positions
    // with simple forces (e.g. wind).
    for( size_t i=0; i<ext_forces.size(); ++i ){
        ext_forces[i]->project( dt, m_x, m_v, m_masses );
    }

    // Add gravity
    if( std::abs(m_settings.gravity)>0 ){
        for( int i=0; i<n_nodes; ++i ){
            if (positive_pin(i) > 0)// Free points
            {
                m_v[i*3+1] += dt*m_settings.gravity;
            }
        }
    }

    int count = 0;
    for( int i=0; i<n_nodes; ++i ){
        if (positive_pin(i) > 0)// Free points
        {
            m_x_free.segment<3>(3*count) = m_x.segment<3>(i*3) + dt * m_v.segment<3>(i*3);
            count++;
        }
    }

    // Storing some of these variables as members can reduce allocation overhead
    // and speed up run time.

    // Position without elasticity/constraints
    VecX x_bar = m_x_free;
    VecX M_xbar = m_M * x_bar;
    VecX curr_x = x_bar, new_x = curr_x; // Temperorary x used in optimization

    // Initialize ADMM vars
    VecX C_fix = m_C * m_x_pin;
    VecX A_xbar = m_D*x_bar;
    VecX curr_z = solver_W_inv * (A_xbar - C_fix), default_z = curr_z;//m_D*x_bar;
    int z_size = static_cast<int>(curr_z.rows());

    VecX curr_u = VecX::Zero( z_size ), new_u = curr_u;
    VecX solver_termB = VecX::Zero( m_x_free.rows() );

    //Anderson acceleration
    AndersonAcceleration *accelerator = NULL;

    VecX dual_residual_Vec = VecX::Zero( z_size );
    VecX prim_residual_Vec = VecX::Zero( z_size );
    double comb_residual = 0.0;
    VecX energy_sum = VecX::Zero( n_energyterms );
    double eps = 1e-20, prev_energy = 1e+20, curr_energy = 0.0;

    t.reset();
    //Initialize x0 = curr_x, y0 = curr_z, eta = 0.5, s0 = Ax0 - y0.
    VecX curr_s = m_D * curr_x, curr_v = VecX::Zero( z_size ),
            new_s = VecX::Zero( z_size ), default_g = VecX::Zero( z_size ),
            default_u = curr_u;
    //Initialize u
    curr_u = m_D * curr_x;

//    dual_residual_Vec = solver_W * (curr_z - default_z);
//    prim_residual_Vec = m_D * curr_x - solver_W *curr_z - C_fix;
//    comb_residual = dual_residual_Vec.squaredNorm() + prim_residual_Vec.squaredNorm();
//    step_comb_residual.push_back( comb_residual );
//    m_runtime.step_time.push_back(m_runtime.local_ms+m_runtime.global_ms+m_runtime.acceleration_ms);

    //Initialize acceleration
    accelerator = new AndersonAcceleration(m_settings.Anderson_m,
                                           z_size + z_size,
                                           z_size);
    accelerator->init(curr_s, curr_u);
    m_runtime.initialization_ms += t.elapsed_ms();

    int reset_num = 0;
    bool reset = false;
    // Run a timestep
    int s_i = 0;
    for( ; s_i < m_settings.admm_iters; ++s_i ){

        t.reset();
        //Update z
#pragma omp parallel for num_threads(n_threads)
        for( size_t i=0; i<energyterms.size(); ++i ){
            energyterms[i]->update_DR_z( solver_W_inv, solver_W, curr_z, curr_s, curr_u, C_fix );
        }

        //Update v
        curr_v = solver_W * curr_z + C_fix;

        //Calculate the primal residual
        prim_residual_Vec = curr_v - curr_u;
        curr_energy = prim_residual_Vec.squaredNorm();
        m_runtime.local_ms += t.elapsed_ms();

        //Compute residual
        t.reset();
        //#pragma omp parallel for num_threads(n_threads)
        //            for( size_t i=0; i<energyterms.size(); ++i ){
        //                energy_sum(i) = energyterms[i]->get_all_energy(curr_z);
        //            }

        //            curr_energy = Calc_function(curr_x) + energy_sum.sum() + (curr_s - curr_u).dot(curr_v - curr_u) + 0.5 * (curr_v - curr_u).squaredNorm();

        if (curr_energy > prev_energy && !reset)
        {
            reset_num++;
            reset = true;
            curr_s = default_g;
            curr_u = default_u;
            accelerator->reset(curr_s, curr_u);

            //Update z
    #pragma omp parallel for num_threads(n_threads)
            for( size_t i=0; i<energyterms.size(); ++i ){
                energyterms[i]->update_DR_z( solver_W_inv, solver_W, curr_z, curr_s, curr_u, C_fix );
            }

            //Update v and g
            curr_v = solver_W * curr_z + C_fix;

            //Calculate the primal residual
            prim_residual_Vec = curr_v - curr_u;
            curr_energy = prim_residual_Vec.squaredNorm();
        } else
        {
            reset = false;
        }

        prev_energy = curr_energy;

        //Update s
        new_s = curr_s + curr_v - curr_u;

        //Update x
        solver_termB.noalias() = M_xbar + solver_Dt_Wt_W * new_s;
        m_linsolver->solve( curr_x, solver_termB );

        //Update u
        new_u = m_D * curr_x;
        dual_residual_Vec = new_u - curr_u;

        //Calculate the combined resiudal
        comb_residual = curr_energy + dual_residual_Vec.squaredNorm();

        default_g = new_s;
        default_u = new_u;
        accelerator->compute(default_g, default_u, curr_s, curr_u);

        m_runtime.acceleration_ms += t.elapsed_ms();

        step_energy.push_back( curr_energy );
        //Save step time
        step_comb_residual.push_back( comb_residual );
        m_runtime.step_time.push_back(m_runtime.local_ms+m_runtime.global_ms+m_runtime.acceleration_ms);


        // End iteration
        //        if (comb_residual < eps)
        //        {
        //            break;
        //        }
    } // end solver loop

    std::cout << "XUZ_us report! reset number = " << reset_num << std::endl;

    VecX actual_x = m_S_free * curr_x + m_S_fix * m_x_pin;
    m_v.noalias() = ( actual_x - m_x ) * ( 1.0 / dt );
    m_x = actual_x;

    // Output run time
    if( m_settings.verbose > 0 ){ m_runtime.print(m_settings); }

    saveDR();
}*/

void Solver::stepADMMxzu(){
    if( m_settings.verbose > 0 ){
        std::cout << "\nSimulating with dt: " <<
                     m_settings.timestep_s << "s..." << std::flush;
    }

    mcl::MicroTimer t;

    // Other const-variable short names and runtime data
    const int dof = static_cast<int>(m_x.rows());
    const int n_nodes = dof/3;
    const double dt = m_settings.timestep_s;
    const int n_energyterms = static_cast<int>(energyterms.size());
    const int n_threads = std::min(n_energyterms, omp_get_max_threads());
    m_runtime = RuntimeData(); // reset

    // Take an explicit step to get predicted node positions
    // with simple forces (e.g. wind).
    for( size_t i=0; i<ext_forces.size(); ++i ){
        ext_forces[i]->project( dt, m_x, m_v, m_masses );
    }

    // Add gravity
    if( std::abs(m_settings.gravity)>0 ){
        for( int i=0; i<n_nodes; ++i ){
            if (positive_pin(i) > 0)// Free points
            {
                m_v[i*3+1] += dt*m_settings.gravity;
            }
        }
    }

    int count = 0;
    for( int i=0; i<n_nodes; ++i ){
        if (positive_pin(i) > 0)// Free points
        {
            m_x_free.segment<3>(3*count) = m_x.segment<3>(i*3) + dt * m_v.segment<3>(i*3);
            count++;
        }
    }
    // Storing some of these variables as members can reduce allocation overhead
    // and speed up run time.

    // Position without elasticity/constraints
    VecX x_bar = m_x_free;
    VecX M_xbar = m_M * x_bar;
    VecX curr_x = x_bar; // Temperorary x used in optimization

    // Initialize ADMM vars
    VecX C_fix = m_C * m_x_pin;
    VecX C_fix_noW = C_noW * m_x_pin;
    VecX A_xbar = m_D*x_bar;
    VecX curr_z = D_noW * x_bar - C_fix_noW;//m_D*x_bar;
    VecX last_z = curr_z;

    VecX curr_u = VecX::Zero( curr_z.rows() );
    VecX solver_termB = VecX::Zero( m_x_free.rows() );

    int z_size = static_cast<int>(curr_z.rows());
    VecX dual_residual_Vec = VecX::Zero( z_size );
    VecX prim_residual_Vec = VecX::Zero( z_size );
    double prim_residual = 0.0, comb_residual = 0.0, prev_prim_residual = 1e+20, eps = 1e-20, beta_inv = 1.0/m_settings.beta;

    // Run a timestep
    int s_i = 0;
    for( ; s_i < m_settings.admm_iters; ++s_i ){
        // Update x
        t.reset();
        solver_termB.noalias() = M_xbar + solver_Dt_Wt_W * ( solver_W * (curr_z- curr_u) + C_fix  );
        m_runtime.inner_iters += m_linsolver->solve( curr_x, solver_termB );
        m_runtime.global_ms += t.elapsed_ms();

        t.reset();
        last_z = curr_z;
        //Update z
#pragma omp parallel for num_threads(n_threads)
        for( size_t i=0; i<energyterms.size(); ++i ){
            energyterms[i]->update_z( D_noW, solver_W_inv, solver_W, curr_x, curr_z, curr_u, C_fix_noW);
        }

        // Update u
#pragma omp parallel for num_threads(n_threads)
        for( size_t i=0; i<energyterms.size(); ++i ){
            energyterms[i]->update_u( D_noW, solver_W, curr_x, curr_z, curr_u, C_fix_noW);
        }
        m_runtime.local_ms += t.elapsed_ms();

        //Compute residual
        dual_residual_Vec = solver_W * (curr_z - last_z);
        prim_residual_Vec = m_D * curr_x - solver_W *curr_z - C_fix;
        comb_residual = dual_residual_Vec.squaredNorm() + prim_residual_Vec.squaredNorm();

        //Save step time
        step_prim_residual.push_back( prim_residual );
        step_comb_residual.push_back( comb_residual );
        m_runtime.step_time.push_back(m_runtime.local_ms+m_runtime.global_ms);

        // End iteration
//        if (comb_residual < eps)
//        {
//            break;
//        }
    } // end solver loop

    VecX actual_x = m_S_free * curr_x + m_S_fix * m_x_pin;
    m_v.noalias() = ( actual_x - m_x ) * ( 1.0 / dt );
    m_x = actual_x;

    // Output run time
    if( m_settings.verbose > 0 ){ m_runtime.print(m_settings); }

    saveDR();
} // end timestep iteration

void Solver::stepDRzux_s(){
    if( m_settings.verbose > 0 ){
        std::cout << "\nSimulating with dt: " <<
                     m_settings.timestep_s << "s..." << std::flush;
    }

    mcl::MicroTimer t;

    // Other const-variable short names and runtime data
    const int dof = static_cast<int>(m_x.rows());
    const int n_nodes = dof/3;
    const double dt = m_settings.timestep_s;
    const int n_energyterms = static_cast<int>(energyterms.size());
    const int n_threads = std::min(n_energyterms, omp_get_max_threads());
    m_runtime = RuntimeData(); // reset

    // Take an explicit step to get predicted node positions
    // with simple forces (e.g. wind).
    for( size_t i=0; i<ext_forces.size(); ++i ){
        ext_forces[i]->project( dt, m_x, m_v, m_masses );
    }

    // Add gravity
    if( std::abs(m_settings.gravity)>0 ){
        for( int i=0; i<n_nodes; ++i ){
            if (positive_pin(i) > 0)// Free points
            {
                m_v[i*3+1] += dt*m_settings.gravity;
            }
        }
    }

    int count = 0;
    for( int i=0; i<n_nodes; ++i ){
        if (positive_pin(i) > 0)// Free points
        {
            m_x_free.segment<3>(3*count) = m_x.segment<3>(i*3) + dt * m_v.segment<3>(i*3);
            count++;
        }
    }

    // Storing some of these variables as members can reduce allocation overhead
    // and speed up run time.

    // Position without elasticity/constraints
    VecX x_bar = m_x_free;
    VecX M_xbar = m_M * x_bar;
    VecX curr_x = x_bar;

    // Initialize ADMM vars
    VecX C_fix = m_C * m_x_pin;
    VecX C_fix_noW = C_noW * m_x_pin;
    VecX A_xbar = m_D*x_bar;
    VecX curr_z = D_noW * x_bar - C_fix_noW, default_z = curr_z, new_z = curr_z;//m_D*x_bar; // Temperorary z used in optimization
    int z_size = static_cast<int>(curr_z.rows());
    std::cout << "z_size = " << z_size << std::endl;

    VecX curr_u = VecX::Zero( z_size ), new_u = curr_u;
    VecX solver_termB = VecX::Zero( m_x_free.rows() );

    //Anderson acceleration
    AndersonAcceleration *accelerator = NULL;

    VecX dual_residual_Vec = VecX::Zero( z_size );
    VecX prim_residual_Vec = VecX::Zero( z_size );
    double comb_residual = 0.0;
    VecX energy_sum = VecX::Zero( n_energyterms );
    double eps = 1e-20, prev_energy = 1e+20, curr_energy = 0.0, beta = m_settings.beta;

    // Energy before optimizaiton
    double new_err = 0.0;
    VecX energy_z = D_noW * curr_x - C_fix_noW;
#pragma omp parallel for num_threads(n_threads)
    for( int i=0; i<n_energyterms; ++i ){
        energy_sum(i) = energyterms[i]->get_all_energy( energy_z );
    }
    new_err = Calc_function(curr_x) + dt2_ * energy_sum.sum();
    step_LBFGS_compare_energy.push_back(new_err);
    m_runtime.step_time.push_back(0.0);

    t.reset();
    //Initialize x0 = curr_x, y0 = curr_z, eta = 0.5, s0 = Az0 - y0.
    VecX curr_s = solver_W * curr_z - curr_u, curr_v = VecX::Zero( z_size ),
            new_s = VecX::Zero( z_size ), default_g = VecX::Zero( z_size ), W_inv_s = curr_s;
    SparseMat W_Wt = solver_W * solver_W;
    std::shared_ptr<LinearSolver> solver_W_Wt;
    solver_W_Wt = std::make_shared<LDLTSolver>( LDLTSolver() );
    solver_W_Wt->update_system(W_Wt);

    //Initialization
    // Update x for initialization
    solver_termB.noalias() = M_xbar + solver_Dt_Wt_W * ( solver_W * (curr_z- curr_u) + C_fix  );
    m_runtime.inner_iters += m_linsolver->solve( curr_x, solver_termB );

    //Update z
#pragma omp parallel for num_threads(n_threads)
    for( size_t i=0; i<energyterms.size(); ++i ){
        energyterms[i]->update_z( D_noW, solver_W_inv, solver_W, curr_x, curr_z, curr_u, C_fix_noW);
    }

    // Update u
#pragma omp parallel for num_threads(n_threads)
    for( size_t i=0; i<energyterms.size(); ++i ){
        energyterms[i]->update_u( D_noW, solver_W, curr_x, curr_z, curr_u, C_fix_noW);
    }
    curr_s = solver_W * curr_z - curr_u;

    //Initialize acceleration
    accelerator = new AndersonAcceleration(m_settings.Anderson_m,
                                           z_size,
                                           z_size);
    accelerator->init(curr_s);
    m_runtime.initialization_ms += t.elapsed_ms();

    int reset_num = 0;
    bool reset = false;
    // Run a timestep
    int s_i = 0;
    for( ; s_i < m_settings.admm_iters; ++s_i ){

        t.reset();

        //Update z
        solver_W_Wt->solve(W_inv_s, solver_W*curr_s);
#pragma omp parallel for num_threads(n_threads)
        for( size_t i=0; i<energyterms.size(); ++i ){
            energyterms[i]->update_DR_xzu_z( solver_W_inv, solver_W, curr_z, W_inv_s);
        }

        //Update u
        curr_u = solver_W * curr_z;

        //Update x
        solver_termB.noalias() = M_xbar + solver_Dt_Wt_W * (2*curr_u - curr_s + C_fix);
        m_linsolver->solve( curr_x, solver_termB );

        //Update v
        curr_v = m_D * curr_x - C_fix;

        //Update G(s)
        new_s = curr_s + curr_v - curr_u;

        prim_residual_Vec = curr_v - curr_u;
        curr_energy = prim_residual_Vec.squaredNorm();

        m_runtime.local_ms += t.elapsed_ms();

        //===============================================================//

        //To get the combined residual, unnecessary in application.
        //Update new_z_ from G(.)
        solver_W_Wt->solve(W_inv_s, solver_W*new_s);
#pragma omp parallel for num_threads(n_threads)
        for( size_t i=0; i<energyterms.size(); ++i ){
            energyterms[i]->update_DR_xzu_z( solver_W_inv, solver_W, new_z, W_inv_s);
        }

        //Update new_u_ from new_z_
        new_u = solver_W * new_z;

        dual_residual_Vec = new_u - curr_u;
        comb_residual = beta*((curr_v - new_u).squaredNorm()
                              + dual_residual_Vec.squaredNorm());

        energy_z = D_noW * curr_x - C_fix_noW;
        #pragma omp parallel for num_threads(n_threads)
        for( size_t i=0; i<energyterms.size(); ++i ){
            energy_sum(i) = energyterms[i]->get_all_energy(energy_z);
        }
        new_err = Calc_function(curr_x) + dt2_ * energy_sum.sum();
        step_LBFGS_compare_energy.push_back(new_err);

        //Compute residual
        t.reset();
        if(m_settings.check_type == Settings::DRE)
        {
            #pragma omp parallel for num_threads(n_threads)
            for( size_t i=0; i<energyterms.size(); ++i ){
                energy_sum(i) = energyterms[i]->get_all_energy(curr_z);
            }

            curr_energy = Calc_function(curr_x) / dt2_ + energy_sum.sum()
                    + beta * ((curr_s - curr_u).dot(curr_v - curr_u) + 0.5 * (curr_v - curr_u).squaredNorm());
        }

        if (curr_energy < prev_energy || reset)
        {
            reset = false;
            prev_energy = curr_energy;

            default_g = new_s;
//                accelerator.compute(curr_s, default_g);
            accelerator->compute(default_g, curr_s);

        } else
        {
            reset_num++;
            reset = true;
            curr_s = default_g;

//                accelerator.replace(curr_s);
            accelerator->replace(curr_s);
            s_i--;
        }

        m_runtime.acceleration_ms += t.elapsed_ms();

        if(!reset){
            step_energy.push_back( curr_energy );
            //Save step time
            step_comb_residual.push_back( comb_residual );
            m_runtime.step_time.push_back(m_runtime.local_ms+m_runtime.acceleration_ms);
        }
        //std::cout << "iter = " << s_i << " primal residual = " << curr_energy << std::endl;

        // End iteration
//        if (comb_residual < eps)
//        {
//            break;
//        }
    } // end solver loop

    std::cout << "ZUX_S report! reset number = " << reset_num << std::endl;

    VecX actual_x = m_S_free * curr_x + m_S_fix * m_x_pin;
    m_v.noalias() = ( actual_x - m_x ) * ( 1.0 / dt );
    m_x = actual_x;

    // Output run time
    if( m_settings.verbose > 0 ){ m_runtime.print(m_settings); }

    saveDR();
} // end timestep iteration

double Solver::Calc_function(const VecX &curr_x)
{
    VecX x_div_x_bar = curr_x - m_x_free;
    double energy = x_div_x_bar.transpose() * m_M * x_div_x_bar;
    return 0.5 * energy;// / dt2_
}

double Solver::Calc_function(MatrixXX &all_s, const VecX &curr_x, const VecX &prev_x, int row)
{
    VecX x_div_x_bar = curr_x - m_x_free;

    if (row > -1) {
        VecX F = curr_x - prev_x;

        all_s.row(row) = F.transpose();
    }
    double energy = x_div_x_bar.transpose() * m_M * x_div_x_bar;
    return 0.5 * energy;// / dt2_
}

void Solver::reset_fix_free_S_matrix(){

    const int dof = static_cast<int>(m_x.rows());
    const int pin_dof = 3 * static_cast<int>(m_constraints->pins.size());
    const int free_dof = dof - pin_dof;

    positive_pin.setOnes(static_cast<int>(dof/3));

    SparseMat S_fix(dof, pin_dof), S_free(dof, free_dof);
    Eigen::VectorXi nnz = Eigen::VectorXi::Ones( dof ); // non zeros per column
    S_fix.reserve(nnz);
    S_free.reserve(nnz);

    int count = 0;
    std::map<int,Vec3>::iterator pinIter = m_constraints->pins.begin();
    for( ; pinIter != m_constraints->pins.end(); ++pinIter ){
        //Construct Selection Matrix to select x
        int pin_id = pinIter->first;
        S_fix.coeffRef(3*pin_id, 3*count) = 1.0;
        S_fix.coeffRef(3*pin_id+1, 3*count+1) = 1.0;
        S_fix.coeffRef(3*pin_id+2, 3*count+2) = 1.0;
        positive_pin(pin_id) = 0;
        count++;
        //m_pin_energies[ pinIter->first ] = std::make_shared<SpringPin>( SpringPin(pinIter->first,pinIter->second) );
        //energyterms.emplace_back( m_pin_energies[ pinIter->first ] );
    }
    std::cout << " pin count = " << count << std::endl;

    count = 0;
    for( int i=0; i<positive_pin.size(); ++i )
    {
        if( positive_pin(i) == 1 )//Free point
        {
            S_free.coeffRef(3*i, 3*count) = 1.0;
            S_free.coeffRef(3*i+1, 3*count+1) = 1.0;
            S_free.coeffRef(3*i+2, 3*count+2) = 1.0;
            count++;
        }
    }

    m_S_fix = S_fix;
    m_S_free = S_free;

}

void Solver::set_pins( const std::vector<int> &inds, const std::vector<Vec3> &points ){

    int n_pins = static_cast<int>(inds.size());
    const int dof = static_cast<int>(m_x.rows());
    bool pin_in_place = (points.size() != inds.size());
    if( (dof == 0 && pin_in_place) || (pin_in_place && points.size() > 0) ){
        throw std::runtime_error("**Solver::set_pins Error: Bad input.");
    }

    if (m_x_pin.size() == 0)
    {
        m_x_pin.resize(n_pins*3);
    }

    m_constraints->pins.clear();
    for( size_t i=0; i<inds.size(); ++i ){
        int idx = inds[i];
        int pin_id = static_cast<int>(i)*3;
        if( pin_in_place ){
            m_constraints->pins[idx] = m_x_pin.segment<3>(pin_id);
        } else {
            m_constraints->pins[idx] = points[i];
            m_x_pin.segment<3>(pin_id) = points[i]; // We will not operate the pinned points.
        }
    }

    // If we're using energy based hard constraints, the pin locations may change
    // but which vertex is pinned may NOT change (aside from setting them to
    // active or inactive). So we need to do some extra work here.
    if( initialized ){
        //Update m_S_fix and m_S_free
        reset_fix_free_S_matrix();
    } // end set energy-based pins
}

void Solver::add_obstacle( std::shared_ptr<PassiveCollision> obj ){
    m_constraints->collider->add_passive_obj(obj);
}

void Solver::add_dynamic_collider( std::shared_ptr<DynamicCollision> obj ){
    m_constraints->collider->add_dynamic_obj(obj);
}

bool Solver::initialize( const Settings &settings_ ){
    using namespace Eigen;
    m_settings = settings_;

    mcl::MicroTimer t;
    const int dof = static_cast<int>(m_x.rows());
    if( m_settings.verbose > 0 ){ std::cout << "Solver::initialize: " << std::endl; }

    if( m_settings.timestep_s <= 0.0 ){
        std::cerr << "\n**Solver Error: timestep set to " << m_settings.timestep_s <<
                     "s, changing to 1/24s." << std::endl;
        m_settings.timestep_s = 1.0/24.0;
    }
    if( !( m_masses.rows()==dof && dof>=3 ) ){
        std::cerr << "\n**Solver Error: Problem with node data!" << std::endl;
        return false;
    }
    if( m_v.rows() != dof ){ m_v.resize(dof); }

    // Clear previous runtime stuff settings
    m_v.setZero();


    ///////////////////////////////////////////////
    // If we want energy-based constraints, set them up now.
    int pin_size = static_cast<int>(m_constraints->pins.size());
    int free_dof = dof - 3*pin_size;
    std::cout << "Energy term size = " << energyterms.size() << std::endl;
    std::cout << "pin size = " << pin_size << std::endl;

    Eigen::VectorXi pos_pin = Eigen::VectorXi::Ones( static_cast<int>(dof/3) );
    positive_pin = pos_pin;
    m_x_free.resize(free_dof);
    m_x_free.setZero();

    reset_fix_free_S_matrix();
    std::cout << "Initialized free and fix matrix." << std::endl;
    // end create energy based hard constraints
    ///////////////////////////////////////////////


    // Set up the selector matrix (D) and weight (W) matrix
    std::vector<Eigen::Triplet<double> > triplets;
    std::vector<double> weights;
    for(size_t i = 0; i < energyterms.size(); ++i){
        energyterms[i]->get_reduction( triplets, weights );
    }

    // Create the Selector+Reduction matrix
    m_W_diag = Eigen::Map<VecX>(&weights[0], static_cast<int>(weights.size()));
    int n_D_rows = static_cast<int>(weights.size());
    m_D.resize( n_D_rows, dof );
    m_D.setZero();
    m_D.setFromTriplets( triplets.begin(), triplets.end() );

    //    m_Dt = m_D.transpose();

    // Compute mass matrix
    SparseMat M( free_dof, free_dof ), Inv_M( free_dof, free_dof ); // Inv_M is just for test.
    Eigen::VectorXi nnz = Eigen::VectorXi::Ones( free_dof ); // non zeros per column
    M.reserve(nnz);
    Inv_M.reserve(nnz);
    double eps = 1e-6;
    int count = 0;
    for( int i=0; i<static_cast<int>(dof/3); ++i )
    {
        if (positive_pin(i) > 0)//free point
        {
            M.coeffRef(3*count,3*count) = m_masses[3*i];
            M.coeffRef(3*count+1,3*count+1) = m_masses[3*i+1];
            M.coeffRef(3*count+2,3*count+2) = m_masses[3*i+2];
            if (m_masses[3*i] > eps) {
                Inv_M.coeffRef(3*count,3*count) = 1.0/m_masses[3*i];
                Inv_M.coeffRef(3*count+1,3*count+1) = 1.0/m_masses[3*i+1];
                Inv_M.coeffRef(3*count+2,3*count+2) = 1.0/m_masses[3*i+2];
            } else {
                //                Inv_M.coeffRef(count,count) = 100000000.0;
                Inv_M.coeffRef(3*count,3*count) = 100000000.0;
                Inv_M.coeffRef(3*count+1,3*count+1) = 100000000.0;
                Inv_M.coeffRef(3*count+2,3*count+2) = 100000000.0;
            }
            count++;
        }

    }

    // Set global matrices
    SparseMat W( n_D_rows, n_D_rows ), W_inv( n_D_rows, n_D_rows );
    W.reserve(n_D_rows);
    W_inv.reserve(n_D_rows);
    for( int i=0; i<n_D_rows; ++i ){ W.coeffRef(i,i) = m_W_diag[i]; W_inv.coeffRef(i, i) = 1.0/m_W_diag[i]; }
    const double dt2 = (m_settings.timestep_s*m_settings.timestep_s);

    C_noW = - m_D * m_S_fix;
    D_noW = m_D * m_S_free;
    m_C = W * C_noW;
    m_D = W * D_noW;
    m_Dt = m_D.transpose();
    solver_Dt_Wt_W = dt2 * m_settings.beta * m_Dt;//beta
    solver_termA = M + SparseMat(solver_Dt_Wt_W * m_D);

    solver_W = W;
    solver_W_inv = W_inv;
    dt2_ = dt2;
    m_M = M;

    // Set up the linear solver
    m_linsolver = std::make_shared<LDLTSolver>( LDLTSolver() );

    // If we haven't set a global solver, make one:
    if( !m_linsolver ){ throw std::runtime_error("What happened to the global solver?"); }
    if( m_settings.constraint_w > 0.0 ){ m_constraints->constraint_w = m_settings.constraint_w; }
    m_linsolver->update_system( solver_termA );

    // Make sure they don't have any collision obstacles
    if( m_constraints->collider->passive_objs.size() > 0 ||
            m_constraints->collider->dynamic_objs.size() > 0 ){
        throw std::runtime_error("**Solver::add_obstacle Error: No collisions with LDLT solver");
    }

    // All done
    if( m_settings.verbose >= 1 ){
        std::cout << m_x.size()/3 << " nodes, " << energyterms.size() << " energy terms" << std::endl;
    }
    initialized = true;
    return true;

} // end init


void Solver::save_matrix( const std::string &filename ){

    std::cout << "Saving matrix (" << solver_termA.rows() << "x" <<
                 solver_termA.cols() << ") to " << filename << std::endl;
    std::ofstream(filename.c_str()) << solver_termA;
}


template<typename T> void myclamp( T &val, T min, T max ){ if( val < min ){ val = min; } if( val > max ){ val = max; } }
bool Solver::Settings::parse_args( int argc, char **argv ){

    // Check args with params
    for( int i=1; i<argc-1; ++i ){
        std::string arg( argv[i] );
        std::stringstream val( argv[i+1] );
        if( arg == "-help" || arg == "--help" || arg == "-h" ){ help(); return true; }
        else if( arg == "-dt" ){ val >> timestep_s; }
        else if( arg == "-v" ){ val >> verbose; }
        else if( arg == "-it" ){ val >> admm_iters; }
        else if( arg == "-g" ){ val >> gravity; }
        //        else if( arg == "-ls" ){ val >> linsolver; }
        else if( arg == "-ck" ){ val >> constraint_w; }
        else if( arg == "-a" ){ int acc; val >> acc; (acc == 0) ? (acceleration_type = Solver::Settings::NOACC) : (acceleration_type = Solver::Settings::ANDERSON); }
        else if( arg == "-am" ){ val >> Anderson_m; acceleration_type = Solver::Settings::ANDERSON; }
        else if( arg == "-c" ){ int check_v; val >> check_v; (check_v == 0) ? (check_type = Solver::Settings::RESIDUAL) : (check_type = Solver::Settings::DRE); }
        else if( arg == "-ab" ){ val >> beta; }
    }

    // Check if last arg is one of our no-param args
    std::string arg( argv[argc-1] );
    if( arg == "-help" || arg == "--help" || arg == "-h" ){ help(); return true; }

    return false;

} // end parse settings args

void Solver::Settings::help(){
    std::stringstream ss;
    ss << "\n==========================================\nArgs:\n" <<
          "\t-dt: time step (s)\n" <<
          "\t-v: verbosity (higher -> show more)\n" <<
          "\t-it: # admm iters\n" <<
          "\t-g: gravity (m/s^2)\n" <<
          "\t-ls: linear solver (0=LDLT, 1=NCMCGS, 2=UzawaCG) \n" <<
          "\t-ck: constraint weights (-1 = auto) \n" <<
          "\t-a: acceleration type (0=NoAcc, 1=Anderson) \n" <<
          "\t-am: anderson window size (>0, int) \n" <<
          "\t-c: check type (0=residual, 1=DRE) \n" <<
          "\t-ab: penalty (>0, double) \n" <<
          "==========================================\n";
    printf( "%s", ss.str().c_str() );
}

void Solver::RuntimeData::print( const Settings &settings ){
    std::cout << "\nTotal global step: " << global_ms << "ms";
    std::cout << "\nTotal local step: " << local_ms << "ms";
    std::cout << "\nTotal acceleration step: " << acceleration_ms << "ms";
    std::cout << "\nTotal Initialization time: " << initialization_ms << "ms";
    std::cout << "\nAvg global step: " << global_ms/double(settings.admm_iters) << "ms";
    std::cout << "\nAvg local step: " << local_ms/double(settings.admm_iters) << "ms";
    std::cout << "\nAvg acceleration step: " << acceleration_ms/double(settings.admm_iters) << "ms";
    std::cout << "\nAvg Initialization step: " << initialization_ms/double(settings.admm_iters) << "ms";
    std::cout << "\nADMM Iters: " << settings.admm_iters;
    std::cout << "\nAvg Inner Iters: " << float(inner_iters) / float(settings.admm_iters);
    if (settings.acceleration_type == Solver::Settings::NOACC)
        std::cout << "\nNo Acceleration ";
    else{
        std::cout << "\nAnderson M: " << settings.Anderson_m;
        std::cout << "\nCheck Type: " << ((settings.check_type == Solver::Settings::RESIDUAL)?("check with resiudal."):("check with DRE"));
    }
    std::cout << "\nBeta: " << settings.beta;
    std::cout << std::endl;
}


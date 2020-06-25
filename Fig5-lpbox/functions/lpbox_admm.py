# BSD 3-Clause License

# Original work Copyright (c) 2016, Baoyuan Wu and Bernard Ghanem
# Modified work Copyright (c) 2020, Wenqing Ouyang

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import time
import scipy.sparse.linalg as linalg
from scipy.sparse import csc_matrix


"""
min_x (x^T A x + b^T x) such that x is {0,1}^n
ADMM update steps with x0 being feasible and binary
"""


class Anderson:
    def __init__(self, x0, num):
        self.mk = num
        self.dim = len(x0)
        self.current_F_ = x0.copy()
        self.prev_dG_ = np.zeros((num, self.dim))
        self.prev_dF_ = np.zeros((num, self.dim))
        self.M_ = np.zeros((num, num))  # num*num array
        self.theta_ = np.zeros(num)
        self.dF_scale_ = np.zeros(num)
        self.dG_scale_ = np.zeros(num)
        self.current_u_ = x0.copy()
        self.iter_ = 0
        self.col_idx_ = 0

    def compute(self, g):
        G = g.copy()
        self.current_F_ = G - self.current_u_

        if self.iter_ == 0:
            self.prev_dF_[0, :] = -self.current_F_
            self.prev_dG_[0, :] = - G
            self.current_u_ = G.copy()
        else:
            self.prev_dF_[self.col_idx_, :] += self.current_F_.copy()
            self.prev_dG_[self.col_idx_, :] += G

            eps = 1e-14
            norm = np.linalg.norm(self.prev_dF_[self.col_idx_, :])
            scale = np.maximum(eps, norm)
            self.dF_scale_[self.col_idx_] = scale
            self.prev_dF_[self.col_idx_, :] /= scale

            m_k = np.minimum(self.mk, self.iter_)

            if m_k == 1:
                self.theta_[0] = 0
                dF_norm = np.linalg.norm(self.prev_dF_[self.col_idx_, :])
                self.M_[0, 0] = dF_norm * dF_norm

                if dF_norm > eps:
                    self.theta_[0] = np.dot(self.prev_dF_[self.col_idx_, :] / dF_norm, self.current_F_[:] / dF_norm)
            else:
                new_inner_prod = np.dot(self.prev_dF_[self.col_idx_, :], np.transpose(self.prev_dF_[0:m_k, :]))
                self.M_[self.col_idx_, 0:m_k] = np.transpose(new_inner_prod)
                self.M_[0:m_k, self.col_idx_] = new_inner_prod

                b = np.dot(self.prev_dF_[0:m_k, :], self.current_F_)
                self.theta_[0:m_k] = np.linalg.lstsq(self.M_[0:m_k, 0:m_k], b, rcond=None)[0]

            v = self.theta_[0:m_k] / self.dF_scale_[0:m_k]
            self.current_u_ = G - np.dot(v, self.prev_dG_[0:m_k, :])

            self.col_idx_ = (self.col_idx_ + 1) % self.mk
            self.prev_dF_[self.col_idx_, :] = -self.current_F_
            self.prev_dG_[self.col_idx_, :] = -G
        self.iter_ += 1

        return self.current_u_.copy()

    def replace(self, x):
        self.current_u_ = x.copy()

    def reset(self, x):
        self.current_u_ = x.copy()
        self.iter_ = 0
        self.col_idx_ = 0



def ADMM_bqp_unconstrained(A, b, all_params=None):
    initial_params = {'std_threshold':1e-6, 'gamma_val':1.0, 'gamma_factor':0.99, \
          'initial_rho':5, 'learning_fact':1+3/100, 'rho_upper_limit':1000, 'history_size':5, 'rho_change_step':5, \
          'rel_tol':1e-5, 'stop_threshold':1e-3, 'max_iters':1e4, 'projection_lp': 2, 'pcg_tol':1e-6, 'pcg_maxiters':1e3}
    
    if all_params==None:
        all_params = initial_params
    else:
        for k in initial_params.keys():
            if k not in all_params.keys():
                all_params[k] = initial_params[k]
    
    n = b.size
    stop_threshold = all_params['stop_threshold']
    std_threshold = all_params['std_threshold']
    max_iters = all_params['max_iters']
    initial_rho = all_params['initial_rho']
    rho_change_step = all_params['rho_change_step']
    gamma_val = all_params['gamma_val']
    learning_fact = all_params['learning_fact']
    history_size = all_params['history_size']
    projection_lp = all_params['projection_lp']
    gamma_factor = all_params['gamma_factor']
    pcg_tol = all_params['pcg_tol']
    pcg_maxiters = all_params['pcg_maxiters']
    rho_upper_limit=all_params['rho_upper_limit']
    
    
    x_sol = all_params['x0']
    N_z= 2*len(x_sol[:])
    y1 = x_sol
    y2 = x_sol
    z1 = np.zeros_like(y1)
    z2 = np.zeros_like(y2)
    x_0 = x_sol.copy()
    z1_0 = z1.copy()
    z2_0 = z2.copy()

    rho1 = initial_rho
    rho2 = rho1
    obj_list = []
    std_obj = 1
    
    
    # initiate the binary solution
    prev_idx = x_sol
    best_sol = prev_idx
    best_bin_obj = compute_cost(best_sol,A,b)
    
    
    time_elapsed = 0
    total_time = 0
    ff = open('residual-no.txt', 'w');
    for iter in range(int(max_iters)):
        t1 = time.time()
        
        # update y1
        y1 = project_box(x_sol+z1/rho1)
    
        # update y2
        y2 = project_shifted_Lp_ball(x_sol+z2/rho2, projection_lp)


        #update x
        x_pre = x_sol.copy()
        row = np.array(range(n))
        colum = np.array(range(n))
        data = (rho1+rho2)*np.ones(n)
        sparse_matrix = csc_matrix((data, (row, colum)), shape=(n, n))
        x_sol,cg_flag = linalg.cg(2*A+sparse_matrix, rho1*y1+rho2*y2-(b+z1+z2), y1, tol=pcg_tol, maxiter=pcg_maxiters)
        x_sol = x_sol.reshape(-1,1)

        
        # update z1 and z2
        z1 = z1+gamma_val*rho1*(x_sol-y1)
        z2 = z2+gamma_val*rho2*(x_sol-y2)

        r_com=np.linalg.norm(x_sol-y1)**2+np.linalg.norm(x_sol-y2)**2+2*np.linalg.norm(x_pre-x_sol)**2
        r_s = (np.linalg.norm(x_sol-y1)+np.linalg.norm(x_sol-y2))/np.linalg.norm(x_sol)
        t2 = time.time()
        total_time = total_time + (t2 - t1)
        #ff.write(str(total_time) + '        ' + str(r_s) + '\n');
        ff.write(str(total_time) + '        ' + str(np.sqrt(rho1*r_com/N_z)) + '\n');
        
        # increase rho1 and rho2
        if np.mod(iter+2, rho_change_step)==0:
            rho1 = min(learning_fact*rho1,rho_upper_limit)
            rho2 = min(learning_fact*rho2,rho_upper_limit)

            # evaluate this iteration
        # temp1 = (np.linalg.norm(x_sol - y1)) / max(np.linalg.norm(x_sol), 2.2204e-16)
        # temp2 = (np.linalg.norm(x_sol - y2)) / max(np.linalg.norm(x_sol), 2.2204e-16)
        # if max(temp1, temp2) <= stop_threshold:
        #     print('iter: %d, stop_threshold: %.6f' % (iter, max(temp1, temp2)))
        #     break
        
        # evaluate this iteration

        #print('iter: %d, combine residual: %.6f' % (iter, r_com))

        # obj_list.append(compute_cost(x_sol,A,b))
        # if len(obj_list)>=history_size:
        #     std_obj = compute_std_obj(obj_list, history_size)
        #     if std_obj<=std_threshold:
        #         print('iter: %d, std_threshold: %.6f' %(iter, std_obj))
        #         break
        # temp1= (np.linalg.norm(x_sol-y1)) / max(np.linalg.norm(x_sol), 2.2204e-16)
        # temp2= (np.linalg.norm(x_sol-y2)) / max(np.linalg.norm(x_sol), 2.2204e-16)
        # if max(temp1,temp2)<=stop_threshold:
        #     print('iter: %d, stop_threshold: %.6f' %(iter, max(temp1,temp2)))
        #     break   
    
        cur_idx = x_sol>=0.5
        prev_idx = cur_idx
        cur_obj = compute_cost(prev_idx,A,b)
        
        # maintain best binary solution so far; in case the cost function oscillates
        if best_bin_obj >= cur_obj:
            best_bin_obj = cur_obj
            best_sol = x_sol
    ff.close()
    x_sol_ADMM = x_sol.copy()
    print("Anderson Acceleration:")
    for andersonmk in range(6,7):
        rho1 = initial_rho
        rho2 = initial_rho
        x_sol = x_0.copy()
        z1 = z1_0.copy()
        z2 = z2_0.copy()
        x_d = x_sol.copy()
        z1_d = z1.copy()
        z2_d = z2.copy()
        res_pre = 9e20
        reset = False
        sca_z = rho1
        size1 = x_sol.flatten().shape[0]
        size2 = z1.flatten().shape[0]
        sign = 0
        total_time = 0
        acc1 = Anderson(np.concatenate((x_sol.flatten(), z1.flatten()/sca_z, z2.flatten()/sca_z), axis=0), andersonmk)
        ff = open('residual-' + str(andersonmk) + '.txt', 'w');
        r_s_pre = 1e20
        for iter in range(int(max_iters)):
            t1 = time.time()

            # update y1
            y1 = project_box(x_sol + z1 / rho1)

            # update y2
            y2 = project_shifted_Lp_ball(x_sol + z2 / rho2, projection_lp)

            # update x
            x_pre = x_sol.copy()
            row = np.array(range(n))
            colum = np.array(range(n))
            data = (rho1 + rho2) * np.ones(n)
            sparse_matrix = csc_matrix((data, (row, colum)), shape=(n, n))
            x_sol, cg_flag = linalg.cg(2 * A + sparse_matrix, rho1 * y1 + rho2 * y2 - (b + z1 + z2), y1, tol=pcg_tol,
                                       maxiter=pcg_maxiters)
            x_sol = x_sol.reshape(-1, 1)

            # update z1 and z2
            z1 = z1 + gamma_val * rho1 * (x_sol - y1)
            z2 = z2 + gamma_val * rho2 * (x_sol - y2)
            res = np.linalg.norm(x_sol - y1) ** 2 + np.linalg.norm(x_sol - y2) ** 2 + 2*np.linalg.norm(x_pre - x_sol) ** 2
            r_s = (np.linalg.norm(x_sol - y1) + np.linalg.norm(x_sol - y2)) / np.linalg.norm(x_sol)
            if res< res_pre or reset==True:
                res_pre = res
                r_s_pre = r_s
                reset = False
                x_d = x_sol.copy()
                z1_d = z1.copy()
                z2_d = z2.copy()
                tt = acc1.compute(np.concatenate((x_sol.flatten(), z1.flatten()/sca_z, z2.flatten()/sca_z), axis=0))
                x_sol = tt[:size1].reshape(x_sol.shape)
                z1 = sca_z*tt[size1:size1+size2].reshape(z1.shape)
                z2 = sca_z*tt[size1+size2:].reshape(z2.shape)
                # if np.mod(iter + 2, rho_change_step) == 0:
                #     rho1 = min(learning_fact * rho1, rho_upper_limit)
                #     rho2 = min(learning_fact * rho2, rho_upper_limit)
            else:
                sign = sign + 1
                reset=True
                x_sol = x_d.copy()
                z1 = z1_d.copy()
                z2 = z2_d.copy()
                acc1.reset(np.concatenate((x_sol.flatten(), z1.flatten()/sca_z, z2.flatten()/sca_z), axis=0))


            if np.mod(iter + 2, rho_change_step) == 0:
                sca_z = rho1
                rho1 = min(learning_fact * rho1, rho_upper_limit)
                rho2 = min(learning_fact * rho2, rho_upper_limit)
                if rho1<rho_upper_limit:
                    acc1.reset(np.concatenate((x_sol.flatten(), z1.flatten()/sca_z, z2.flatten()/sca_z), axis=0))


            t2 = time.time()
            total_time = total_time + (t2 - t1)
            #ff.write(str(total_time) + '        ' + str(r_s_pre) + '\n');
            ff.write(str(total_time) + '        ' + str(np.sqrt(rho1*res_pre/N_z)) + '\n');
            # obj_list.append(compute_cost(x_sol, A, b))
            # if len(obj_list) >= history_size:
            #     std_obj = compute_std_obj(obj_list, history_size)
            #     if std_obj <= std_threshold:
            #         print('iter: %d, std_threshold: %.6f' % (iter, std_obj))
            #         break
            # temp1= (np.linalg.norm(x_sol-y1)) / max(np.linalg.norm(x_sol), 2.2204e-16)
            # temp2= (np.linalg.norm(x_sol-y2)) / max(np.linalg.norm(x_sol), 2.2204e-16)
            # if max(temp1,temp2)<=stop_threshold:
            #     print('iter: %d, stop_threshold: %.6f' %(iter, max(temp1,temp2)))
            #     break

            cur_idx = x_sol >= 0.5
            prev_idx = cur_idx
            cur_obj = compute_cost(prev_idx, A, b)

            # maintain best binary solution so far; in case the cost function oscillates
            if best_bin_obj >= cur_obj:
                best_bin_obj = cur_obj
                best_sol = x_sol
        ff.close()
    x_sol_AAADMM = x_sol.copy()
    print("Anderson Acceleration with Douglas-Ranchford splitting:")
    for andersonmk in range(6, 7):
        rho1 = initial_rho
        rho2 = initial_rho
        x_sol = x_0.copy()
        z1 = z1_0.copy()
        z2 = z2_0.copy()
        x_d = x_sol.copy()
        z1_d = z1.copy()
        z2_d = z2.copy()
        s = [x_sol.copy(),x_sol.copy()]
        u = s.copy()
        unew = u.copy()
        v = s.copy()
        temp = v.copy()
        s_d = s.copy()
        res_pre = 9e20
        r_s_pre = 1e20
        reset = False
        sca_z = rho1
        size1 = x_sol.flatten().shape[0]
        sign = 0
        total_time = 0
        acc1 = Anderson(np.concatenate((s[0].flatten(), s[1].flatten()), axis=0),andersonmk)
        ff = open('dr-' + str(andersonmk) + '.txt', 'w');
        for iter in range(int(max_iters)):
            t1 = time.time()

            # update x

            row = np.array(range(n))
            colum = np.array(range(n))
            data = (rho1 + rho2) * np.ones(n)
            sparse_matrix = csc_matrix((data, (row, colum)), shape=(n, n))
            x_sol, cg_flag = linalg.cg(2 * A + sparse_matrix, rho1 * (s[0]) + rho2 * (s[1]) - (b), y1, tol=pcg_tol,
                                       maxiter=pcg_maxiters)
            x_sol = x_sol.reshape(-1, 1)

            u[0] = x_sol.copy()
            u[1] = x_sol.copy()


            temp[0] = 2 * u[0] - s[0]
            temp[1] = 2 * u[1] - s[1]
            # update y1
            y1 = project_box(temp[0])

            # update y2
            y2 = project_shifted_Lp_ball(temp[1], projection_lp)

            v[0] = y1.copy()
            v[1] = y2.copy()
            s[0] += v[0] - u[0]
            s[1] += v[1] - u[1]
            res = np.linalg.norm(v[0] - u[0]) ** 2+np.linalg.norm(v[1] - u[1]) ** 2

            t2 = time.time()
            total_time += t2 - t1
            row = np.array(range(n))
            colum = np.array(range(n))
            data = (rho1 + rho2) * np.ones(n)
            sparse_matrix = csc_matrix((data, (row, colum)), shape=(n, n))
            x_sol, cg_flag = linalg.cg(2 * A + sparse_matrix, rho1 * (s[0]) + rho2 * (s[1]) - (b), y1, tol=pcg_tol,
                                       maxiter=pcg_maxiters)
            x_sol = x_sol.reshape(-1, 1)

            unew[0] = x_sol.copy()
            unew[1] = x_sol.copy()
            r_com=np.linalg.norm(v[0] - unew[0]) ** 2+np.linalg.norm(v[1] - unew[1]) ** 2 + np.linalg.norm(unew[0]-u[0])**2+np.linalg.norm(unew[1]-u[1])**2
            r_s = (np.linalg.norm(v[0] - u[0]) + np.linalg.norm(v[1] - u[1]))/np.linalg.norm(x_sol)
            t1= time.time()
            if res < res_pre or reset == True:
                res_pre = res
                r_s_pre = r_s
                r_com_pre = r_com
                reset = False
                s_d = s.copy()
                tt = acc1.compute(np.concatenate((s[0].flatten(), s[1].flatten()), axis=0))
                s[0] = tt[:size1].reshape(x_sol.shape)
                s[1] = tt[size1:].reshape(x_sol.shape)

                # if np.mod(iter + 2, rho_change_step) == 0:
                #     rho1 = min(learning_fact * rho1, rho_upper_limit)
                #     rho2 = min(learning_fact * rho2, rho_upper_limit)

            else:
                sign = sign + 1
                reset = True
                s = s_d.copy()
                acc1.reset(np.concatenate((s[0].flatten(), s[1].flatten()), axis=0))

            if np.mod(iter + 2, rho_change_step) == 0:
                if rho1<rho_upper_limit:
                    acc1.reset(np.concatenate((s[0].flatten(), s[1].flatten()), axis=0))
                rho1 = min(learning_fact * rho1, rho_upper_limit)
                rho2 = min(learning_fact * rho2, rho_upper_limit)

            t2 = time.time()
            total_time = total_time + (t2 - t1)
            #ff.write(str(total_time) + '        ' + str(r_s_pre) + '\n');
            ff.write(str(total_time) + '        ' + str(np.sqrt(rho1*r_com_pre/N_z)) + '\n');
            # obj_list.append(compute_cost(x_sol, A, b))
            # if len(obj_list) >= history_size:
            #     std_obj = compute_std_obj(obj_list, history_size)
            #     if std_obj <= std_threshold:
            #         print('iter: %d, std_threshold: %.6f' % (iter, std_obj))
            #         break
            # temp1= (np.linalg.norm(x_sol-y1)) / max(np.linalg.norm(x_sol), 2.2204e-16)
            # temp2= (np.linalg.norm(x_sol-y2)) / max(np.linalg.norm(x_sol), 2.2204e-16)
            # if max(temp1,temp2)<=stop_threshold:
            #     print('iter: %d, stop_threshold: %.6f' %(iter, max(temp1,temp2)))
            #     break

            cur_idx = x_sol >= 0.5
            prev_idx = cur_idx
            cur_obj = compute_cost(prev_idx, A, b)

            # maintain best binary solution so far; in case the cost function oscillates
            if best_bin_obj >= cur_obj:
                best_bin_obj = cur_obj
                best_sol = x_sol
        ff.close()
        x_sol_AADR = x_sol.copy()


    return best_sol,x_sol_ADMM,x_sol_AAADMM,x_sol_AADR,y1,y2,time_elapsed
    
    

"""
min_x x^T A x+b^Tx such that x is {0,1}^n; Cx=d
"""
def ADMM_bqp_linear_eq(A,b,C,d, all_params=None):

    initial_params = {'stop_threshold':1e-4,'std_threshold':1e-6,'gamma_val':1.6,'gamma_factor':0.95, 'rho_change_step':5, \
    'max_iters':1e3,'initial_rho':25,'history_size':3,'learning_fact':1+1/100,'x0':None,'pcg_tol':1e-4, 'pcg_maxiters':1e3,'rel_tol':5*1e-5, 'projection_lp':2}
    
    if all_params==None:
        all_params = initial_params
    else:
        for k in initial_params.keys():
            if k not in all_params.keys():
                all_params[k] = initial_params[k]
                            
    n = b.size
    stop_threshold = all_params['stop_threshold']
    std_threshold = all_params['std_threshold']
    max_iters = all_params['max_iters']
    initial_rho = all_params['initial_rho']
    rho_change_step = all_params['rho_change_step']
    gamma_val = all_params['gamma_val']
    learning_fact = all_params['learning_fact']
    history_size = all_params['history_size']
    projection_lp = all_params['projection_lp']
    gamma_factor = all_params['gamma_factor']
    pcg_tol = all_params['pcg_tol']
    pcg_maxiters = all_params['pcg_maxiters']


    # initialization
    x_sol = all_params['x0']
    y1 = x_sol    
    y2 = x_sol    
    z1 = np.zeros_like(y1)
    z2 = np.zeros_like(y2)
    z3 = np.zeros_like(d)
    rho1 = initial_rho
    rho2 = rho1
    rho3 = rho1
    obj_list = []
    Csq = C.transpose()@C  


    # initiate the binary solution
    prev_idx = x_sol
    best_sol = prev_idx
    best_bin_obj = compute_cost(best_sol,A,b)


    time_elapsed = 0
    for iter in range(int(max_iters)):
        t1 = time.time()
        
        # update y1: project onto box
        y1 = project_box(x_sol+z1/rho1)
    
    
        # update y2: project onto lp sphere
        y2 = project_shifted_Lp_ball(x_sol+z2/rho2, projection_lp)
        
     
        # update x: this is an exact solution to the subproblem
        # + solve a PD linear system, using pre-conditioned conjugate gradient algorithm  
        row = np.array(range(n))
        colum = np.array(range(n))
        data = (rho1+rho2)*np.ones(n)
        sparse_matrix = csc_matrix((data, (row, colum)), shape=(n, n))
        x_sol,cg_flag = linalg.cg(2*A+rho3*Csq+sparse_matrix, -(b+z1+z2+C.transpose()@z3)+rho1*y1+rho2*y2+rho3*C.transpose()@d, y1,  pcg_tol, pcg_maxiters)
        x_sol = x_sol.reshape(-1,1)
        
     
        # update z1 and z2 and z3
        z1 = z1+gamma_val*rho1*(x_sol-y1)
        z2 = z2+gamma_val*rho2*(x_sol-y2)
        z3 = z3+gamma_val*rho3*(C@x_sol-d)

        t2 = time.time()
        time_elapsed = time_elapsed+ (t2-t1)
        
        
        # increase rhos and update gamma is needed
        if np.mod(iter+1, rho_change_step)==0:
            rho1 = learning_fact*rho1
            rho2 = learning_fact*rho2
            rho3 = learning_fact*rho3
            gamma_val = max(gamma_val*gamma_factor,1)      
        
        
        # evaluate this iteration
        temp1= (np.linalg.norm(x_sol-y1)) / max(np.linalg.norm(x_sol), 2.2204e-16)
        temp2= (np.linalg.norm(x_sol-y2)) / max(np.linalg.norm(x_sol), 2.2204e-16)
        if max(temp1,temp2)<=stop_threshold:
            print('iter: %d, stop_threshold: %.6f' %(iter, max(temp1,temp2)))
            break

        obj_list.append(compute_cost(x_sol,A,b))
        if len(obj_list)>=history_size:
            std_obj = compute_std_obj(obj_list, history_size)
            if std_obj <= std_threshold:
                print('iter: %d, std_threshold: %.6f' %(iter, std_obj))
                break
        
        # maintain best binary solution so far; in case the cost function oscillates
        cur_idx = x_sol>=0.5
        prev_idx = cur_idx
        cur_obj = compute_cost(prev_idx,A,b)
    
        if best_bin_obj > cur_obj and max(temp1,temp2)>=1e-3:
            best_bin_obj = cur_obj
            best_sol = x_sol
            

    return best_sol,x_sol, y1,y2,time_elapsed



"""
This function implement lpbox ADMM to solve the following problem
min_x x'*A*x+b'*x such that x is {0,1}^n; Ex<=f
"""
def ADMM_bqp_linear_ineq(A,b,E,f,all_params):

    initial_params = {'stop_threshold':1e-4,'std_threshold':1e-6,'gamma_val':1.6,'gamma_factor':0.95, 'rho_change_step':5, \
        'max_iters':1e3, 'initial_rho':25, 'history_size':3, 'learning_fact':1+1/100, 'x0':[], 'pcg_tol':1e-4, 'pcg_maxiters':1e3, 'rel_tol':5e-5, 'projection_lp':2}

    if all_params==None:
        all_params = initial_params
    else:
        for k in initial_params.keys():
            if k not in all_params.keys():
                all_params[k] = initial_params[k]
                            
    n = b.size
    stop_threshold = all_params['stop_threshold']
    std_threshold = all_params['std_threshold']
    max_iters = all_params['max_iters']
    initial_rho = all_params['initial_rho']
    rho_change_step = all_params['rho_change_step']
    gamma_val = all_params['gamma_val']
    learning_fact = all_params['learning_fact']
    history_size = all_params['history_size']
    projection_lp = all_params['projection_lp']
    gamma_factor = all_params['gamma_factor']
    pcg_tol = all_params['pcg_tol']
    pcg_maxiters = all_params['pcg_maxiters']
    

    # initialization
    x_sol = all_params['x0']
    y1 = x_sol
    y2 = x_sol  
    y3 = f-E@x_sol
    z1 = np.zeros_like(y1)
    z2 = np.zeros_like(y2)
    z4 = np.zeros_like(y3)
    rho1 = initial_rho
    rho2 = rho1
    rho4 = rho1
    obj_list = []
    Esq = E.transpose()@E


    # initiate the binary solution
    prev_idx = x_sol
    best_sol = prev_idx
    best_bin_obj = compute_cost(best_sol,A,b)
    
    time_elapsed = 0;
    for iter in range(int(max_iters)):
        t1 = time.time()
        
        # update y1: project on box
        y1 = project_box(x_sol+z1/rho1)
        
        # update y2: project on circle
        y2 = project_shifted_Lp_ball(x_sol+z2/rho2, projection_lp)


        # update y3: project on non-negative quadrant
        y3 = f-E*x_sol-z4/rho4
        y3[y3<0]=0
        
     
        # update x: this is an exact solution to the subproblem
        # + solve a convex QP with linear constraints  
        
        row = np.array(range(n))
        colum = np.array(range(n))
        data = (rho1+rho2)*np.ones(n)
        sparse_matrix = csc_matrix((data, (row, colum)), shape=(n, n))
        x_sol,cg_flag = linalg.cg(2*A + rho4*Esq + sparse_matrix, \
                                  -(b+z1+z2+E.transpose()@z4)+rho1*y1+rho2*y2+rho4*E.transpose()@(f-y3), \
                                  x_sol, pcg_tol,pcg_maxiters)    
        x_sol = x_sol.reshape(-1,1)
        
    
        # update z1 and z2 and z4
        z1 = z1+gamma_val*rho1*(x_sol-y1)
        z2 = z2+gamma_val*rho2*(x_sol-y2)
        z4 = z4+gamma_val*rho4*(E*x_sol+y3-f)

        
        t2 = time.time()
        time_elapsed = time_elapsed+ (t2-t1)
        
        
        # increase rhos and update gamma is needed
        if np.mod(iter+2, rho_change_step)==0:
            rho1 = learning_fact*rho1
            rho2 = learning_fact*rho2
            rho4 = learning_fact*rho4
            gamma_val = max(gamma_val*gamma_factor,1)
       
        
        # evaluate this iteration
        temp1= (np.linalg.norm(x_sol-y1)) / max(np.linalg.norm(x_sol), 2.2204e-16)
        temp2= (np.linalg.norm(x_sol-y2)) / max(np.linalg.norm(x_sol), 2.2204e-16)
        if max(temp1,temp2)<=stop_threshold:
            print('iter: %d, stop_threshold: %.6f' %(iter, max(temp1,temp2)))
            break

        obj_list.append(compute_cost(x_sol,A,b))
        if len(obj_list)>=history_size:
            std_obj = compute_std_obj(obj_list, history_size)
            if std_obj <= std_threshold:
                print('iter: %d, std_threshold: %.6f' %(iter, std_obj))
                break
        
        # maintain best binary solution so far; in case the cost function oscillates
        cur_idx = x_sol>=0.5
        prev_idx = cur_idx
        cur_obj = compute_cost(prev_idx,A,b)
    
        if best_bin_obj > cur_obj and np.linalg.norm(E@x_sol-f)<=((1e-3)*np.sqrt(n)):
            best_bin_obj = cur_obj
            best_sol = x_sol
        
        
    return best_sol,x_sol,y1,y2,time_elapsed


"""
This function implement lpbox ADMM to solve the following problem
min_x x'*A*x+b'*x such that x is {0,1}^n; Cx=d, Ex<=f
"""
def ADMM_bqp_linear_eq_and_uneq(A,b,C,d,E,f, all_params=None):

    initial_params = {'stop_threshold':1e-4,'std_threshold':1e-6,'gamma_val':1.6,'gamma_factor':0.95, 'rho_change_step':5, \
    'max_iters':1e3,'initial_rho':25,'history_size':3,'learning_fact':1+1/100,'x0':None,'pcg_tol':1e-4, 'pcg_maxiters':1e3,'rel_tol':5*1e-5, 'projection_lp':2}
    
    if all_params==None:
        all_params = initial_params
    else:
        for k in initial_params.keys():
            if k not in all_params.keys():
                all_params[k] = initial_params[k]
                            
    n = b.size
    stop_threshold = all_params['stop_threshold']
    std_threshold = all_params['std_threshold']
    max_iters = all_params['max_iters']
    initial_rho = all_params['initial_rho']
    rho_change_step = all_params['rho_change_step']
    gamma_val = all_params['gamma_val']
    learning_fact = all_params['learning_fact']
    history_size = all_params['history_size']
    projection_lp = all_params['projection_lp']
    gamma_factor = all_params['gamma_factor']
    pcg_tol = all_params['pcg_tol']
    pcg_maxiters = all_params['pcg_maxiters']


    # initialization
    x_sol = all_params['x0']
    y1 = x_sol    
    y2 = x_sol
    y3 = f-E@x_sol   
    
    z1 = np.zeros_like(y1)
    z2 = np.zeros_like(y2)
    z3 = np.zeros_like(d)
    z4 = np.zeros_like(y3)
    
    rho1 = initial_rho
    rho2 = rho1
    rho3 = rho1
    rho4 = rho1
    
    obj_list = []
    Csq = C.transpose()@C  
    Esq = E.transpose()@E


    # initiate the binary solution
    prev_idx = x_sol
    best_sol = prev_idx
    best_bin_obj = compute_cost(best_sol,A,b)


    time_elapsed = 0
    for iter in range(int(max_iters)):
        t1 = time.time()
        
        # update y1: project onto box
        y1 = project_box(x_sol+z1/rho1)
    
    
        # update y2: project onto lp sphere
        y2 = project_shifted_Lp_ball(x_sol+z2/rho2, projection_lp)
        
        
        # update y3: project on non-negative quadrant
        y3 = f-E@x_sol-z4/rho4
        y3[y3<0]=0
        
        # update x: this is an exact solution to the subproblem
        # + solve a PD linear system, using pre-conditioned conjugate gradient algorithm  
        row = np.array(range(n))
        colum = np.array(range(n))
        data = (rho1+rho2)*np.ones(n)
        sparse_matrix = csc_matrix((data, (row, colum)), shape=(n, n))
        

        x_sol,cg_flag = linalg.cg(2*A+rho3*Csq+rho4*Esq+sparse_matrix, -(b+z1+z2+C.transpose()@z3+E.transpose()@z4) \
                                                                       +rho1*y1+rho2*y2+rho3*C.transpose()@d+rho4*E.transpose()@(f-y3), \
                                                                         y1,  pcg_tol, pcg_maxiters)
        x_sol = x_sol.reshape(-1,1)
        
     
        # update z1 and z2 and z3 and z4
        z1 = z1+gamma_val*rho1*(x_sol-y1)
        z2 = z2+gamma_val*rho2*(x_sol-y2)
        z3 = z3+gamma_val*rho3*(C@x_sol-d)
        z4 = z4+gamma_val*rho4*(E@x_sol+y3-f)

        t2 = time.time()
        time_elapsed = time_elapsed+ (t2-t1)
        
        
        # increase rhos and update gamma is needed
        if np.mod(iter+1, rho_change_step)==0:
            rho1 = learning_fact*rho1
            rho2 = learning_fact*rho2
            rho3 = learning_fact*rho3
            rho4 = learning_fact*rho4
            gamma_val = max(gamma_val*gamma_factor,1)      
        
        
        # evaluate this iteration
        temp1= (np.linalg.norm(x_sol-y1)) / max(np.linalg.norm(x_sol), 2.2204e-16)
        temp2= (np.linalg.norm(x_sol-y2)) / max(np.linalg.norm(x_sol), 2.2204e-16)
        if max(temp1,temp2)<=stop_threshold:
            print('iter: %d, stop_threshold: %.6f' %(iter, max(temp1,temp2)))
            break

        obj_list.append(compute_cost(x_sol,A,b))
        if len(obj_list)>=history_size:
            std_obj = compute_std_obj(obj_list, history_size)
            if std_obj <= std_threshold:
                print('iter: %d, std_threshold: %.6f' %(iter, std_obj))
                break
        
        # maintain best binary solution so far; in case the cost function oscillates
        cur_idx = x_sol>=0.5
        prev_idx = cur_idx
        cur_obj = compute_cost(prev_idx,A,b)
    
        if best_bin_obj > cur_obj and max(temp1,temp2)>=1e-3:
            best_bin_obj = cur_obj
            best_sol = x_sol
            

    return best_sol,x_sol,y1,y2,time_elapsed



def project_box(x):
    xp = x
    xp[x>1]=1
    xp[x<0]=0

    return xp


def project_shifted_Lp_ball(x, p):
    shift_vec = 1/2*np.ones((x.size, 1))
    shift_x = x-shift_vec
    normp_shift = np.linalg.norm(shift_x, p)
    n = x.size
    xp = (n**(1/p)) * shift_x / (2*normp_shift) + shift_vec

    return xp

    
def compute_cost(x,A,b):
    c = x.transpose()@A@x + b.transpose()@x
    
    return c


def compute_std_obj(obj_list, history_size):
    
    std_obj = np.std(obj_list[-1-history_size:])
    
    # normalize 
    std_obj = std_obj/abs(obj_list[-1])

        
    return std_obj[0][0]

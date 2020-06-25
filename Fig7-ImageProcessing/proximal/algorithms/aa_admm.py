# The MIT License (MIT)

# Original work Copyright (c) 2016 comp-imaging

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



# BSD 3-Clause License

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
# SOFTWARE.

import numpy as np
from proximal.lin_ops import CompGraph, scale, vstack
from proximal.utils.timings_log import TimingsLog, TimingsEntry
import time
from .invert import get_least_squares_inverse, get_diag_quads


class Anderson:
    def __init__(self, x0, e_dim, num):
        self.mk = num
        self.dim = len(x0)
        self.e_dim= e_dim
        self.current_u_ = x0.copy()
        self.current_F_ =  np.zeros(self.e_dim)
        self.prev_dG_ = np.zeros((num, self.dim))
        self.prev_dF_ = np.zeros((num, self.e_dim))
        self.M_ = np.zeros((num, num))  # num*num array
        self.theta_ = np.zeros(num)
        self.dF_scale_ = np.zeros(num)
        self.dG_scale_ = np.zeros(num)
        self.iter_ = 0
        self.col_idx_ = 0

    def compute(self, g):
        G = g.copy()
        self.current_F_ = G[:self.e_dim] - self.current_u_[:self.e_dim]

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


def partition(prox_fns, try_diagonalize=True):
    """Divide the proxable functions into sets Psi and Omega.
    """
    # Merge these quadratic functions with the v update.
    quad_funcs = []
    # All lin ops must be gram diagonal for the least squares problem
    # to be diagonal.
    func_opts = {True: [], False: []}
    for freq in [True, False]:
        if all([fn.lin_op.is_gram_diag(freq) for fn in prox_fns]):
            func_opts[freq] = get_diag_quads(prox_fns, freq)
    # Quad funcs is the max cardinality set.
    if len(func_opts[True]) >= len(func_opts[False]):
        quad_funcs = func_opts[True]
    else:
        quad_funcs = func_opts[False]
    psi_fns = [fn for fn in prox_fns if fn not in quad_funcs]
    return psi_fns, quad_funcs


def solve(psi_fns, omega_fns, rho=1.0,
          max_iters=1000, eps_abs=1e-10, eps_rel=1e-3, x0=None,
          lin_solver="cg", lin_solver_options=None,
          try_diagonalize=True, try_fast_norm=False,
          scaled=True,
          metric=None, convlog=None, verbose=0):
    # C=np.array([[1,0],[0,0]]);
    # b=np.array([2,0]);
    # print(np.linalg.lstsq(C,b,rcond=None)[0])
    prox_fns = psi_fns + omega_fns
    stacked_ops = vstack([fn.lin_op for fn in psi_fns])
    K = CompGraph(stacked_ops)
    # Rescale so (rho/2)||x - b||^2_2
    rescaling = np.sqrt(2. / rho)
    quad_ops = []
    const_terms = []
    for fn in omega_fns:
        fn = fn.absorb_params()
        quad_ops.append(scale(rescaling * fn.beta, fn.lin_op))
        const_terms.append(fn.b.flatten() * rescaling)
    # Check for fast inverse.
    op_list = [func.lin_op for func in psi_fns] + quad_ops
    stacked_ops = vstack(op_list)

    # Get optimize inverse (tries spatial and frequency diagonalization)
    v_update = get_least_squares_inverse(op_list, None, try_diagonalize, verbose)

    # Initialize everything to zero.
    input_size = K.input_size
    output_size = K.output_size
    v = np.zeros(input_size)
    z = np.zeros(output_size)
    u = np.zeros(output_size)

    print(output_size)

    # Initialize
    if x0 is not None:
        v[:] = np.reshape(x0, input_size)
        K.forward(v, z)

    # Buffers.
    v0 = v.copy()
    z0 = z.copy()
    u0 = u.copy()
    N_z = len(z[:])
    Kv = np.zeros(output_size)
    KTu = np.zeros(input_size)
    s = np.zeros(input_size)
    Kv_pre = Kv.copy()
    # Log for prox ops.
    prox_log = TimingsLog(prox_fns)
    # Time iterations.
    iter_timing = TimingsEntry("ADMM iteration")
    # Convergence log for initial iterate
    if convlog is not None:
        K.update_vars(v)
        objval = sum([func.value for func in prox_fns])
        convlog.record_objective(objval)
        convlog.record_timing(0.0)

    # --------------------------------------------------------------------------------------------------
    print("Anderson Acceleration:")
    for andersonmk in range(6,7):
        v = v0.copy()
        u = u0.copy()
        v_d = v.copy()
        u_d = u.copy()
        res_pre = 9e20
        total_energy = []
        total_time = []
        Combine_res = []
        reset = False
        sca_z = 1
        size = v.flatten().shape[0]
        total_size=(u.flatten()).shape[0]+size
        print(size)
        sign = 0
        curr_time = 0
        AA_compute_time = 0
        acc1 = Anderson(np.concatenate((v.flatten(), sca_z * u.flatten()), axis=0),total_size, andersonmk)
        for i in range(max_iters):
            t1 = time.time()
            if convlog is not None:
                convlog.tic()

            K.forward(v, Kv)
            # Update z.
            Kv_pre = Kv.copy()
            K.forward(v, Kv)
            Kv_u = Kv + u
            offset = 0
            for fn in psi_fns:
                tmp = np.hstack([z - u] + const_terms)
                v = v_update.solve(tmp, x_init=v, lin_solver=lin_solver, options=lin_solver_options)
                K.forward(v, Kv)
                Kv_u = Kv + u
                slc = slice(offset, offset + fn.lin_op.size, None)
                Kv_u_slc = np.reshape(Kv_u[slc], fn.lin_op.shape)
                # Apply and time prox.
                z_pre = z.copy()
                prox_log[fn].tic()
                z[slc] = fn.prox(rho, Kv_u_slc, i).flatten()
                prox_log[fn].toc()
                offset += fn.lin_op.size
            # Update u.
            r = Kv - z
            u += r
            K.adjoint(u, KTu)
            # print(np.linalg.norm(u))

            # Check convergence.

            # K.adjoint(rho * (z - z_prev), s)
            s = z - z_pre
            res = np.linalg.norm(r) ** 2 + np.linalg.norm(s) ** 2
            # K.adjoint((z-z_prev),s)
            # eps_pri = np.sqrt(output_size) * eps_abs + eps_rel * \
            #           max([np.linalg.norm(Kv), np.linalg.norm(z)])
            # eps_dual = np.sqrt(input_size) * eps_abs + eps_rel * np.linalg.norm(KTu) / rho

            t3=time.time()
            if res < res_pre or reset == True:
                v_d = v.copy()
                u_d = u.copy()
                res_pre = res
                reset = False            
                tt = acc1.compute(np.concatenate((v.flatten(), sca_z * u.flatten()), axis=0))
                v = tt[0:size].reshape(v.shape)
                u = tt[size:].reshape(u.shape) / sca_z
            else:
                sign = sign + 1
                v = v_d.copy()
                u = u_d.copy()
                reset = True
                acc1.reset(np.concatenate((v.flatten(), sca_z * u.flatten()), axis=0))
            t4=time.time()
            AA_compute_time += t4-t3

            t2 = time.time()
            curr_time += t2 - t1

            # Convergence log
            if convlog is not None:
                convlog.toc()
                K.update_vars(v)
                objval = sum([fn.value for fn in prox_fns])
                convlog.record_objective(objval)

            # Show progess
            if verbose > 0:
                # Evaluate objective only if required (expensive !)
                objstr = ''
                if verbose == 2:
                    K.update_vars(v)
                    objstr = ", obj_val = %02.03e" % sum([fn.value for fn in prox_fns])

                # Evaluate metric potentially
                metstr = '' if metric is None else ", {}".format(metric.message(v))
                # print("iter %d: ||r||_2 = %.3f, eps_pri = %.3f, ||s||_2 = %.3f, eps_dual = %.3f%s%s" % (
                #     i, np.linalg.norm(r), eps_pri, np.linalg.norm(s), eps_dual, objstr, metstr))
                print("iter %d: combine residual = %.8f" % (
                    i, res))
            
            Combine_res.append(np.sqrt(rho*res_pre/N_z))
            total_time.append(curr_time)
            # Exit if converged.
            if (res) < eps_abs:
                break
        print("current time: %.6f, AA compute: %.6f, sign: %d"%(curr_time, AA_compute_time, sign))

        hm_src_path = 'residual-'+str(andersonmk)+'.txt'
        iter_num = []
        iter_num.append(len(total_time))
        iter_num.append(len(Combine_res))
        with open(hm_src_path, 'w') as f:
            for i in range(0, min(iter_num)):
                f.write('%f\t%.20f\n' % (total_time[i], Combine_res[i]))
        f.close()
    print("Anderson Acceleration with Douglas-Rachford splitting:")
    for andersonmk in range(6, 7):
        v = v0.copy()
        u = u0.copy()
        K.forward(v, Kv)
        v_d = v.copy()
        u_d = u.copy()
        d_s = z0.copy()
        d_u = d_s.copy()
        d_s_d = d_s.copy()
        d_v = d_s.copy()
        d_unew= d_u.copy()
        res_pre = 9e20
        r_com = 0
        r_com_pre = r_com
        total_energy = []
        total_time = []
        Combine_res = []
        reset = False
        size = v.flatten().shape[0]
        sign = 0
        curr_time = 0
        acc1 = Anderson(d_s.flatten(), size, andersonmk)
        for i in range(max_iters):
            t1 = time.time()
            if convlog is not None:
                convlog.tic()
            # K.forward(v, Kv)
                # Update v.
            Kv_u = d_s.copy()
            offset = 0
            for fn in psi_fns:
                slc = slice(offset, offset + fn.lin_op.size, None)
                Kv_u_slc = np.reshape(Kv_u[slc], fn.lin_op.shape)
                # Apply and time prox.
                prox_log[fn].tic()
                z[slc] = fn.prox(rho, Kv_u_slc, i).flatten()
                prox_log[fn].toc()
                offset += fn.lin_op.size

            d_u=z.copy()
            temp = 2*d_u-d_s
            tmp = np.hstack([temp] + const_terms)
            v = v_update.solve(tmp, x_init=v, lin_solver=lin_solver, options=lin_solver_options)
            K.forward(v,d_v)
            # z_prev = z.copy()
            # Update z.
            # Update d_s
            r = d_v - d_u
            d_s += r
            res = np.linalg.norm(r) ** 2
            t2 = time.time()
            curr_time += t2 - t1
            # print(np.linalg.norm(u))
            Kv_u = d_s.copy()
            offset = 0
            for fn in psi_fns:
                slc = slice(offset, offset + fn.lin_op.size, None)
                Kv_u_slc = np.reshape(Kv_u[slc], fn.lin_op.shape)
                # Apply and time prox.
                prox_log[fn].tic()
                z[slc] = fn.prox(rho, Kv_u_slc, i).flatten()
                prox_log[fn].toc()
                offset += fn.lin_op.size
            d_unew=z.copy()
            # Check convergence.
            # K.adjoint(rho * (z - z_prev),

            r_com= np.linalg.norm(d_unew-d_v)**2 + np.linalg.norm(d_unew-d_u)**2
            # K.adjoint((z-z_prev),s)
            # eps_pri = np.sqrt(output_size) * eps_abs + eps_rel * \
            #           max([np.linalg.norm(Kv), np.linalg.norm(z)])
            # eps_dual = np.sqrt(input_size) * eps_abs + eps_rel * np.linalg.norm(KTu) / rho

            # Convergence log
            if convlog is not None:
                convlog.toc()
                K.update_vars(v)
                objval = sum([fn.value for fn in prox_fns])
                convlog.record_objective(objval)
            t1 = time.time()
            if res < res_pre or reset == True:
                d_s_d = d_s.copy()
                res_pre = res
                r_com_pre = r_com
                reset = False
                tt = acc1.compute(d_s.flatten())
                d_s = tt.reshape(d_s.shape)
            else:
                sign = sign + 1
                d_s = d_s_d.copy()
                reset = True
                acc1.reset(d_s.flatten())
            # Show progess
            if verbose > 0:
                # Evaluate objective only if required (expensive !)
                objstr = ''
                if verbose == 2:
                    K.update_vars(v)
                    objstr = ", obj_val = %02.03e" % sum([fn.value for fn in prox_fns])

                # Evaluate metric potentially
                metstr = '' if metric is None else ", {}".format(metric.message(v))
                # print("iter %d: ||r||_2 = %.3f, eps_pri = %.3f, ||s||_2 = %.3f, eps_dual = %.3f%s%s" % (
                #     i, np.linalg.norm(r), eps_pri, np.linalg.norm(s), eps_dual, objstr, metstr))
                print("iter %d: combine residual = %.8f" % (
                    i, r_com))
            t2 = time.time()
            curr_time += t2 - t1
            Combine_res.append(np.sqrt(rho*r_com_pre/N_z))
            total_time.append(curr_time)
            # Exit if converged.
            # if (res) < eps_abs:
            #     break
        hm_src_path = 'dr-' + str(andersonmk) + '.txt'
        iter_num = []
        iter_num.append(len(total_time))
        iter_num.append(len(Combine_res))
        with open(hm_src_path, 'w') as f:
            for i in range(0, min(iter_num)):
                f.write('%f\t%.20f\n' % (total_time[i], Combine_res[i]))
        f.close()
    # Print out timings info.
    if verbose > 0:
        print(iter_timing)
        print("prox funcs:")
        print(prox_log)
        print("K forward ops:")
        print(K.forward_log)
        print("K adjoint ops:")
        print(K.adjoint_log)

    # Assign values to variables.
    K.update_vars(v)
    # Return optimal value.
    # return sum([fn.value for fn in prox_fns])
    return total_time, Combine_res

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

import numpy as np
from proximal.lin_ops import CompGraph, scale, vstack
from proximal.utils.timings_log import TimingsLog, TimingsEntry
import time
from .invert import get_least_squares_inverse, get_diag_quads




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
          max_iters=1000, eps_abs=1e-1, eps_rel=1e-3, x0=None,
          lin_solver="cg", lin_solver_options=None,
          try_diagonalize=True, try_fast_norm=False,
          scaled=True,
          metric=None, convlog=None, verbose=0):
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
    N_z = len(z[:])
    print(input_size)
    print(output_size)

    # Initialize
    if x0 is not None:
        v[:] = np.reshape(x0, input_size)
        K.forward(v, z)

    # Buffers.

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
    res_pre = 9e20
    res = 0

    curr_time = 0
    total_time = []
    Combine_res = []
    # ------------------------------------------------------------------------------------
    for i in range(max_iters):
        # iter_timing.tic()
        t1 = time.time()
        if convlog is not None:
            convlog.tic()
        K.forward(v, Kv)
        Kv_pre = Kv.copy()
        # z_prev = z.copy()
        # Update z.
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
        # Update v.

        # Check convergence.
        r = Kv - z
        # Update u.
        u += r
        K.adjoint(u, KTu)
     
        # K.adjoint(rho * (z - z_prev), s)
        s = z - z_pre
        t2 = time.time()
        curr_time += t2 - t1

        res = np.linalg.norm(r) ** 2 + np.linalg.norm(s) ** 2

        # K.adjoint((z-z_prev),s)
        # eps_pri = np.sqrt(output_size) * eps_abs + eps_rel * \
        #   max([np.linalg.norm(Kv), np.linalg.norm(z)])
        # eps_dual = np.sqrt(input_size) * eps_abs + eps_rel * np.linalg.norm(KTu) / rho

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

        #curr_time = curr_time + iter_timing.toc()

        Combine_res.append(np.sqrt(rho*res/N_z))
        total_time.append(curr_time)
        # Exit if converged.
        if (res) < eps_abs:
            break

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

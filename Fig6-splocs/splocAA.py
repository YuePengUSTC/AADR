# The MIT License (MIT)

# Original work Copyright (c) 2013 Thomas Neumann

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# BSD 3-Clause License

# Modified work Copyright (c) 2020 Yuxin Yao

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


import argparse
import numpy as np
from scipy.linalg import svd, norm, cho_factor, cho_solve
import h5py
from numpy import linalg as la
import anderson
import time

from geodesic import GeodesicDistanceComputation


def project_weight(x):
    x = np.maximum(0., x)
    max_x = x.max()
    if max_x == 0:
        return x
    else:
        return x / max_x

def prox_l1l2(Lambda, x, beta):
    xlen = np.sqrt((x**2).sum(axis=-1))
    with np.errstate(divide='ignore'):
        shrinkage = np.maximum(0.0, 1 - beta * Lambda / xlen)
    return (x * shrinkage[...,np.newaxis])

def compute_support_map(idx, geodesics, min_dist, max_dist):
    phi = geodesics(idx)
    return (np.clip(phi, min_dist, max_dist) - min_dist) / (max_dist - min_dist)

def print_res(file, times, energys, residuals):
    with open(file, 'w') as f:
        for i in range(np.shape(times)[0]):
            f.write('%.16f\t%.16f\n' % (times[i], residuals[i]))


def main(input_animation_file, output_sploc_file, output_animation_file, mid, anderson_m, outiter, out_res, input_rho):
    rest_shape = "first" # which frame to use as rest-shape ("first" or "average")
    K = 50 # number of components
    smooth_min_dist = 0.1 # minimum geodesic distance for support map, d_min_in paper
    smooth_max_dist = 0.7 # maximum geodesic distance for support map, d_max in paper
    num_iters_max = 1 # number of iterations to run
    sparsity_lambda = 2. # sparsity parameter, lambda in the paper
    input_mid = mid

    rho = input_rho # penalty parameter for ADMM
    num_admm_iterations = 3000 # number of ADMM iterations


    # preprocessing: (external script)
        # rigidly align sequence
        # normalize into -0.5 ... 0.5 box

    with h5py.File(input_animation_file, 'r') as f:
        verts = f['verts'].value.astype(np.float)
        tris = f['tris'].value

    F, N, _ = verts.shape

    if rest_shape == "first":
        Xmean = verts[0]
    elif rest_shape == "average":
        Xmean = np.mean(verts, axis=0)

    # prepare geodesic distance computation on the restpose mesh
    compute_geodesic_distance = GeodesicDistanceComputation(Xmean, tris)

    # form animation matrix, subtract mean and normalize
    # notice that in contrast to the paper, X is an array of shape (F, N, 3) here
    X = verts - Xmean[np.newaxis]
    pre_scale_factor = 1 / np.std(X)
    X *= pre_scale_factor
    R = X.copy() # residual

    # find initial components explaining the residual
    C = []
    W = []
    for k in range(K):
        # find the vertex explaining the most variance across the residual animation
        magnitude = (R**2).sum(axis=2)
        idx = np.argmax(magnitude.sum(axis=0))
        # find linear component explaining the motion of this vertex
        U, s, Vt = la.svd(R[:,idx,:].reshape(R.shape[0], -1).T, full_matrices=False)
        wk = s[0] * Vt[0,:] # weights
        # invert weight according to their projection onto the constraint set 
        # this fixes problems with negative weights and non-negativity constraints
        wk_proj = project_weight(wk)
        wk_proj_negative = project_weight(-wk)
        wk = wk_proj \
                if norm(wk_proj) > norm(wk_proj_negative) \
                else wk_proj_negative
        s = 1 - compute_support_map(idx, compute_geodesic_distance, smooth_min_dist, smooth_max_dist)
        # solve for optimal component inside support map
        ck = (np.tensordot(wk, R, (0, 0)) * s[:,np.newaxis])\
                / np.inner(wk, wk)
        C.append(ck)
        W.append(wk)
        # update residual
        R -= np.outer(wk, ck).reshape(R.shape)
    C = np.array(C)
    W = np.array(W).T

    # prepare auxiluary variables
    Lambda = np.empty((K, N))
    U = np.zeros((K, N, 3))

    total_begintime = time.time()
    # main global optimization
    for it in range(num_iters_max):
        if it == outiter:
            mid = input_mid
        else:
            mid = 0

        # update weights
        Rflat = R.reshape(F, N*3) # flattened residual
        for k in range(C.shape[0]): # for each component
            Ck = C[k].ravel()
            Ck_norm = np.inner(Ck, Ck)
            if Ck_norm <= 1.e-8:
                # the component seems to be zero everywhere, so set it's activation to 0 also
                W[:,k] = 0
                continue # prevent divide by zero
            # block coordinate descent update
            Rflat += np.outer(W[:, k], Ck)
            opt = np.dot(Rflat, Ck) / Ck_norm
            W[:,k] = project_weight(opt)
            Rflat -= np.outer(W[:,k], Ck)
        # update spatially varying regularization strength
        for k in range(K):
            ck = C[k]
            # find vertex with biggest displacement in component and compute support map around it
            idx = (ck**2).sum(axis=1).argmax()
            support_map = compute_support_map(idx, compute_geodesic_distance, 
                                              smooth_min_dist, smooth_max_dist)
            # update L1 regularization strength according to this support map
            Lambda[k] = sparsity_lambda * support_map
        # update components
        Z = C.copy() # dual variable
        # prefactor linear solve in ADMM
        G = np.dot(W.T, W)
        c = np.dot(W.T, X.reshape(X.shape[0], -1))
        solve_prefactored = cho_factor(G + rho * np.eye(G.shape[0]))

        times = []
        residuals = []
        energys = []
        # ADMM iterations
        size1 = Z.flatten().shape[0]
        print("Z size = %d"%(size1))
        if mid == 0:
            ignore_t = 0
            total_t = 0
            for admm_it in range(num_admm_iterations):
                begin_time = time.time()
                Z_prev = Z.copy()
                C = cho_solve(solve_prefactored, c + rho * (Z - U).reshape(c.shape)).reshape(C.shape)
                Z = prox_l1l2(Lambda, C + U, 1. / rho)
                U = U + C - Z                

                # ignore_st = time.time()
                # # f(x) + g(z)
                # R = X - np.tensordot(W, C, (1, 0))  # residual
                # sparsity = np.sum(Lambda * np.sqrt((C ** 2).sum(axis=2)))
                # e = (R ** 2).sum() + sparsity
                # ignore_et = time.time()
                # ignore_t = ignore_t + ignore_et - ignore_st

                # energys.append(e)
                run_time = time.time()
                total_t = total_t + run_time - begin_time
                times.append(total_t)
                # times.append(run_time - begin_time - ignore_t)
                res = la.norm(C - Z)**2 + la.norm(Z - Z_prev)**2
                residuals.append(np.sqrt(rho*res/size1))
                if res < 1e-10:
                    break

        # AA-ADMM iterations        
        if mid == 1:
            # acc parameters        
            Z_default = Z.copy()
            U_default = U.copy()
            acc = anderson.Anderson(np.concatenate((Z.flatten(), U.flatten()), axis=0), anderson_m)
            reset = True
            r_prev = 1e10
            admm_it = 0
            ignore_t = 0
            reset_count = 0
            total_t = 0
            AA_t = 0
            while admm_it < num_admm_iterations:
                begin_time = time.time()
                Z_prev = Z.copy()
                C = cho_solve(solve_prefactored, c + rho * (Z - U).reshape(c.shape)).reshape(C.shape)
                Z = prox_l1l2(Lambda, C + U, 1. / rho)
                U = U + C - Z
                res = la.norm(C - Z)**2 + la.norm(Z-Z_prev)**2
                # res = np.square(la.norm(C-Z)) + np.square(la.norm(Z-Z_prev))

                # ignore_st = time.time()
                # # f(x) + g(z)
                # R = X - np.tensordot(W, C, (1, 0))  # residual
                # sparsity = np.sum(Lambda * np.sqrt((C ** 2).sum(axis=2)))
                # e = (R ** 2).sum() + sparsity
                # ignore_et = time.time()
                # ignore_t = ignore_t + ignore_et - ignore_st
                if reset or res < r_prev:
                    Z_default = Z.copy()
                    U_default = U.copy()
                    r_prev = res
                    reset = False
                    acc_ZU = acc.compute(np.concatenate((Z.flatten(), U.flatten()), axis=0))
                    Z = acc_ZU[:size1].reshape(Z.shape)
                    U = acc_ZU[size1:].reshape(U.shape)

                    admm_it = admm_it + 1
                else:
                    Z = Z_default.copy()
                    U = U_default.copy()
                    reset = True
                    reset_count = reset_count + 1
                    acc.replace(np.concatenate((Z.flatten(), U.flatten()), axis=0))
                    continue

                # energys.append(e)
                run_time = time.time()
                total_t = total_t + run_time - begin_time
                times.append(total_t)
                # times.append(run_time - begin_time - ignore_t)
                residuals.append(np.sqrt(rho*r_prev/size1))
                if res < 1e-10:
                    break
            print('AA ADMM reset number:%d, total time: %.6f'%(reset_count, total_t))

        ## AA-DR primal
        if mid == 2:
            # acc parameters
            s = Z + U
            s_default = s.copy()
            acc = anderson.Anderson(s.flatten(), anderson_m)
            reset = True
            r_prev = 1e10
            admm_it = 0
            ignore_t = 0
            reset_count = 0
            total_t = 0
            while admm_it < num_admm_iterations:
                begin_time = time.time()
                Z = prox_l1l2(Lambda, s, 1. / rho)
                C = cho_solve(solve_prefactored, c + rho * (2*Z - s).reshape(c.shape)).reshape(C.shape)
                s = s + C - Z
                res = la.norm(C - Z)**2
                # res = np.square(la.norm(C - Z))

                # ignore_st = time.time()
                run_time = time.time()
                total_t = total_t + run_time - begin_time

                Z_p = prox_l1l2(Lambda, s, 1. / rho)
                r_com = la.norm(C - Z_p)**2 + la.norm(Z_p - Z)**2
                # r_com = np.square(la.norm(C - Z_p)) + np.square(la.norm(Z_p - Z))
                # f(x) + g(z)
                # R = X - np.tensordot(W, C, (1, 0))  # residual
                # sparsity = np.sum(Lambda * np.sqrt((C ** 2).sum(axis=2)))
                # e = (R ** 2).sum() + sparsity
                # ignore_et = time.time()
                # ignore_t = ignore_t + (ignore_et - ignore_st)
                begin_time = time.time()

                if reset or res < r_prev:
                    s_default = s.copy()
                    r_prev = res
                    reset = False
                    acc_s = acc.compute(s_default.flatten())
                    s = acc_s.reshape(s.shape)                   
                    admm_it = admm_it + 1
                else:
                    s = s_default.copy()
                    reset = True
                    reset_count = reset_count + 1
                    acc.replace(s_default.flatten())
                    continue

                # energys.append(e)
                run_time = time.time()
                total_t = total_t + run_time - begin_time
                times.append(total_t)
                # times.append(run_time - begin_time - ignore_t)
                residuals.append(np.sqrt(rho*r_com/size1))
                if r_com < 1e-10:
                    break
            print('DR reset number:%d, total time: %.6f'%(reset_count, total_t))

        ## AA-DR DR envelope
        # if mid == 3:
        #     # acc parameters
        #     s = Z - U
        #     s_default = s.copy()
        #     acc = anderson.Anderson(s.flatten(), anderson_m)
        #     reset = True
        #     dre_prev = 1e10
        #     admm_it = 0
        #     ignore_t = 0
        #     reset_count = 0
        #     begin_time = time.time()
        #     while admm_it < num_admm_iterations:
        #         Z = prox_l1l2(Lambda, s, 1. / rho)
        #         C = cho_solve(solve_prefactored, c + rho * (2*Z - s).reshape(c.shape)).reshape(C.shape)
        #         s = s + C - Z
        #         res = np.square(la.norm(C - Z))

        #         ignore_st = time.time()
        #         Z_p = prox_l1l2(Lambda, s, 1. / rho)
        #         r_com = res + np.square(la.norm(Z_p - Z))
        #         ignore_et = time.time()
        #         ignore_t = ignore_t + (ignore_et - ignore_st)

        #         # f(x) + g(z)
        #         R = X - np.tensordot(W, C, (1, 0))  # residual
        #         sparsity = np.sum(Lambda * np.sqrt((C ** 2).sum(axis=2)))
        #         e = (R ** 2).sum() + sparsity

        #         #DRE
        #         dre = e + rho * np.sum(np.multiply(s-Z, C-Z)) + res * rho / 2

        #         if reset or dre < dre_prev:
        #             s_default = s.copy()
        #             dre_prev = dre
        #             reset = False
        #             acc_s = acc.compute(s_default.flatten())
        #             s = acc_s.reshape(s.shape)
        #             admm_it = admm_it + 1
        #         else:
        #             s = s_default.copy()
        #             reset = True
        #             reset_count = reset_count + 1
        #             acc.replace(s_default.flatten())
        #             continue

        #         energys.append(e)
        #         run_time = time.time()
        #         times.append(run_time - begin_time - ignore_t)
        #         residuals.append(np.sqrt(rho*r_com/size1))
        #         if r_com < 1e-10:
        #             break
        #     print('DR Envolop reset number:%d'%(reset_count))

        if it == outiter:
            fname = out_res + '.txt'
            print_res(fname, times, energys, residuals)

        # set updated components to dual Z, 
        # this was also suggested in [Boyd et al.] for optimization of sparsity-inducing norms
        C = Z
        # evaluate objective function
        R = X - np.tensordot(W, C, (1, 0)) # residual
        sparsity = np.sum(Lambda * np.sqrt((C**2).sum(axis=2)))
        e = (R**2).sum() + sparsity


        # TODO convergence check
        print("iteration %03d, E=%f" % (it, e))

        if it == outiter:
            break

    # undo scaling
    C /= pre_scale_factor
    total_endtime = time.time()
    run_total_time = total_endtime - total_begintime

    # # save components
    # with h5py.File(output_sploc_file, 'w') as f:
    #     f['default'] = Xmean
    #     f['tris'] = tris
    #     for i, c in enumerate(C):
    #         f['comp%03d' % i] = c + Xmean
    #
    # # save encoded animation including the weights
    # if output_animation_file:
    #     with h5py.File(output_animation_file, 'w') as f:
    #         f['verts'] = np.tensordot(W, C, (1, 0)) + Xmean[np.newaxis]
    #         f['tris'] = tris
    #         f['weights'] = W

    return run_total_time, e


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description='Find Sparse Localized Deformation Components')
    # parser.add_argument('input_animation_file')
    # parser.add_argument('output_sploc_file')
    # parser.add_argument('-a', '--output-anim',
    #                     help='Output animation file (will also save the component weights)')
    # args = parser.parse_args()
    path = 'data/coma_data/'
    for fid in (2, 5):
        mu = 10
        # for mu in (10, 100, 1000, 10000, 100000, 1000000):
        for mid in range(0, 3):
            inputf = path + 'h5file/f' + str(fid)  + '_align.h5'
            outf = path + 'outf/nf' + str(fid) + '_mid' + str(mid) +'_mu' + str(mu) + '.h5'
            anif = path + 'anif/nf' + str(fid) + '_mid' + str(mid) +'_mu' + str(mu) + '.h5'
            outfile = path + 'res/nf' + str(fid) + '_mid' + str(mid) +'_mu' + str(mu)
            totaltime, energy = main(inputf,  outf, anif, mid, 6, 0, outfile, mu)
            print('fid%d\tm%d\t%.2f\t%.16f'% (fid, mid, totaltime, energy))
    # path = 'data/jumping/'
    # for fid in range(1, 2):
    #     mu = 10
    #     # for mu in (10, 100, 1000, 10000, 100000, 1000000):
    #     for mid in range(0, 4):
    #         inputf = path + 'h5file/f' + str(fid) + '_align.h5'
    #         outf = path + 'outf/addif' + str(fid) + '_mid' + str(mid) + '_mu' + str(mu) + '.h5'
    #         anif = path + 'anif/addif' + str(fid) + '_mid' + str(mid) + '_mu' + str(mu) + '.h5'
    #         outfile = path + 'res/addif' + str(fid) + '_mid' + str(mid) + '_mu' + str(mu)
    #         main(inputf, outf, anif, mid, 6, 0, outfile, mu)


#     GNU GENERAL PUBLIC LICENSE

#     Original work Copyright (C) 1989, 1991 Free Software Foundation, Inc., <http://fsf.org/>
#     Modified work Copyright (C) 2020,  Wenqing Ouyang

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.

from itertools import count
import logging
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from math import sqrt
import time
from .laplacian import compute_mesh_laplacian
from .orthomax import orthomax
from .prefactor import factorized
from .anderson import *

def compressed_manifold_modes(verts, tris, K, mu, init=None, scaled=False,
                              return_info=False, return_eigenvalues=False, return_D=False,
                              order=True, algorithm=None,
                              **algorithm_params):
    Q, vertex_area = compute_mesh_laplacian(verts, tris, 'cotangent',
                                            return_vertex_area=True, area_type='lumped_mass')
    if scaled:
        D = sparse.spdiags(np.sqrt(vertex_area), 0, len(verts), len(verts))
        Dinv = sparse.spdiags(1 / np.sqrt(vertex_area), 0, len(verts), len(verts))
    else:
        D = Dinv = None

    if init == 'mh':
        Phi_init = manifold_harmonics(verts, tris, K)
    elif init == 'varimax':
        Phi_init = varimax_modes(verts, tris, K)
    elif type(init) == np.ndarray:
        Phi_init = init
    else:
        Phi_init = None

    if algorithm is None:
        algorithm = solve_compressed_splitorth

    Phi, info = algorithm(
        Q, K, mu1=mu, Phi_init=Phi_init, D=D, Dinv=Dinv,
        **algorithm_params)

    l = eigenvalues_from_eigenvectors(verts, tris, Phi, Q=Q)
    if order:
        idx = np.abs(l).argsort()
        Phi = Phi[:, idx]
        l = l[idx]

    result = [Phi]
    if return_eigenvalues:
        result.append(l)
    if return_D:
        if D is not None:
            result.append(D * D)
        else:
            result.append(None)
    if return_info:
        result.append(info)
    if len(result) == 1:
        return result[0]
    else:
        return result


def manifold_harmonics(verts, tris, K, scaled=True, return_D=False, return_eigenvalues=False):
    Q, vertex_area = compute_mesh_laplacian(
        verts, tris, 'cotangent',
        return_vertex_area=True, area_type='lumped_mass'
    )
    if scaled:
        D = sparse.spdiags(vertex_area, 0, len(verts), len(verts))
    else:
        D = sparse.spdiags(np.ones_like(vertex_area), 0, len(verts), len(verts))

    try:
        lambda_dense, Phi_dense = eigsh(-Q, M=D, k=K, sigma=0)
    except RuntimeError as e:
        if e.message == 'Factor is exactly singular':
            logging.warn("factor is singular, trying some regularization and cholmod")
            chol_solve = factorized(-Q + sparse.eye(Q.shape[0]) * 1.e-9)
            OPinv = sparse.linalg.LinearOperator(Q.shape, matvec=chol_solve)
            lambda_dense, Phi_dense = eigsh(-Q, M=D, k=K, sigma=0, OPinv=OPinv)
        else:
            raise e

    result = [Phi_dense]
    if return_D:
        result.append(D)
    if return_eigenvalues:
        result.append(lambda_dense)
    if len(result) == 1:
        return result[0]
    else:
        return result


def varimax_modes(verts, tris, K, scaled=False):
    Phi_dense = manifold_harmonics(verts, tris, K, scaled=scaled)
    return orthomax(Phi_dense, maxiter=100, gamma=1.0)


def eigenvalues_from_eigenvectors(verts, tris, eigs, Q=None):
    if Q is None:
        Q = compute_mesh_laplacian(verts, tris, 'cotangent', return_vertex_area=False)
    return (eigs * (-Q * eigs)).sum(axis=0)


def shrink(u, delta):
    """ proximal operator of L1 """
    return np.multiply(np.sign(u), np.maximum(0, np.abs(u) - delta))


def solve_compressed_splitorth(L, K,mu1, Phi_init=None, maxiter=1000, callback=None,
                               D=None, Dinv=None,
                               rho=0.5, auto_adjust_penalty=False,
                               rho_adjust=2.0, rho_adjust_sensitivity=5.,
                               tol_abs=1.e-6, tol_rel=1.e-6,
                               verbose=100, check_interval=10):
    #L = L/3
    N = L.shape[0]  # alias
    mu = (mu1 / float(N))
    #mu = mu1
    val,vec=sparse.linalg.eigsh(L,K);
    for i in range(vec.shape[0]):
        vec[i, :] = vec[i, :][::-1]
    Phi=vec.copy();
    # initial variables
    # if Phi_init is None:
    #     Phi = np.linalg.qr(np.random.uniform(-1, 1, (L.shape[0], K)))[0]
    # else:
    #     Phi = Phi_init
    Phi0 = Phi.copy();
    P = Phi.copy()
    Q = Phi.copy()
    U = np.zeros((2, Phi.shape[0], Phi.shape[1]))
    U0 = U.copy()
    # status variables for Phi-solve
    Hsolve = None
    refactorize = False
    maxiter = 10000
    # iteration state
    iters = count() if maxiter is None else range(maxiter)
    converged = False

    info = {}
    m, n = P.shape;
    if D is None:
        D = sparse.identity(N)
        Dinv = sparse.identity(N)
    if Hsolve is None or refactorize:
        A = (-L - L.T + sparse.eye(L.shape[0], L.shape[0]) * rho).tocsc()
        if Hsolve is None or (refactorize and not hasattr(Hsolve, 'cholesky_inplace')):
            Hsolve = factorized(A)
        elif refactorize:
            # when cholmod is available, use faster in-place refactorization
            Hsolve.cholesky_inplace(A)
            refactorize = False

    # debug = np.random.rand(Phi.shape[0])
    # print(debug.T@(-L-L.T)@debug)
#-------------------------------------------------------------------------------------------------------------
    #no anderson
    size = P.flatten().shape[0]
    admm_iters = count() if maxiter is None else range(maxiter+10000)
    ff=open('residual-no.txt','w');
    t1=time.time()
    for i in admm_iters:
        # update Phi
        _PA = D * (0.5 * (P - U[0] + Q - U[1]))
        _PP = np.dot(_PA.T, _PA)
        S, sigma, St = np.linalg.svd(_PP)
        Sigma_inv = np.diag(np.sqrt(1.0 / sigma))
        Phi = Dinv * (_PA.dot(S.dot(Sigma_inv.dot(St))))
        # update Q
        Q_old = Q.copy()
        Q = shrink(Phi + U[1], mu / rho)
        #  P_old = P.copy();
        # update P
        rhs = np.asfortranarray(rho * (Phi + U[0]))
        P_old = P.copy()
        P = Hsolve(rhs)


        # update residuals
        U[0] += Phi - P
        U[1] += Phi - Q
   #     rnorm = (((Phi - Q) ** 2 + (Phi - P) ** 2).sum()) + ((2 * (P - P_old) ** 2).sum());
        rnorm = (((Phi - Q) ** 2 + (Phi - P) ** 2).sum()) + (((Q - Q_old) ** 2+ (P_old - P) ** 2).sum())
        t2=time.time()
        ff.write(str(t2-t1)+'        '+str(sqrt(rho*rnorm/2/N/K))+'\n')
        # if i % check_interval == 0:
        #     # compute primal and dual residual
        #     #rnorm=(((Phi - Q) ** 2 + (Phi - P) ** 2).sum()) + ((2 * (P - P_old) ** 2).sum());
        #     # convergence checks
        #     # TODO check if convergence check is correct for 3-function ADMM
        #     print(i, end=' ')
        #       #  print("o %0.8f" % (sparsity + eig), end=' ')  # %0.4f %0.4f" % (sparsity + eig, sparsity, eig),
        #     print(" | r ", (rnorm))
        #     if (rnorm)<tol_abs:
        #         break

    ff.close()
    print((abs(Phi)>1e-6).sum()/K)

    for andersoni in range(6,7):
        ff = open('residual-'+str(andersoni)+'.txt', 'w');
        sign = 0
        P = Phi0.copy()
        Q = P.copy()
        U = U0.copy()
        acc1 = Anderson(np.concatenate((P.flatten(order='C'),Q.flatten(order='C'), U.flatten(order='C')),axis=0), andersoni)
        iters = count() if maxiter is None else range(maxiter)
        # P_d = P.copy();
        # U_d = U.copy();
        # P_old = P.copy();
        r = 0
        r_pre = 999999;
        AA_time=0
        print("--------------------------------------------------");
        print("Anderson m="+str(andersoni)+":");
        reset = False
        total_time = 0
        # Anderson m=1
        for i in iters:
            t1 = time.time()
            # update Phi
            # t1 = time.time()
            # update Phi
            _PA = D * (0.5 * (P - U[0] + Q - U[1]))
            _PP = np.dot(_PA.T, _PA)
            S, sigma, St = np.linalg.svd(_PP)
            Sigma_inv = np.diag(np.sqrt(1.0 / sigma))
            Phi = Dinv * (_PA.dot(S.dot(Sigma_inv.dot(St))))
            # update Q
            Q_old = Q.copy()
            Q = shrink(Phi + U[1], mu / rho)
            #  P_old = P.copy();
            # update P
            rhs = np.asfortranarray(rho * (Phi + U[0]))
            P_old = P.copy()
            P = Hsolve(rhs)

            # update residuals
            U[0] += Phi - P
            U[1] += Phi - Q
            r = (((Phi - Q) ** 2 + (Phi - P) ** 2).sum()) + (((Q - Q_old) ** 2+ (P_old - P) ** 2).sum());
            t3=time.time()
            if r < r_pre or reset == True:
                # compute primal and dual residual
                P_d = P.copy()
                Q_d = Q.copy()
                U_d = U.copy()
                r_pre = r
                reset = False
                # acc1.replace(P.flatten(order='C'), U.flatten(order='C'));
                # update Phi
                tt = acc1.compute(np.concatenate((P.flatten(order='C'),Q.flatten(order='C'), U.flatten(order='C')), axis=0))
                P = tt[0:size].reshape(Phi.shape, order='C')
                Q = tt[size:2*size].reshape(Q.shape, order='C')
                U = tt[2*size:].reshape(U.shape, order='C')
            # r = (((Phi - Q) ** 2 + (Phi - P) ** 2).sum())

            else:
                sign += 1
                P = P_d.copy()
                Q = Q_d.copy()
                U = U_d.copy()
                reset = True
                acc1.reset(np.concatenate((P.flatten(order='C'),Q.flatten(order='C'), U.flatten(order='C')), axis=0))
            #    acc1 = Anderson(np.concatenate((P.flatten(order='C'), U.flatten(order='C')), axis=0), andersoni);
            #   if np.linalg.norm(acc1.gg[-1]-U_d.flatten(order='C'))>1e-12:
            #      print('Wrong');
            #   if i % check_interval==0:
            #          priznt(i, end=' ')
            #       #  print("o %0.8f" % (sparsity + eig), end=' ')  # %0.4f %0.4f" % (sparsity + eig, sparsity, eig),
            #          print(" | r ", (r))
            #          if (r) < tol_abs:
            #               break
            # update P
            t2 = time.time()
            AA_time+=t2-t3
            total_time += t2 - t1
            ff.write(str(total_time) + '        ' + str(sqrt(rho*r_pre/N/K/2)) + '\n');
        print(sign);
        print("time for Anderson Accleration:",(AA_time/total_time))
        ff.close();
#------------------------------------------------------------------------------------------------------------------------


#----------------------test the descent of DRE
    for drm in range(6,7):
        Phi = Phi0.copy();
        P = Phi0.copy();
        Q = Phi0.copy();
        U = U0.copy()
        s = U.copy()
        s[0] = Phi0.copy()
        s[1] = Phi0.copy()
        acc1 = Anderson(s.flatten(order='C'), drm)
        v = s.copy()
        u = s.copy()
        u_new = s.copy()
        r_pre = 9999999;
        ff = open('dr-'+str(drm)+'.txt', 'w')
        total_time= 0
        sign=0
        dr_iters = count() if maxiter is None else range(maxiter+10000)
        for i in dr_iters:
            # u=prox(s)
            t1 = time.time()
            Q = shrink(s[1], mu / rho)
            rhs = np.asfortranarray(rho * s[0])
            P = Hsolve(rhs)
            u[0] = P.copy()
            u[1] = Q.copy()
            temp = 2*u - s
            _PA = D * (0.5*(temp[0]+temp[1]))
            _PP = np.dot(_PA.T, _PA)
            S, sigma, St = np.linalg.svd(_PP)
            Sigma_inv = np.diag(np.sqrt(1.0 / sigma))
            Phi = Dinv * (_PA.dot(S.dot(Sigma_inv.dot(St))))
            v[0] = Phi.copy()
            v[1] = Phi.copy()
            # v=prox(2u-s)
            s += v - u
            r = ((v - u) ** 2).sum()
            t2=time.time()
            total_time += t2 - t1
            Q = shrink(s[1], mu / rho)
            rhs = np.asfortranarray(rho * s[0])
            P = Hsolve(rhs)
            u_new[0] = P.copy()
            u_new[1] = Q.copy()

            r_com = r + ((u_new - u) ** 2).sum()
            t1 = time.time()
            if r < r_pre or reset == True:
                # compute primal and dual residual
                s_d = s.copy()
                r_pre = r;
                reset = False;
                r_com_pre=r_com
                tt = acc1.compute(s.flatten(order='C'))
                s[0] = tt[0:size].reshape(P.shape, order='C')
                s[1] = tt[size:].reshape(P.shape, order='C')
            # r = (((Phi - Q) ** 2 + (Phi - P) ** 2).sum())
            else:
                sign += 1
                s = s_d.copy()
                reset = True
                acc1.reset(s.flatten(order='C'))
            #    acc1 = Anderson(np.concatenate((P.flatten(order='C'), U.flatten(order='C')), axis=0), andersoni);
            #   if np.linalg.norm(acc1.gg[-1]-U_d.flatten(order='C'))>1e-12:
            #      print('Wrong');
            #   if i % check_interval==0:
            #          print(i, end=' ')
            #       #  print("o %0.8f" % (sparsity + eig), end=' ')  # %0.4f %0.4f" % (sparsity + eig, sparsity, eig),
            #          print(" | r ", (r))
            #          if (r) < tol_abs:
            #               break
            # update P
            t2 = time.time()
            total_time += t2 - t1
            ff.write(str(total_time) + '        ' + str(sqrt(r_com_pre/2/K/N)) + '\n')
        ff.close()
        print(sign)

#----------------------000---------------------------------------------------------------------------------------------------------

    # ----------------------000---------------------------------------------------------------------------------------------------------
    info['num_iters'] = i
    info['r_primal'] = rnorm
    info['r_dual'] = 0
    info['Q'] = Q
    info['P'] = P
    info['Phi'] = Phi
    return Phi, info


def solve_compressed_osher(L, K, mu1=10., Phi_init=None, maxiter=None, callback=None,
                           D=None, Dinv=None,
                           r=0.5, lambda_=1.,
                           tol_abs=1.e-8, tol_rel=1.e-6,
                           verbose=100):
    N = L.shape[0]  # alias
    mu = mu1 / float(N)

    # initial variables
    if Phi_init is None:
        Phi = np.linalg.qr(np.random.uniform(-1, 1, (L.shape[0], K)))[0]
    else:
        Phi = Phi_init
    P = Q = Phi
    b = np.zeros_like(Phi)
    B = np.zeros_like(Phi)

    # status variables for Phi-solve
    Hsolve = None
    refactorize = False

    # iteration state
    iters = count() if maxiter is None else range(maxiter)
    converged = False

    info = {}

    if D is None:
        D = sparse.identity(N)
        Dinv = sparse.identity(N)

    for i in iters:
        # update Phi
        if Hsolve is None or refactorize:
            A = (-L - L.T + sparse.eye(L.shape[0], L.shape[0]) * (lambda_ + r)).tocsc()
            Hsolve = factorized(A)
        rhs = np.asfortranarray(r * (P - B) + lambda_ * (Q - b))
        Phi = Hsolve(rhs)

        # update Q
        Q_old = Q
        Q = shrink(Phi + b, mu / lambda_)

        # update P
        _PA = D * (Phi + B)
        _PP = np.dot(_PA.T, _PA)
        U, sigma, St = np.linalg.svd(_PP)
        Sigma_inv = np.diag(np.sqrt(1.0 / sigma))
        P_old = P
        P = (Dinv * (_PA.dot(U.dot(Sigma_inv.dot(St))))) / r

        # update residuals
        b += Phi - Q
        B += Phi - P

        # compute primal and dual residual
        snorm1 = np.linalg.norm(lambda_ * (Q - Q_old))
        rnorm1 = np.linalg.norm(Phi - Q)
        snorm2 = np.linalg.norm(r * (P - P_old))
        rnorm2 = np.linalg.norm(Phi - P)
        snorm = np.sqrt(snorm1 ** 2 + snorm2 ** 2)
        rnorm = np.sqrt(rnorm1 ** 2 + rnorm2 ** 2)

        if callback is not None:
            try:
                callback(L, mu, Phi, P, Q, r_primal=rnorm, r_dual=snorm)
            except StopIteration:
                converged = True

        # convergence checks
        eps_pri = np.sqrt(Phi.shape[1]) * tol_abs + tol_rel * max(map(np.linalg.norm, [Phi, Q, P]))
        eps_dual = np.sqrt(Phi.shape[1]) * tol_abs + tol_rel * max(np.linalg.norm(r * B), np.linalg.norm(lambda_ * b))
        if rnorm < eps_pri and snorm < eps_dual or converged:
            if verbose:
                print("converged!")
            converged = True
        if verbose and (i % verbose == 0 or converged or (maxiter is not None and i == maxiter - 1)):
            sparsity = np.sum(mu * np.abs(Phi))
            eig = -(Phi * (L * Phi)).sum()
            gap1 = np.linalg.norm(Q - Phi)
            gap2 = np.linalg.norm(P - Phi)
            # ortho = np.linalg.norm(Phi.T.dot((D.T * D) * Phi) - np.eye(Phi.shape[1]))
            print(i),
            print("o %0.8f" % (sparsity + eig)),  # %0.4f %0.4f" % (sparsity + eig, sparsity, eig),
            print(" | r [%.8f %.8f %.8f] s [%.8f]" % (gap1, gap2, rnorm, snorm))
        if converged:
            break
    info['num_iters'] = i
    info['r_primal'] = np.linalg.norm(Phi - Q) + np.linalg.norm(Phi - P)
    info['Q'] = Q
    info['P'] = P
    info['Phi'] = Phi
    return P, info

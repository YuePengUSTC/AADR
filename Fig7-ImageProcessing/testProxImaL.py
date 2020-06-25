#!/usr/bin/env python
import sys

sys.path.append('/usr/local/lib/python3.6/dist-packages/proximal-0.1.6-py3.6.egg/proximal/')
# sys.path.append('/home/py/Downloads/proximal')

from proximal import *
from proximal.utils.utils import *
from proximal.utils.metrics import *
from proximal.halide.halide import *
from proximal.lin_ops import *
from proximal.prox_fns import *
from proximal.algorithms import *
from scipy import misc
import numpy as np
import math
# import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image

import cv2
import imageio
from scipy import signal
from scipy import ndimage


# Generate data.
# I = scipy.misc.ascent()
# np.random.seed(1)
# b = I + 10*np.random.randn(*I.shape)
# x = Variable(I.shape)

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def main():
    img = Image.open('./data/mandril/mandril_gray.jpg')  # opens the file using Pillow - it's not an array yet
    x = np.asfortranarray(im2nparray(img))
    x = np.mean(x, axis=2)
    x = np.maximum(x, 0.0)
    I = x.copy()

    print(img.size)

    # plt.ion()
    # plt.figure()
    # imgplot = plt.imshow(x, interpolation="nearest", clim=(0.0, 1.0))
    # imgplot.set_cmap('gray')
    # plt.title('Original Image')
    # plt.show()

    # Kernel
    K = Image.open('./data/kernel_snake.png')  # opens the file using Pillow - it's not an array yet
    K = np.mean(np.asfortranarray(im2nparray(K)), axis=2)
    K = np.maximum(cv2.resize(K, (15, 15), interpolation=cv2.INTER_LINEAR), 0)
    K /= np.sum(K)

    # Generate observation
    sigma_noise = 0.01
    # b = ndimage.convolve(x, K, mode='wrap') + sigma_noise * np.random.randn(*x.shape)
    img_b = Image.open('./data/mandril/noisy.png')
    b = np.asfortranarray(im2nparray(img_b))

    lambda_tv = 1.0
    lambda_data = 400.0
    psnrval = psnr_metric(I, pad=None, decimals=4)
    diag = 0
    verbose = 0

    x = Variable(I.shape)
    y = x.value

    # Construct and solve problem.
    # nonquad_fns = [group_norm1(grad(x, dims=2), [2], alpha=lambda_tv)]  # Isotropic
    # quad_funcs = [sum_squares(conv(K, x), b=b, alpha=lambda_data)]

    quad_funcs = [sum_squares(conv(K, x, implem='numpy', dims=2), b=b, alpha=lambda_data)]
    nonquad_fns = [group_norm1(grad(x, dims=2, implem='numpy'), [2],
                               alpha=lambda_tv, implem='numpy')]

    prox_fns = nonquad_fns + quad_funcs

    options = cg_options(tol=1e-7, num_iters=10, verbose=False)

    # ================ Start AA acceleration ============== #
    x.value = y
    [total_time, Combine_res] = aa_admm.solve(nonquad_fns, quad_funcs, rho=100, max_iters=2000,
                                              eps_abs=1e-10, eps_rel=1e-10, x0=x.value,
                                              lin_solver="cg", lin_solver_options=options,
                                              try_diagonalize=True, try_fast_norm=True,
                                              scaled=True,
                                              metric=psnrval, convlog=None, verbose=verbose)

    # prob = Problem(prox_fns)#, implem='numpy', try_diagonalize=True,absorb=True, merge=True,try_split=True, try_fast_norm=True, scale=True,psi_fns=None, omega_fns=None,lin_solver="cg", solver="admm"
    # [total_time, Combine_res]=prob.solve(solver="aaadmm", lin_solver_options=options, verbose=verbose)
    # prob.solve()


    imageio.imwrite('original.png', I)
    imageio.imwrite('noisy.png', b)
    imageio.imwrite('result_AA.png', x.value * 255)

    d = psnrval.eval(x.value)#psnr(I, x.value)
    print("AA psnr = %.5f" % d)

    # Plot the original, noisy, and denoised images.
    plt.figure(figsize=(15, 8))
    plt.subplot(131)
    plt.gray()
    plt.imshow(I)
    plt.title('Original image')

    plt.subplot(132)
    plt.gray()
    plt.imshow(b)
    plt.title('Noisy image')

    plt.subplot(133)
    plt.gray()
    plt.imshow(x.value * 255)  # x.value is the optimal value of x.
    plt.title('Denoising results')
    plt.show()
main()

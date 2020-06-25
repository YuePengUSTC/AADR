#!/usr/bin/env python
import sys

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
    [total_time, Combine_res] = admm.solve(nonquad_fns, quad_funcs, rho=100, max_iters=2000,
                                           eps_abs=1e-10, eps_rel=1e-10, x0=x.value,
                                           lin_solver="cg", lin_solver_options=options,
                                           try_diagonalize=True, try_fast_norm=True,
                                           scaled=True,
                                           metric=psnrval, convlog=None, verbose=verbose)

    # prob = Problem(prox_fns)#, implem='numpy', try_diagonalize=True,absorb=True, merge=True,try_split=True, try_fast_norm=True, scale=True,psi_fns=None, omega_fns=None,lin_solver="cg", solver="admm"
    # [total_time, Combine_res]=prob.solve(solver="aaadmm", lin_solver_options=options, verbose=verbose)
    # prob.solve()
    hm_src_path = 'residual-no.txt'
    iter_num = []
    iter_num.append(len(total_time))
    iter_num.append(len(Combine_res))
    with open(hm_src_path, 'w') as f:
        for i in range(0, min(iter_num)):
            f.write('%f\t%.20f\n' % (total_time[i], Combine_res[i]))

    imageio.imwrite('result_ADMM.png', x.value * 255)

    d = psnrval.eval(x.value)#psnr(I, x.value)
    print("ADMM psnr = %.5f" % d)

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

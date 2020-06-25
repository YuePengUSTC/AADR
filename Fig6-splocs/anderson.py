# BSD 3-Clause License

# Copyright (c) 2020 Wenqing Ouyang

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
from scipy.linalg import cholesky
from scipy.linalg import svd, norm, cho_factor, cho_solve



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

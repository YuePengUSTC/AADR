% BSD 3-Clause License
% 
% Original work Copyright (c) 2015, Yu Wang, Wotao Yin, and Jinshan Zeng
% Modified work Copyright (c) 2020, Yuxin Yao
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
% * Redistributions of source code must retain the above copyright notice, this
% list of conditions and the following disclaimer.
% 
% * Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
% 
% * Neither the name of the copyright holder nor the names of its
% contributors may be used to endorse or promote products derived from
% this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function [x, intercp, FLAG, iter, betas, obj, err, eachtimes,combineres,resets, Lagrangian, timing, err_rel, err_abs] = LqLogRegAA(A, b, q, mu, beta, offset,anderson_m)
  % LqLogReg  Solve Lq logistic regression via AA-ADMM
  %
  % solves the following problem via ADMM:
  %
  %   minimize   sum( log(1 + exp(-b_i*(a_i'x + intercp)) ) + m*mu*||x||_q^q  
  %
  % where m is the number of samples
  %
  % Args:
  %   A         : feature matrix [a_1';...;a_m]
  %   b         : observation vector [b_1;...;b_m]
  %   q         : 0 < q <= 1 constant, can only be 0.5 or 1.
  %   mu        : balancing parameter
  %   beta      : initial penalty parameter in ADMM
  %   offset    : whether allow intercept to be nonzero
  %
  % Returns:
  %   x         : estimated vector 
  %   intercp   : intercept
  %   FLAG      : a string indicating the exit status
  %              'Max iteration' : Max iteration has been met.
  %              'Relative error': RELTOL has been met, i.e., 
  %                    ||x^k - z^k||_inf < RELTOL * \| [x^k;z^k] \|_inf
  %              'Absolute error': ABSTOL has been met, i.e.,
  %                    ||x^k - z^k||_inf < ABSTOL
  %              'Unbounded'     : sequence is unbounded, i.e.,
  %                    ||x^k||_inf > LARGE
  %              'beta too large': cannot find appropriate beta, i.e.,
  %                    beta > LARGE (only happens when beta is set automatically)
  %                   
  %   iter       : final iteration
  %   betas      : beta of different iterates
  %   obj        : vector of  sum( log(1 + exp(-b_i*(a_i'w^k)) ) + m*mu*||w^k||_q^q
  %   err        : vector of || x^k - z^k ||_inf for each k
  %   Lagrangian : vector of Lagrangian function for each k
  %   timing     : elapsed time of the algorithm
  %   err_rel    : the relative lower bound of different iterates
  %   err_abs    : the absolute lower bound of different iterates
  
  % other parameters
  %   x0        : initial point, default is the zero vector
  %   AUTO      : whether beta can be changed by the program
  %   MAXCOUNTS : when Lagrangian increases MAXCOUNTS times, beta = beta * SCALE
  %   SCALE     : beta = beta * SCALE when Lagrangian increases MAXCOUNTS times.
  %   RELTOL    : relative error tolerance 
  %   ABSTOL    : absolute error tolerance 
  %   MAXITER   : max iteration
  %   VERBOSE   : whether print details of the algorithm
  if (offset)
      x0    = zeros(size(A,2)+1,1);
      A     = [b,A];%FIXME: mismatch
  else
      x0    = zeros(size(A,2),1);
  end
  AUTO      = false;
  SCALE     = 1.2;
  RELTOL    = 1e-5;
  ABSTOL    = 1e-5;
  MAXCOUNTS = 100;
  MAXITER   = 500;
  VERBOSE   = true;
  % Sanity check
  assert(size(A,1) == size(b,1));
  assert(size(b,2) == 1);
  assert(q==0.5 || q==1);
  assert(beta > 0);
  assert(size(A,2) == size(x0,1));
  assert(size(x0,2) == 1);
  % Default constant
  LARGE  = 1e6;
  [m, n] = size(A);
  if AUTO
    increasecounts = 0;
  end
  C = -A;%FIXME: there is a mismatch 
  % Main body
  tic; % record the time
  % Initialize
  x       = x0;
  z       = x0;
  w       = zeros(size(x0));
  obj     = nan(MAXITER, 1);
%   err     = nan(MAXITER, 1);
%   err_rel = nan(MAXITER, 1);
%   err_abs = nan(MAXITER, 1);
  betas   = nan(MAXITER, 1);
%   lagrng  = nan(MAXITER, 1);
  eachtimes = nan(MAXITER, 1);
  combineres = nan(MAXITER, 1);
  resets = nan(MAXITER, 1);
  
  % pre-defined functions
  InfNorm    = @(x) max(abs(x));
  Lq         = @(x) sum(abs(x).^q);
  Logistic   = @(x) sum(log(exp(C * x) + 1));
  Lagrangian = @(x,z,w,beta) Lq(z(2:end))*m*mu + Logistic(x) + w' * (x - z) + beta/2 * sum((x - z).^2);
  if VERBOSE
      fprintf('%4s\t%10s\t%10s\t%10s\n', 'iter', 'time','obj', 'combined res');
  end
  
  default_xw = [x; w];
  acc = Anderson(default_xw, anderson_m, length(default_xw));
  r_prev = 1e10;
  reset = true;
  true_k = 0;
  
  while true_k <= MAXITER
    prev_x = x;
    % z update
    %------------------------------------------------
    %   minimize_z m*mu*\|z\|_q^q + beta/2 \| x - z + w/beta \|_2^2
    %------------------------------------------------
    z = x + w/beta;
    z(2:end) = Threshold(z(2:end), beta/m/mu,q);
    
     % x update
    %------------------------------------------------
    %   minimize   sum(log(exp(C * x) + 1)) + beta/2 * \| x - z + w/beta \|_2^2
    %------------------------------------------------
    [x, fx] = update_x(C, x, beta, z, w);
    
    % w update
    w = w + beta * (x - z);
    
    % Anderson acceleration
    r = (norm(x-z).^2 + norm(x-prev_x).^2);
    each_obj = Lq(z(2:end))*m*mu + fx;
    resets(true_k+1) = reset;
    if (r < r_prev || reset == true)
        default_xw = [x; w/beta];
        r_prev = r;
        reset = false;
        acc_xw = acc.compute(default_xw);
        x = acc_xw(1:length(x));
        w = acc_xw(length(x)+1:end)*beta; 
        true_k = true_k+1;
    else
        x = default_xw(1:length(x));
        w = default_xw(length(x)+1:end)*beta;
        acc.reset(default_xw);
        reset = true;   
        continue;
    end
    
    % record the values
    obj(true_k)     = each_obj;
     err(true_k)     = InfNorm(x - z);
    betas(true_k)   = beta;
    lagrng(true_k)  = Lagrangian(x, z, w, beta);
    err_rel(true_k) = RELTOL * InfNorm([x;z]);
    err_abs(true_k) = ABSTOL;
    eachtimes(true_k) = toc;
    combineres(true_k) = sqrt(r_prev*beta/length(z));
   % combineres(true_k) = r_prev;
    
    if VERBOSE 
      fprintf('%4d\t%.2f\t%10.4f\t%10.4g\n', true_k, eachtimes(true_k), obj(true_k), combineres(true_k));
    end
    % beta update
    if AUTO && true_k > 1 && lagrng(true_k) > lagrng(true_k - 1);
      increasecounts = increasecounts + 1;
    end
    if AUTO && increasecounts > MAXCOUNTS;
      increasecounts = -1;
      beta = beta * SCALE;
    end
    % stopping criteria
    if true_k == MAXITER
      FLAG = 'Max iteration';
      break;
    end
    if AUTO && beta > LARGE
      FLAG = 'beta too large';
      break;
    end
    if combineres(true_k) < 1e-9
        FLAG = 'Residuals';
        break;
    end
  end
  iter = true_k;
  timing = toc;
  if VERBOSE
    fprintf('ADMM has stopped at iter %4d because of %10s.\n',true_k,FLAG);
    fprintf('Elapsed time is %8.1e seconds .\n',timing);
  end
  if offset
      intercp = x(1);
      x = x(2:end);
  end
end
function z = HalfThres(x, beta)
  tmp = 3/2*(beta)^(-2/3);  
  z = 2/3*x.*(1+cos(2/3*pi - 2/3 * acos(1/(4*beta)*(abs(x)/3).^(-3/2))));
  z(abs(x) <= tmp) = 0;
end  
function out = Threshold(x, beta, q)
    %  solve argmin_z |z|_q^q + beta/2 | z - x |^2
    if q == 1
      out = ((x - 1/beta).*(x > 1/beta) + (x + 1/beta).*(x < -1/beta));
    elseif q == 1/2
      out = HalfThres(x,beta);
    else
      error('q can only be 1 or 0.5');
    end
end

function [x, f1x] = update_x(C, x, beta, z, w)
    % solve the x update
    %   minimize [ sum log(1 + exp(Cx)) + beta/2 * ||x - z^k + w^k/beta||^2 ]
    % via Newton's method; for a single subsystem only.
    alpha = 0.1;
    BETA  = 0.5;
    TOLERANCE = 1e-7;
    MAX_ITER = 50;
    [m n] = size(C);
    I = eye(n);
    f1 = @(x) sum(log(1 + exp(C*x)));
    f2 = @(x) beta/2*norm(x - z + w/beta).^2;
    for iter = 1:MAX_ITER
        expcx = exp(C * x);
        f1x =  sum(log(1+expcx));
        subf2 = x - z + w/beta;
        fx = f1x + beta/2*norm(subf2).^2;
        tempe = expcx./(1 + expcx);
        
        g = C'*tempe + beta*(subf2);
        diagv = repmat(tempe./(1 + expcx),1,size(C,2));
        H = C' * (diagv.* C) + beta*I;
        dx = -H\g;   % Newton step
        dfx = g'*dx; % Newton decrement
        if abs(dfx) < TOLERANCE
            break;
        end
        % backtracking
        t = 1;
        f1x = f1(x+t*dx);
        while f1x + f2(x + t*dx) > fx + alpha*t*dfx
            t = BETA*t;
            f1x = f1(x+t*dx);
        end
        x = x + t*dx;
    end
end

function SimpleDemo(seed, savep, train_set_size, beta, dimension)
rng(seed)

test_set_size  = 1000;
w = (sprandn(dimension, 1, 0.1));  % N(0,1), 10% sparse
v = randn(1);            % random intercept

noise = false;
A_train             = sparse(sprandn(train_set_size, dimension, 0.1));

totalzeror =sum(sum(A_train==0))/( size(A_train,1)*size(A_train,2))
totalnor =sum(sum(A_train~=0))/( size(A_train,1)*size(A_train,2))
if noise
    train_label = sign(A_train*w + v + 0.1*randn(train_set_size,1)); % labels with noise
else
    train_label  = sign(A_train*w + v);
end
A_test              = sprandn(test_set_size, dimension, 10/dimension);
test_label   = sign(A_test*w + v);
size(A_train)
% Solve problem
mu = 0.0001;
q = 0.5;
norm_a = 1;
anderson_m = 6;
output_path = ['res/dim' num2str(dimension) '_' num2str(train_set_size) '_bt' num2str(beta) '_' ];

% ADMM
[x, intercp, FLAG, iter, betas, obj, err, times, combineres, Lagrangian, timing, err_rel, err_abs] = LqLogReg(A_train, train_label, q, mu, beta, true);
fid = fopen([output_path savep 'admm_scales.txt'], 'w');
absx = abs(x);
norm_a = max(absx);
fprintf(fid, '%.16f\t%.16f\t%.16f\n', mean(absx), median(absx), max(absx));
fclose(fid);

%Compute normalized combined residual
norm_combineres = combineres./norm_a;
fid = fopen([output_path savep 'admm.txt'], 'w');
for i=1:length(obj)
    fprintf(fid, '%.16f\t%.16f\t%.16f\n', times(i), combineres(i), norm_combineres(i));
end
fclose(fid);


% AA-DR primal residual
[x3, intercp3, FLAG3, iter3, betas3, obj3, err3, times3, combineres3, resets3, Lagrangian3, timing3, err_rel3, err_abs3] = LqLogRegDR(A_train, train_label, q, mu, beta, true, anderson_m);
norm_combineres = combineres3./norm_a;
fid = fopen([output_path savep 'aadr' num2str(anderson_m) '.txt'], 'w');
for i=1:length(obj3)
    fprintf(fid, '%.16f\t%.16f\t%.16f\n', times3(i), combineres3(i), norm_combineres(i));
end
fclose(fid);

% AA admm
[x2, intercp2, FLAG2, iter2, betas2, obj2, err2, times2, combineres2,resets2, Lagrangian2, timing2, err_rel2, err_abs2] = LqLogRegAA(A_train, train_label, q, mu, beta, true, anderson_m); 
norm_combineres = combineres2./norm_a;
fid = fopen([output_path savep 'aaadmm' num2str(anderson_m) '.txt'], 'w');
for i=1:length(obj2)
    fprintf(fid, '%.16f\t%.16f\t%.16f\n', times2(i), combineres2(i), norm_combineres(i));
end
fclose(fid);
%    
% AA-DR envelope
[x4, intercp4, FLAG4, iter4, betas4, obj4, err4, times4, combineres4,resets4, allDREs, Lagrangian4, timing4, err_rel4, err_abs4] = LqLogRegDRE(A_train, train_label, q, mu, beta, true, anderson_m);
 norm_combineres = combineres4./norm_a;
 fid = fopen([output_path savep 'aadre' num2str(anderson_m) '.txt'], 'w');
for i=1:length(obj4)
    fprintf(fid, '%.16f\t%.16f\t%.16f\n', times4(i), combineres4(i), norm_combineres(i));
end
fclose(fid);

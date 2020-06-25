#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.core.pylabtools import figsize

font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
labels = ['', '', '', '']
colors = []
colors.append([0 / 255, 113 / 255, 188 / 255])  # 1
colors.append([216 / 255, 82 / 255, 24 / 255])  # 2
colors.append([236 / 255, 176 / 255, 31 / 255])  # 3
colors.append([125 / 255, 46 / 255, 141 / 255])  # 4
colors.append([118 / 255, 171 / 255, 47 / 255])  # 5
colors.append([76 / 255, 189 / 255, 237 / 255])  # 6
colors.append([255 / 255, 128 / 255, 0 / 255])  # 7


def plot_errmore(data, is_iter, id, is_dr, min_err):
    len1 = np.shape(data)[0]
    if is_iter:
        x = np.linspace(0, len1, len1)
    else:
        x = data[:, 0] - data[0, 0]
    # y = data[:, 2]  # /data[0, 1]
    y = data[:, 2]  #/ data[0, 2]
    if id == 0:
        label1 = 'ADMM'
    else:
        if is_dr:
            label1 = 'DR m=' + str(id)
        else:
            label1 = 'AA m=' + str(id)
    l1, = plt.semilogy(x, y, label=label1, color=colors[id], linewidth=2.5)
    # l1, = plt.plot(x, y, label=label1, color=colors[id], linewidth=2)
    max_t = max(x)
    return (l1, max_t)


def plot_err3(data, is_iter, id, is_dr, cid, min_err, ytype, resetype):
    len1 = np.shape(data)[0]
    if is_iter:
        x = np.linspace(0, len1, len1)
    else:
        x = data[:, 0] - data[0, 0]
    if ytype == 'e':
        y = data[:, 1] - min_err
    else:
        y = data[:, 2]
    if id == 0:
        label1 = 'ADMM'
    else:
        if is_dr:
            label1 = 'ours' + resetype
        else:
            label1 = 'AA ADMM' + resetype
    if ytype == 'r':
        l1, = plt.semilogy(x, y, label=label1, color=colors[cid], linewidth=2.5)
    else:
        l1, = plt.semilogy(x, y, label=label1, color=colors[cid], linewidth=2.5)
    max_t = max(x)
    return (l1, max_t)



def plot_reset(data, is_iter):
    len1 = np.shape(data)[0]
    if is_iter:
        x = np.linspace(0, len1, len1)
    else:
        x = data[:, 0] - data[0, 0]
    y = data[:, 1] / data[0, 1]
    reset = data[:, 2]
    nx = []
    ny = []
    for i in range(1, len(reset)):
        if reset[i] > reset[i - 1]:
            nx.append(x[i])
            ny.append(y[i])
    plt.scatter(nx, ny, color='blue', alpha=0.6, s=20)
    # max_t = max(x)
    # return (l1, max_t)


## diff m
if 0:
    for fid in range(1, 11):
        path = "D:/project/ADMMAA/BasisPursuit/res/"
        # os.system("mkdir D:\\project\\ADMMAA\\data\\no_noise_fig3\\file" + str(fid))
        savepath = "D:/project/ADMMAA/BasisPursuit/fig/" #'D:/project/ADMMAA/data/diffpm/p0.5/fig/2efile' + str(fid) + '_'
        # for is_iter in (1, 0):
        is_iter = 1
        outer = 0
        for is_iter in range(0, 2):
            for nmid in range(1, 2):

                # outer = 0
                #
                # nmid = 1
                mid = str(nmid)
                ls = []
                maxts = []

                res = np.loadtxt(path + str(fid) + 'admm.txt')

                (l1, maxt) = plot_errmore(res, is_iter, 0, nmid % 3 == 2, 0)
                ls.append(l1)
                maxts.append(maxt)

                # min_err = min(res[:, 3])
                for i in range(1, 7):
                    name = path + str(fid) + 'aaadmm' + str(i) + '.txt'
                    res = np.loadtxt(name)
                    # min_err1 = min(res[:, 3])
                    # min_err = min(min_err, min_err1)
                    (l1, maxt) = plot_errmore(res, is_iter, i, nmid % 3 == 2, 0)
                    ls.append(l1)
                    maxts.append(maxt)

                # for i in range(1, 7):
                #     name = path + 'aadr' + str(i)  + '.txt'
                #     res = np.loadtxt(name)
                #

                if is_iter:
                    plt.xlabel("#Iters", font1)
                    plt.xlim(0, 1000)
                else:
                    plt.xlabel("Time(ms)", font1)
                    plt.xlim(0, max(maxts))

                plt.legend(handles=ls, loc='best', prop=font2)
                # plt.title('file' + str(fid) + '  outer ' + str(outer))
                plt.tight_layout()

                save_name = savepath + str(fid) + "emid_" + mid + str(is_iter) + "_outer_" + str(outer) + ".png"

                print(save_name)
                plt.savefig(save_name, transparent=True, dpi=150)
                plt.clf()
        # plt.show()

# AA-DR
if 1:
    ytype = 'r'
    for fid in range(1, 4):
        path = "D:/project/ADMMAA/localdeformation/splocs-master/data/coma_data/coma_data/res/"
        # os.system("mkdir D:\\project\\ADMMAA\\data\\no_noise_fig3\\file" + str(fid))
        savepath = "D:/project/ADMMAA/localdeformation/splocs-master/data/coma_data/coma_data/fig/"
        # for is_iter in (1, 0):
        is_iter = 1
        outer = 1
        nmid = 1
        aa_admm_m = 6
        aa_dr_m = 6

        for is_iter in range(0, 2):
            mid = str(nmid)
            ls = []
            maxts = []
            maxis = []

            name = path + str(fid) + '_mid0_outer_0.txt'
            res1 = np.loadtxt(name)
            # print(res)

            name = path + str(fid) + '_mid1_outer_0.txt'
            res2 = np.loadtxt(name)

            name = path + str(fid) + '_mid2_outer_0.txt'
            res3 = np.loadtxt(name)

           # ADMM
            min_err=0
            (l1, maxt) = plot_err3(res1, is_iter, 0, 0, 0, min_err, ytype, ' ')
            ls.append(l1)
            maxts.append(maxt)
            iter = 0
            for eachiter in range(1, len(res1)):
                if np.isnan([res1[eachiter, 0]]):
                    break
                # if res3[eachiter, 2] == 0:
                #     break
                iter = iter + 1
            maxis.append(iter)
            # print(iter)

            # AA-ADMM
            (l1, maxt) = plot_err3(res2, is_iter, aa_admm_m, 0, 1, min_err,ytype, ' ')
            ls.append(l1)
            maxts.append(maxt)
            iter = 0
            for eachiter in range(1, len(res2)):
                if np.isnan([res2[eachiter, 0]]):
                    break
                # if res2[eachiter, 2] == 0:
                #     break
                iter = iter + 1
            maxis.append(iter)
            # print(iter)

            ## AA-DR

            (l1, maxt) = plot_err3(res3, is_iter, aa_dr_m, 1, 2, min_err,ytype, '-PR')
            ls.append(l1)
            maxts.append(maxt)
            iter = 0
            for eachiter in range(1, len(res3)):
                if np.isnan([res3[eachiter, 0]]):
                    break
                # if res3[eachiter, 2] == 0:
                #     break
                iter = iter + 1
            maxis.append(iter)
            # print(iter)
            # plot_err3(data, is_iter, id, is_dr, cid, min_err, ytype, resetype):

            name =  path + str(fid) + '_mid3_outer_0.txt'
            res3 = np.loadtxt(name)
            (l1, maxt) = plot_err3(res3, is_iter, aa_dr_m, 1, 3, min_err, ytype, '-DRE')
            ls.append(l1)
            maxts.append(maxt)
            iter = 0
            for eachiter in range(1, len(res3)):
                if np.isnan([res3[eachiter, 0]]):
                    break
                # if res3[eachiter, 2] == 0:
                #     break
                iter = iter + 1
            maxis.append(iter)

            if is_iter:
                plt.xlabel("#Iters", font1)
                plt.xlim(0, max(maxis))
            else:
                plt.xlabel("Time(s)", font1)
                plt.xlim(0, max(maxts))
                # plt.xlim(0, 50)

            if ytype == 'r':
                plt.ylabel("Combined residual", font1)
            else:
                plt.ylabel("f(x)+g(z)", font1)
                # plt.ylabel("f(x)+g(z)", font1)
            plt.legend(handles=ls, loc='best', prop=font2)
            # plt.title('file' + str(fid))
            plt.tight_layout()

            save_name = savepath + str(fid) + ytype + "AA" + str(aa_admm_m) + "_DR" + str(aa_dr_m) + "_t" + str(is_iter) + "_outer_" + str(
                outer) + ".png"

            print(save_name)
            plt.savefig(save_name, transparent=True, dpi=600)
            plt.clf()

# AA-DR test mu
if 0:
    for fid in range(1, 5):
        path = "data/coma_data/res/f" + str(fid) + '_'
        # os.system("mkdir D:\\project\\ADMMAA\\data\\test_mu\\file" + str(fid))
        savepath = 'data/coma_data/fig/file' + str(fid) + '_'
        # for is_iter in (1, 0):
        is_iter = 1
        outer = 0
        nmid = 1
        aa_admm_m = 6
        aa_dr_m = 6

        for mu in (10, 100, 1000, 10000, 100000, 1000000):
            for is_iter in range(1, 2):
                mid = str(nmid)
                ls = []
                maxts = []

                res = np.loadtxt(path + 'mid0_mu' + str(mu) + '.txt')
                # ADMM
                # data, is_iter, id, is_dr, cid, min_err, ytype, resetype
                (l1, maxt) = plot_err3(res, is_iter, 0, 0, 0, 0, ytype, ' ')
                ls.append(l1)
                maxts.append(maxt)

                # AA-ADMM
                res = np.loadtxt(path + 'mid1_mu' + str(mu) + '.txt')
                (l1, maxt) = plot_err3(res, is_iter, 1, 0, 1, 0, ytype, ' ')
                ls.append(l1)
                maxts.append(maxt)
                # plot_reset(res, is_iter)

                ## AA-DR
                res = np.loadtxt(path + 'mid2_mu' + str(mu) + '.txt')
                (l1, maxt) = plot_err3(res, is_iter, 2, 1, 2, 0, ytype, '-PR')
                ls.append(l1)
                maxts.append(maxt)
                # plot_reset(res, is_iter)

                res = np.loadtxt(path + 'mid3_mu' + str(mu) + '.txt')
                (l1, maxt) = plot_err3(res, is_iter, 2, 1, 3, 0, ytype, '-DRE')
                ls.append(l1)
                maxts.append(maxt)

                if is_iter:
                    plt.xlabel("#Iters", font1)
                    plt.xlim(0, 100)
                else:
                    plt.xlabel("Time(ms)", font1)
                    plt.xlim(0, max(maxts))

                plt.legend(handles=ls, loc='best', prop=font2)
                plt.title('file' + str(fid) + '  outer ' + str(outer))
                plt.tight_layout()

                save_name = savepath + "mu" + str(mu) + ".png"

                print(save_name)
                plt.savefig(save_name, transparent=True, dpi=100)
                plt.clf()


def find_minerr(path):
    min_errs = []
    name = path + 'mid0_mu10_m' + str(1)  + '.txt'
    # print(name)
    res = np.loadtxt(name)
    min_errs.append(min(res[:, 1]))
    for m in range(1, 7):
        res = np.loadtxt(path + 'mid1_mu10_m' + str(m) + '.txt')
        min_errs.append(min(res[:, 1]))
        res = np.loadtxt(path + 'mid2_mu10_m' + str(m) + '.txt')
        min_errs.append(min(res[:, 1]))
    min_err = min(min_errs)
    return min_err

# AA-DR each iters
if 0:
    ytype = 'e'
    for fid in range(1, 11):
    # fid = 'monkey'
        path = "D:/project/ADMMAA/sparseicp/testmu/0_p5.0_f" + str(fid) + '_'
        savepath = 'D:/project/ADMMAA/sparseicp/figp05nreset/1_f' + str(fid) + '_'
        # for is_iter in (1, 0):
        is_iter = 1
        outer = 1
        nmid = 1
        aa_admm_m = 6
        aa_dr_m = 6

        # for outer in (0, 10, 20, 30, 40, 50, 60, 70, 80, 83):
        # for outer in (0, 10, 20, 30, 40, 50, 60):
        # for outer in (0, 5, 10, 15, 20, 25):
        # for outer in (0, 1, 2, 3, 4, 100, 101, 102, 103, 104, 191, 192, 193, 194, 195):
        # for outer in (100000, 50000, 10000, 5000, 1000, 500, 100):
        # for outer in (1000, 5000, 10000, 50000, 100000):
        for outer in range(10, 11):
            for is_iter in range(0, 2):
                mid = str(nmid)
                ls = []
                maxts = []

                min_err = find_minerr(path)
                print(min_err)
                # min_err = 0

                # print(path + 'mid0_outer' + str(outer) + '.txt')
                res1 = np.loadtxt(path + 'mid0_mu' + str(outer) + '_m1.txt')

                # print(path + 'mid1_m' + str(aa_admm_m) + '_outer' + str(outer) + '.txt')
                res2 = np.loadtxt(path + 'mid1_mu' + str(outer) + '_m' + str(aa_admm_m) + '.txt')

                # print(path + 'res_mid2_m' + str(aa_dr_m) + '_outer' + str(outer) + '.txt')
                res3 = np.loadtxt(path + 'mid2_mu' + str(outer) + '_m' + str(aa_dr_m) + '.txt')

                # ADMM
                (l1, maxt) = plot_err3(res1, is_iter, 0, 0, 0, min_err, ytype, '')
                ls.append(l1)
                maxts.append(maxt)

                # AA-ADMM
                (l1, maxt) = plot_err3(res2, is_iter, aa_admm_m, 0, 1, min_err, ytype, '')
                ls.append(l1)
                maxts.append(maxt)

                ## AA-DR
                (l1, maxt) = plot_err3(res3, is_iter, aa_dr_m, 1, 2, min_err, ytype, '')
                ls.append(l1)
                maxts.append(maxt)

                if is_iter:
                    plt.xlabel("#Iters", font1)
                    plt.xlim(0, 1500)
                else:
                    plt.xlabel("Time(ms)", font1)
                    plt.xlim(0, max(maxts))

                if ytype == 'e':
                    plt.ylabel('Energy', font1)
                else:
                    plt.ylabel('Combined residual', font1)

                plt.legend(handles=ls, loc='best', prop=font2)
                plt.title('file' + str(fid) + '  mu ' + str(outer))
                plt.tight_layout()

                save_name = savepath + ytype + "AA" + str(aa_admm_m) + "_DR" + str(aa_dr_m) + "_t" + str(
                    is_iter) + "_outer_" + str(outer) + "_5.png"

                print(save_name)
                plt.savefig(save_name, transparent=True, dpi=150)
                plt.clf()

# diff m each iters
if 0:
    fid = 'monkey'
    # for outer in (0, 10, 20, 30, 40, 50, 60, 70, 80, 83):
    # for outer in (0, 10, 20, 30, 40, 50, 60):
    # for outer in (0, 5, 10, 15, 20, 25):
    for fid in range(1, 11):
        for outer in range(10, 11):
            path = "D:/project/ADMMAA/sparseicp/testmu/0_p5.0_f" + str(fid) + '_'
            savepath = 'D:/project/ADMMAA/sparseicp/figp05nreset/1_f' + str(fid) + '_'
            # for is_iter in (1, 0):
            is_iter = 1
            for is_iter in range(0, 2):
                for nmid in range(1, 3):
                    mid = str(nmid)
                    ls = []
                    maxts = []

                    # min_err = find_minerr(path, outer)
                    min_err = 0
                    # if nmid > 3:
                    #     res = np.loadtxt(path + 'mid3_m5_outer' + str(outer) + '.txt')
                    # else:
                    res = np.loadtxt(path + 'mid0_mu' + str(outer) + '_m1.txt')

                    (l1, maxt) = plot_errmore(res, is_iter, 0, 0, min_err)
                    ls.append(l1)
                    maxts.append(maxt)

                    for i in range(1, 7):
                        res = np.loadtxt(path + 'mid' + mid + '_mu' + str(outer)+ '_m' + str(i)  + '.txt')
                        (l1, maxt) = plot_errmore(res, is_iter, i, nmid % 3 == 2, min_err)
                        ls.append(l1)
                        maxts.append(maxt)
                    if is_iter:
                        plt.xlabel("#Iters", font1)
                        plt.xlim(0, 2000)
                    else:
                        plt.xlabel("Time(ms)", font1)
                        plt.xlim(0, max(maxts))
                    plt.ylabel("Combined residual", font1)

                    plt.legend(handles=ls, loc='best', prop=font2)
                    plt.title('file' + str(fid) + '  mu ' + str(outer))
                    plt.tight_layout()

                    save_name = savepath + "mid_" + mid + str(is_iter) + "_outer_" + str(outer) + ".png"

                    print(save_name)
                    plt.savefig(save_name, transparent=True, dpi=150)
                    plt.clf()
        # plt.show()

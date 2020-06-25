#!/usr/bin/env python
import os
import sys
import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Set1_9
from palettable.colorbrewer.qualitative import Dark2_8
from cycler import cycler
#colors = cycler('color', [(0.6509803921568628, 0.33725490196078434, 0.1568627450980392), (0.21568627450980393, 0.49411764705882355, 0.7215686274509804), (1.0, 0.4980392156862745, 0.0), (1.0, 1.0, 0.2), (0.9686274509803922, 0.5058823529411764, 0.7490196078431373)])
colors = cycler('color', [(0.21568627450980393, 0.49411764705882355, 0.7215686274509804), (0.30196078431372547, 0.6862745098039216, 0.2901960784313726), (1.0, 0.4980392156862745, 0.0), 
	(0.6509803921568628, 0.33725490196078434, 0.1568627450980392), (0.9686274509803922, 0.5058823529411764, 0.7490196078431373), (0.8980392156862745, 0.5254901960784314, 0.023529411764705882)])
# import seaborn as sns

def main():
        # plt.style.use('ggplot')
	far_ratio = 10
	scale_value = 50*3*2105#1.4685, 1.1841, 1.077, 0.6742
	models=['nf5']
	for model in models:
		residuals = [model+'_mid0_mu10.txt',model+'_mid1_mu10.txt',model+'_mid2_mu10.txt']
		residual_name_lst = []#['residual_0','residual_1','residual_2','residual_3','residual_4','residual_5','residual_6']
		for res in residuals:
			res_name = res[:-4]
			residual_name_lst.append(res_name)

		time_name_lst = ['time_0','time_1','time_2','time_3','time_4','time_5','time_6']
		color_lst=['red','forestgreen','limegreen','gold','firebrick','mediumblue','darkorchid']
		label_lst=['ADMM','AA-ADMM','Ours (PR)', 'Ours (DRE)']
		for model in models:
			model_name = model
			
			res_id = 0
			prepare_list = locals()
			for residual in residuals:
				residual_name = residual
				time_n = time_name_lst[res_id]
				residual_n = residual_name_lst[res_id]
				prepare_list[time_n] = []
				prepare_list[residual_n] = []
				n_iter = 0
				with open(residual_name,'r') as f:
					for line in f:
						lst = line.split()
						prepare_list[time_n].append(float(lst[0]))
						prepare_list[residual_n].append(float(lst[1]))
						if n_iter>0:
							prepare_list[time_n][n_iter] = prepare_list[time_n][n_iter]-prepare_list[time_n][0]
							#prepare_list[residual_n][n_iter] = prepare_list[residual_n][n_iter]/prepare_list[residual_n][0]
						n_iter=n_iter+1
				prepare_list[time_n][0] = 0.0
				#prepare_list[residual_n][0] = 1.0
				res_id = res_id+1
			
			fig = plt.figure()
			plt.rcParams['axes.unicode_minus'] = False 
			plt.rc('font',family='Times New Roman')
			ax = fig.add_subplot(111)
			plt.xticks(fontsize=36)
			plt.yticks(fontsize=36)
	        # f, (ax, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw = {'width_ratios':[far_ratio, 1]})
			ax.set_yscale("log")
			# ax.set_xscale("log")
			
			# print(prepare_list[time_name_lst[0]])
			#ax.set_color_cycle(Dark2_8.mpl_colors)
			ax.set_prop_cycle(cycler(color=colors))
			for i in range(0,3,1):
				time_name = prepare_list[time_name_lst[i]]
				residual_name = prepare_list[residual_name_lst[i]]
				#color_name = color_lst[i]
				label_name = label_lst[i]
				ax.plot(time_name,residual_name,linewidth=4.0,label=label_name)#
			
			#ax.set_title('Soft Constraint: %s'%(model_name.capitalize()))

			# zoom-in / limit the view to different portions of the data
			max_time = 32#2.4
			unit_time = max_time / 4
			ax.set_xlim(0, max_time)  # outliers only
			#ax.set_ylim(0.000001,1000)
			#ax.set_ylim(0, 1000)
			ax.set_xticks([0, unit_time, 2*unit_time, 3*unit_time])
			ax.set_yticks([0.000000001, 0.000001, 0.001, 1])

			ax.set_xlabel('Time (s)',fontsize=36)
			#ax.set_ylabel('Residual (log)')
			#ax.legend(loc='best',fontsize=20)
			plt.minorticks_off()


			plt.savefig('time_%s.pdf'%(model), format='pdf', bbox_inches='tight', dpi=100)
			#plt.savefig('R_res_time_%s.png'%(model), format='png', bbox_inches='tight', dpi=1000)

			plt.clf()

			fig = plt.figure()
			ax = fig.add_subplot(111)
			plt.xticks(fontsize=36)
			plt.yticks(fontsize=36)
			ax.set_yscale("log")

			#ax.set_color_cycle(Dark2_8.mpl_colors)
			ax.set_prop_cycle(cycler(color=colors))
			for i in range(0,3,1):
				residual_name = prepare_list[residual_name_lst[i]]
				#color_name = color_lst[i]
				iter_name = range(0,len(residual_name))
				label_name = label_lst[i]
				ax.plot(iter_name,residual_name,linewidth=4.0,label=label_name)
			
			#ax.set_title('Soft Constraint: %s'%(model_name.capitalize()))

			max_iter = 800#400
			unit_iter = max_iter / 4
			ax.set_xlim(0, max_iter)  # outliers only
			#ax.set_ylim(0.000001,1000)
			#ax.set_ylim(0, 1000)
			ax.set_xticks([0, unit_iter, 2*unit_iter, 3*unit_iter])
			ax.set_yticks([0.000000001, 0.000001, 0.001, 1])

			ax.set_xlabel('Iter',fontsize=36)
			#ax.set_ylabel('Residual (log)')
			#ax.legend(loc='best',fontsize=20)
			plt.minorticks_off()

			plt.savefig('iter_%s.pdf'%(model), format='pdf', bbox_inches='tight', dpi=100)
			#plt.savefig('R_res_iter_%s.png'%(model), format='png', bbox_inches='tight', dpi=1000)
main()

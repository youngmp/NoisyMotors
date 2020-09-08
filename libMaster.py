# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:44:48 2020

@author: zj
"""

import matplotlib.pyplot as plt
import numpy as np



def plot_heatmap(obj):

    if obj.use_storage:

        fig2 = plt.figure(figsize=(4,4))
        ax_fig2 = fig2.add_subplot(111)

        # extract only transition points
        hist_data = ax_fig2.hist2d(obj.X,obj.Y,bins=[np.arange(0,obj.nX+1.5)-0.5,np.arange(0,obj.nY+1.5)-0.5])

        ax_fig2.set_title('raw histogram')
        
        fig3 = plt.figure(figsize=(4,4))
        ax_fig3 = fig3.add_subplot(111)
        
        t = np.arange(len(obj.X))
        #ax_fig3.scatter(obj.X,obj.Y,alpha=.1,label='X',c=t,cmap='viridis')
        th = 200

        # set numbers above threshold to threshold
        #th_mat = hist_data[0]*(hist_data[0]>th)
        #th_mat
        th_mat = hist_data[0]
        th_mat[th_mat > th] = th
        
        #th_mat = hist_data[0]*(hist_data[0]<th) + 
        ax_fig3.imshow(th_mat[:,:])#,vmin=0,vmax=th)
        ax_fig3.set_title('show all with threshold='+str(th))
        
        fig4 = plt.figure(figsize=(4,4))
        ax_fig4 = fig4.add_subplot(111)
        
        ax_fig4.set_title('show only below threshold='+str(th))
        a = hist_data[0]*(hist_data[0]<th)
        ax_fig4.imshow(a[:,:])

        
        fig5 = plt.figure(figsize=(4,4))
        ax_fig5 = fig5.add_subplot(111)
        
        
        fig6 = plt.figure(figsize=(4,4))
        ax_fig6 = fig6.add_subplot(111)
        
        #hist = ax_fig6.hist(obj.X,bins=np.arange(0,obj.nX+1.5)-0.5,alpha=.5)
        #hist = ax_fig6.hist(obj.Y,bins=np.arange(0,obj.nY+1.5)-0.5,alpha=.5)
        
        ax_fig6.hist(obj.X,bins=np.arange(0,obj.nX+1.5)-0.5,alpha=.5)
        ax_fig6.hist(obj.Y,bins=np.arange(0,obj.nY+1.5)-0.5,alpha=.5)
        
        
        fig7 = plt.figure(figsize=(4,4))
        ax_fig7 = fig7.add_subplot(111)
        
        #hist = ax_fig7.hist(obj.V,100,alpha=.5)
        ax_fig7.hist(obj.V,100,alpha=.5)
        
        plt.show()
        
        
def plot_traces(obj):


    if obj.use_storage:

        fig = plt.figure(figsize=(8,3))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.plot(obj.t,obj.X,alpha=.5,label='X')
        ax1.plot(obj.t,obj.Y,alpha=.5,label='Y')

        ax2.plot(obj.t,obj.V,alpha=.5)
        ax2.scatter(obj.switch_times,np.zeros(len(obj.switch_times)),color='tab:red',s=10)
        
        ax1.set_xlabel('t (s)')
        ax1.set_title(r'$n_X=%d,n_Y=%d$'%(obj.nX,obj.nY))
        #ax1.set_title('Linear position-velocity curve')
        ax2.set_xlabel('t (s)')
        
        ax1.set_ylabel('Motor Number')
        ax2.set_ylabel('Velocity')
        
        plt.suptitle('Master Equation Solution with Linear Position-Velocity. MFPT='+str(obj.mfpt))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        #plt.show()

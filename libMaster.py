# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:44:48 2020

@author: zj
"""



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
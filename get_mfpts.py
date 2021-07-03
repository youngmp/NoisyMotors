# -*- coding: utf-8 -*-
"""
extract mfpt from all unordered txt files
"""

import re
import numpy as np

import matplotlib.pyplot as plt


import os


def get_mfpts(path,options,diffs):
        
    fnames = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            #if(file.endswith(".txt")):
            #print(file,[x in file for x in options])
            if all(x in file for x in options):
                fname = os.path.join(root,file)
                #print(fname)
                fnames.append(fname)
    
    
    all_time_diffs = np.array([])
    
    #cutoff = int(len(fnames)/2)
    cutoff = len(fnames)
    
    print('Total seeds =', len(fnames[:cutoff]))
    
    for fname in fnames[:cutoff]:
        
        times = np.loadtxt(fname)
        
        
        # for compatibility with Julia outputs
        #print(fname)
        
        #a = re.search(r'seed=[0-90-9]',fname)
        #idx = a.start()
        #print(fname[idx:idx+7])
        
        if diffs:
            time_diffs = np.diff(times)
            #print(fnames)
        else:
            time_diffs = times
        #print(fname,np.mean(time_diffs))
        
        all_time_diffs = np.append(all_time_diffs,time_diffs)
        
        
    return all_time_diffs

def main():

    import pathlib
    dir_path = os.path.dirname(os.path.realpath(__file__))


    # data directory python
    #path = dir_path + "/p2"
    #options = ['dt=5e-07','.txt',]
    #diffs = True
    
    # datadir julia
    path = dir_path + '/../julia/mfpt2a'
    options = ['dt=1.0e-05','.txt']
    diffs = False

    print(path)
    
    mfpts = get_mfpts(path,options,diffs)
    

    print(np.mean(mfpts))
    
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(mfpts,bins=200,density=True)
        plt.show()
        
if __name__ == "__main__":
    main()

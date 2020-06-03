# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:59:28 2020

@author: YP

"""

from numpy.linalg import norm

import numpy as np


def disp_params(a,show_arrays=False,inline=True):
    """
    
    Parameters
    ----------
    a : object
        Noisy motor object containing simulation and model parameters.
    show_arrays : bool, optional
        If true, display all arrays of the object alongside parameters.
        The default is False.
    inline: bool, optional
        if true, display all parameters in line. 
        Else use newline per parameter

    Returns
    -------
    None.

    """
    
    # verbose information
    par_dict = vars(a)
    
    sorted_keys = sorted(par_dict.keys(),key=lambda v: v.upper())
    print('*\t ',end='')
    for key in sorted_keys:
        val = par_dict[key]
        #print(type(val),end=',')
        if not(type(val) is np.ndarray):
                
            print(key,'=',val,end='; ')
            
    print()
    
    
def disp_norms(obj,ground_truth_values):
    sol_final = obj.sol[-1,:]
    sol_true = ground_truth_values
    
    diff = sol_final - sol_true
    
    print('*\t L1 = ',"{:e}".format(np.sum(np.abs(diff))*obj.dx))
    print('*\t L2 = ',"{:e}".format(np.sum(np.abs(diff**2))*obj.dx))
    print('*\t L_inf = ',"{:e}".format(norm(diff,ord=np.inf)))
    
    theta_n = obj.theta_n
    theta_true = np.sum(obj.ground_truth)*obj.dx
    err = np.abs(theta_n-theta_true)
    print('*\t |theta_n - theta_true| =', "{:e}".format(err))
    
    
def ground_truth(obj):
    # if U is fixed, set up ground truth
    if obj.U < 0:
        obj.part_idxs = obj.idx_full[:obj.A_idx+1]
    else:           
        obj.part_idxs = obj.idx_full[obj.A_idx:obj.B_idx+1]

    obj.ground_truth = np.zeros_like(obj.x)
    obj.ground_truth[obj.part_idxs] = phi(obj)
    
    return obj.ground_truth
    

def phi(obj):
    """
    Ground truth steady-state phi function for U<0 and U>0
    """
    z = obj.x[obj.part_idxs]
    sgn = np.sign(obj.U)

    #out = np.zeros_like(z)
    al = obj.alpha
    be = obj.beta
    U = obj.U
    
    ee = np.exp((obj.A-z)*be/U)
    ee2 = np.exp((obj.A-obj.B)*be/U)
    
    if sgn < 0:
        return sgn*ee*al*be/(obj.U*(al+be))
    else:
        return sgn*ee*al*be/(obj.U*(al-ee2*al+be))


def gauss(x,sig):
    return np.exp(-(x/sig)**2)


def fname_suffix(exclude=[],ftype='.png',**kwargs):
    """
    generate filename suffix based on parameters
    """
    fname = ''
    for key in kwargs:
        if key not in exclude:
            if type(kwargs[key]) is dict:
                kw2 = kwargs[key]
                for k2 in kw2:
                    fname += k2+'='+str(kw2[k2])+'_'
            else:
                fname += key+'='+str(kwargs[key])+'_'
    fname += ftype
    
    return fname


def mass_fn(t,initial_mass,**kwargs):
    """
    total mass obeys
    dI/dt = alpha*(1-I) - beta*I
    solve to obtain
    
    I(t) = alpha/(alpha+beta) + [I(0)-alpha/(alpha+beta)]*exp(-(alpha+beta)*t)

    """
    al = kwargs['alpha']
    be = kwargs['beta']
    
    return al/(al+be) + (initial_mass - al/(al+be))*np.exp(-(al+be)*t)
    
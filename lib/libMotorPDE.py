# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:59:28 2020

@author: YP

"""

import time
from numpy.linalg import norm
#from scipy.interpolate import interp1d

from .interp_basic import interp_basic as interpb
#from cumsumb import cumsum



    
import scipy.stats as st
import numpy as np

#import matplotlib
#matplotlib.use('TkAgg')


def inverse_transform_sampling(obj,pdf_array,n_samples,tol=1e-32,spec='X',
                               vel=0,ext=True):
    """
    inverse transform conditioned on attached motors.
    
    construct PDF only using pdf_array with probabilities above tol.
    
    It is important to keep the zero-value density at the leftmost
    nonzero density point because of the distribution. Without the 
    leftmost density point, the density begins at a nonzero value
    and therefore its integral is also nonzero. Inverting this
    function results in a function with a domain on a subset of 
    [0,1], e.g., [0.2,1]. Using my version of interpolation,
    values put into a function with a domain on [0.2,1] returns 0.
    Since a nontrivial region of the domain returns 0, averages and
    such are thrown off significantly.
    
    At the moment the rightmost point in the distribution returns zero,
    but the distribution function appears to be working for [0,.999]
    which should be good enough for us.
    
    """

    
    array = pdf_array[1:]
    xobj = obj.x[1:]
    #global cumsum
    #print(cumsum)
    #print(pdf_array,n_samples)
    #sum_values_old = np.cumsum(array*obj.dx)\
    #    /(np.add.reduce(array*obj.dx))
    cumsum1 = np.cumsum(array*obj.dx)
    #cumsum1 = cumsum(array*obj.dx)
    sum_values_old = cumsum1/cumsum1[-1]
    

    #print(obj.A_idx,obj.dx,obj.irregular)
    #array[:obj.A_idx] = np.cumsum(array[:obj.A_idx])*obj.dx[0]
    #array[obj.A_idx:] = np.cumsum(array[obj.A_idx:])*obj.dx[-1]

    #sum1 = np.add.reduce(array[:obj.A_idx])*obj.dx[0]
    #sum2 = np.add.reduce(array[obj.A_idx:])*obj.dx[-1]

    #array[:obj.A_idx] /= sum1
    #array[obj.A_idx:] /= sum2
    
    #sum_values_old = array
    
    
    #sum_values_old = np.cumsum(pdf_array)/np.add.reduce(pdf_array)
    
    # ignore positions with probability below tol
    keep_idxs = (array > tol)
    
    # keep the leftmost index with probabily below tol
    # this might seem unnecessary but is extremely important.
    # otherwise interp1d will take positions outside the domain of interest.
    keep_idx_left = np.argmax(keep_idxs > 0)-1
    
    
    if keep_idx_left == -1:
        sum_values_old[0] = 0  # force 0 when at left boundary
    else:
        keep_idxs[keep_idx_left] = True
        
    # find rightmost index
    #b = keep_idxs[::-1]
    #keep_idx_right = (len(b) - np.argmax(b>0) - 1) + 1

    sum_values = sum_values_old[keep_idxs]
    x = xobj[keep_idxs]
    #print(x,obj.dx,sum_values)

    #print()
    #print(obj.A_idx,array[obj.A_idx])
    #print(obj.dx[keep_idxs])
    #print(array[keep_idxs])
    #print(np.cumsum(array)[keep_idxs])
    #print()

    r = np.random.rand(n_samples)
    #print(r,inv_cdf(r))
    
    inv_cdf = interpb(sum_values,x)
    
    if False and spec == 'X': # and obj.i % 10000 == 0: # inv_cdf(1) == 0 and spec == 'X':
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        #ax.plot(x,inv_cdf(x),label='inv_cdf')
        ax1.plot(xobj,array,label='pdf_array')

        #ax.scatter(obj.x[keep_idx_left],0)
        #ax2.plot(xobj,sum_values_old,label='sum_vals_old')
        ax2.plot(x, sum_values, label='sum_vals')

        domain = np.linspace(0,1,1000)
        y = inv_cdf(domain)
        ax3.plot(domain,y,label='F inverse')

        ax4.plot(x[1:],np.diff(sum_values),
                 label='dF')

        #ax.plot(np.linspace(0,1,100),inv_cdf(np.linspace(0,1,100)))
        #ax.set_title(obj.i*obj.dt)


        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()

        #ax.set_xlim(4.9,5.3)
        plt.show(block=True)
        plt.close()
        time.sleep(2)

        #print(r,inv_cdf(r))

    
    

    if False and spec == 'X':
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.plot(x,inv_cdf(x),label='inv_cdf')
        #ax.plot(obj.x,pdf_array,label='pdf_array')
        #ax.scatter(obj.x[keep_idx],0)
        ax.plot(sum_values,x,label='sum_vals')
        ax.plot(np.linspace(0,.999,100),inv_cdf(np.linspace(0,.999,100)))
        ax.set_title(obj.i*obj.dt)
        ax.legend()
        #ax.set_xlim(4.9,5.3)
        plt.show(block=True)
        plt.close()
        time.sleep(.2)
        
        #print(r,inv_cdf(r))

    
    
    return inv_cdf(r)


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
    
    
def force_position(x,p1=4,gamma=0.322,choice='exp'):
    """
    p1 = 4 # pN
    gamma = 0.322 # /nm
    """

    #return x#/self.gamma
    if (choice == 'linear') or (choice == 'lin'):
        return x*p1*gamma
    
    elif choice == 'exp':
        
        return p1*(np.exp(gamma*x)-1)
        
    else:
        raise Exception('Unrecognized force-position curve',choice)
        

    
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
        x = obj.x[obj.part_idxs]
    else:
        obj.part_idxs = obj.idx_full[obj.A_idx:]
        #print(obj.part_idxs)
        #obj.x[obj.part_idxs]+= .002
        x = np.linspace(obj.A,obj.B,len(obj.part_idxs))

    obj.ground_truth = np.zeros_like(obj.x)
    obj.ground_truth[obj.part_idxs] = phi(x,obj.U,obj)
    
    return obj.ground_truth
    

def phi(x,U,obj):
    """
    Ground truth steady-state phi function for U<0 and U>0
    """

    sgn = np.sign(U)

    #out = np.zeros_like(z)
    al = obj.alpha
    be = obj.beta


    ee = np.exp((obj.A-x)*be/U)
    ee2 = np.exp((obj.A-obj.B)*be/U)
    
    if sgn < 0:
        return sgn*ee*al*be/(U*(al+be))
    else:
        return sgn*ee*al*be/(U*(al-ee2*al+be))


def gauss(x,sig):
    return np.exp(-(x/sig)**2)


def fname_suffix(exclude=[],ftype='.png',**kwargs):
    """
    generate filename suffix based on parameters
    """
    
    fname = ''
    for key in kwargs:
        #print(key not in exclude,key,exclude)
        if key not in exclude:
            if type(kwargs[key]) is dict:
                kw2 = kwargs[key]
                for k2 in kw2:
                    fname += k2+'='+str(kw2[k2])+'_'
            elif callable(kwargs[key]):
                fname += key+'=callable_'
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
    

class x_pdf_gen(st.rv_continuous):
    """
    needed to create custom PDF
    
    PDF for motor position based on population distribution.
    
    f must be a function. in our case it is the interp1d function.
    
    to generate PDF, write
    
    x_pdf = x_pdf_gen(a=A0,b=B,name='x_pdf')
    
    then to draw,
    
    x_pdf.rvs(size=10)
    
    to draw 10 samples.
    """
    
    def __init__(self,f,a,b,name):
        st.rv_continuous.__init__(self)
        self.f = f
        self.a = a
        self.b = b
        self.name = name
        
        print(a,b)
    
    def _pdf(self,x):
        #print(x)
        return self.f(x)
  
    
def get_time_index(use_storage,i):
    if use_storage:
        k_next = i+1
        k_current = i
    else:
        k_next = 0
        k_current = 0
    
    return k_next, k_current

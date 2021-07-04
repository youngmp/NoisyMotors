# fit steady-state distribution of the langevin equation to the agent-based model.

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from agents import Agents
import lib.parsets as pset

popfactor = 1
n0 = 100*popfactor
zeta = .4
A = 5
B = 5.05
al = 14
be = 126
k = 4*.322/popfactor


def fpref(u):
    """
    negative-preferred motors, u < 0
    """
    
    return -k*n0*al*(u+A*be)/(be*(al+be))


def fnonpref(u):
    """
    negative-preferred motors, u > 0
    """
    ee = np.exp((A-B)*be/u)
    return -k*n0*al*(u+A*be-ee*(u+B*be))/(be*(al-ee*al+be))


def fv_down(u):
    """
    u: arrays only
    """
    out = np.zeros_like(u)

    out[u<0] = fpref(u[u<0])
    out[u>=0] = fnonpref(u[u>=0])

    return out

def fmotors(u):
    return fv_down(u) - fv_down(-u)

def ps_non_normal(x,sigma,a=-400,b=400,n=10000,return_area=False):
    """
    unnormalized steady-state distribution
    """

    u, du = np.linspace(a,b,n,retstep=True)

    integral = np.cumsum(fmotors(u)-zeta*u)*du/sigma**2
    ps_array = np.exp(2*integral)/sigma**2

    ps_callable = interp1d(u,ps_array)

    if return_area:
        return ps_callable(x),np.sum(ps_array)*du
    else:
        return ps_callable(x)

def ps(x,sigma):
    val, area = ps_non_normal(x,sigma,return_area=True)
    return val/area
    
def run_agents(recompute=True,nX=100):

    # agent simulation
    kwargs = {'T':20,
              'V0':0,
              'Z0':0,
              'dt':5e-6,
              'seed':9,
              'U':None,
              'force_pos_type':'lin',
              'store_position':False,
              'ext':True}
    
    kwargs = dict(kwargs,**pset.parset('figure1'))
    kwargs['nX'] = nX
    kwargs['nY'] = nX
    kwargs['p1'] /= nX/100 # keep mean velocity the same
    kwargs['store_position'] = True

    suffix = ('n0='+str(nX)
              +'_seed='+str(kwargs['seed'])
              +'.txt')
             
    fname_switch = 'data/agents_switch'+suffix
    fname_counts = 'data/agents_counts'+suffix
    fname_bins = 'data/agents_bins'+suffix
    #fname_Z = 'data/agents_Z'+suffix
    
    file_does_not_exist = not(os.path.isfile(fname_switch))
        
    if recompute or file_does_not_exist:
        a = Agents(**kwargs)
        a.run_agents()
        
        switch_agents = a.switch_times
        t_agents = a.t
        V = a.V
        Z = a.Z

        counts, bins, bars = plt.hist(V,bins=40,
                                      density=True)
        
        np.savetxt(fname_switch,a.switch_times)
        np.savetxt(fname_counts,counts)
        np.savetxt(fname_bins,bins[:-1])

        
    else:
        switch_agents = np.loadtxt(fname_switch)
        counts = np.loadtxt(fname_counts)
        bins = np.loadtxt(fname_bins)
        #t_agents = np.loadtxt(fname_t)
        #V = np.loadtxt(fname_V)
        #Z = np.loadtxt(fname_Z)

    #counts, bins, bars = plt.hist(V,bins=40,
    #                              density=True)

    return bins[:-1], counts


def main():

    bins, counts = run_agents(nX=250)

    #bins = np.loadtxt('bins.csv')[:-1]
    #counts = np.loadtxt('counts.csv')
    
    out = curve_fit(ps,bins,counts,65)
    print(out)
    

    #sigma = 47
    sigma = out[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.linspace(-350,350,100000)
    ax.plot(x,ps(x,sigma))
    #ax.plot(x,ps(x,55))
    #ax.plot(x,ps(x,60))
    ax.plot(bins,counts)
    plt.show()

    
if __name__ == "__main__":
    main()

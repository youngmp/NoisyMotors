# -*- coding: utf-8 -*-
"""
import and test MotorPDE.py


"""

from .MotorPDE import MotorPDE as mpde
#from MotorPDE_nokink import MotorPDE as mpde
#from MotorPDE_pref_only import MotorPDE as mpde
from lib.interp_basic import interp_basic as interpb

import lib.libMotorPDE as lib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

import csv
import os
#import time
#import inspect

from scipy.interpolate import interp1d
from scipy.signal import square

from math import log10,floor

#import matplotlib
#matplotlib.use("Agg")


# user modules


# default kwargs to reduce clutter
opt = {'U':-10, 'A0':0, 'A':5, 'B':6,
       'T':0.02,'beta':126,'alpha':14,
       'ivp_method':'euler','store_position':True}





def inverse_transform_sampling(domain,pdf_array,n_samples,dx,tol=1e-32,spec='X',
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
    
    array = pdf_array
    xobj = domain

    #print(pdf_array,n_samples)
    sum_values_old = np.cumsum(array*dx)\
        /(np.add.reduce(array*dx))
    
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
    
    if False and spec == 'X': # inv_cdf(1) == 0 and spec == 'X':

        if True:
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

            p = np.linspace(0,1,10000)
            y = inv_cdf(p)
            ax3.plot(p,y,label='F inverse')

            a = .121
            b = .355
            r_temp = np.random.rand(1000000)*(b-a)+a
            #ax3.scatter(r_temp,inv_cdf(r_temp))
            
            ax4.hist(inv_cdf(r_temp),density=True,bins=50)
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
            #time.sleep(.2)

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




def round_to_n(x,n):
    """

    Parameters
    ----------
    x

    Returns
    -------
    rounded number to n sig figs
    """
    return round(x, -int(floor(log10(x))) + (n - 1))


# create test directory for convergence results and plots
DIR_TESTS = 'tests/'

if not(os.path.exists(DIR_TESTS)):
    os.makedirs(DIR_TESTS)


def test_conservation(show=False,CFL=0.5,Nlist=10**np.arange(2,5,1),**kwargs):
    """
    conservation number should be constant over time.
    
    diff mass gives difference in integration value.
    
    diff final gives max difference between PDE and true solution.
    """
    
    print()
    print('Testing Conservation')
    print(kwargs)

    diff_mass_list = []
    diff_final_list = []

    # exclude these parameters from file names
    exclude_params = ['A0','beta','irregular','ivp_method','source','type',
                      'source','gaussian_pars']

    # save to csv file
    fname_csv = DIR_TESTS+'conservation_'+\
                lib.fname_suffix(exclude=exclude_params,
                                 ftype='.csv',**kwargs)

    with open(fname_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N','diff mass', 'diff final (max)'])
        
        print('N\t diff mass\t diff final (max err)')

        for i in range(len(Nlist)):

            kwargs['N'] = Nlist[i]

            # use finer mesh
            n = np.amax([kwargs['N'],kwargs['N2']])
            dx = (kwargs['B'] - kwargs['A0']) / n
            kwargs['dt'] = CFL * dx / np.abs(kwargs['U'])

            print('dt', kwargs['dt'])

            # print(kwargs)
            a = mpde(**kwargs)  # initialize
            # lib.disp_params(a)  # display non-array parameters

            # t0 = time.time()
            a.run()  # run
            # t1 = time.time()

            sig = kwargs['init_pars']['pars']
            shifted_domain = a.x - (a.A0 + a.B) / 2 - kwargs['U'] * kwargs['T']
            ground_truth_values = lib.gauss(shifted_domain, sig=sig)
            # lib.disp_norms(a,ground_truth_values)

            fig = plt.figure(figsize=(10, 5))
            ax11 = fig.add_subplot(221)
            ax12 = fig.add_subplot(222)
            ax21 = fig.add_subplot(223)
            ax22 = fig.add_subplot(224)

            ax11.plot(a.x,a.init, label='init')
            ax12.plot(a.x,a.sol[-1, :], label='pde')
            ax12.plot(a.x,ground_truth_values, label='ground truth')

            if kwargs['store_position']:
                # get total mass per time
                # print(np.shape(a.sol))

                mass = np.sum(a.sol[:, 1:] * a.dx, axis=1)

                # print(np.shape(mass))
                ax21.plot(mass, label='mass')

                ax21.set_title('Mass')
                ax21.set_xlabel('t index')
                # ax21.set_title('Timecourse at specific points')

            abs_diff = np.abs(a.sol[-1, :] - ground_truth_values)
            ax22.plot(a.x, abs_diff, label=r'$\|\cdot\|_1$')

            # print('max diff',np.amax(np.abs(a.sol[-1,:]-ground_truth_values)))

            ax11.scatter(a.x[a.A_idx], 0, label='x=A index')
            ax12.scatter(a.x[a.A_idx], 0, label='x=A index')

            ax12.set_title('initial')
            ax12.set_title('final')

            ax22.set_title('final-(true)')

            # ax11.set_xlabel('x index')
            ax11.set_xlabel('x')
            ax12.set_xlabel('x')

            ax11.legend()
            ax12.legend()
            ax21.legend()
            ax22.legend()

            plt.tight_layout()

            # a.plot(figure='final-diff')
            fname = DIR_TESTS + 'conservation_' +\
                    lib.fname_suffix(exclude=exclude_params,
                                     ftype='.png',**kwargs)
            # print('*\t saving to ',fname)

            plt.savefig(fname)

            # diff in mass init vs final
            diff_mass = np.abs(
                (np.sum(a.init[:-1] * a.dx) - np.sum(a.sol[-1, :-1] * a.dx)))
            print(diff_mass)

            # diff in final function vs ground truth
            diff_final = np.amax(np.abs(a.sol[-1, :] - ground_truth_values))

            if show:
                plt.show(block=True)

            plt.close()

            diff_mass_list.append(diff_mass)
            diff_final_list.append(diff_final)


            writer.writerow([Nlist[i],diff_mass_list[i],diff_final_list[i]])
            print(Nlist[i],'\t',diff_mass_list[i],'\t',diff_final_list[i])
    
    


def test_flux(CFL=0.5,Nlist=10**np.arange(2,5,1),**kwargs):
    """
    mass over time + integral of flux over time = constant
    
    to quantify, take sum(diff(array)), where array should be constant.
    """
    
    print()
    print('Testing Flux')
    print(kwargs)
    
    mass_with_flux_list = []
    
    for N in Nlist:
        kwargs['N'] = N
        dx = (kwargs['B']-kwargs['A0'])/kwargs['N']
        kwargs['dt'] = CFL*dx/np.abs(kwargs['U'])
        
        run_flux(show=False,**kwargs)
        mass_with_flux = run_flux(show=False,**kwargs)
        
        diff_sum = np.sum(np.diff(mass_with_flux))
        
        mass_with_flux_list.append(diff_sum)
    
        # save to csv file
    fname_csv = (DIR_TESTS+'flux_'
                 + lib.fname_suffix(exclude=['N'],ftype='.csv',**kwargs))
    
    with open(fname_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N','mass + int flux, np.diff, np.sum'])
        
        print('N\t d(flux)==0')
        for i in range(len(Nlist)):
            writer.writerow([Nlist[i],mass_with_flux_list[i]])
            print(Nlist[i],'\t',mass_with_flux_list[i])
    
    
def run_flux(show=False,**kwargs):
    """
    Run simulation for a given set of parameter values
    and generate relevant plots
    """

    assert(kwargs['store_position'] is True)
    
    a = mpde(**kwargs)
    #lib.disp_params(a)  # display non-array parameters
    
    #t0 = time.time()
    a.run()
    #t1 = time.time()
    
    #print('*\t Run time',t1-t0)
    #print('.theta_n',a.theta_n)
    
    sig = kwargs['init_pars']['pars']
    
    true_flux = np.zeros(len(a.sol[:,0]))
    true_mass = np.zeros(len(a.sol[:,0]))
    
    for i in range(len(true_flux)):
        true_flux[i] = lib.gauss(a.x-(a.A0+a.B)/2-a.U*i*a.dt,sig=sig)[-1]*a.U
        true_mass[i] = np.sum(lib.gauss(a.x-(a.A0+a.B)/2-a.U*i*a.dt,sig=sig))*a.dx
        
    #lib.disp_norms(a,ground_truth_values)
    #print(a.sol[0,0])
    flux = a.U*(a.sol[:,-1] - a.sol[:,0])
    cumulative_flux = np.cumsum(flux[:-1])*a.dt
    mass = np.sum(a.sol,axis=1)[1:]*a.dx
    
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(cumulative_flux+mass,label='cumulative flux PDE + mass PDE')
    ax1.plot(cumulative_flux,label='cumulative flux PDE')
    ax1.plot(mass,label='mass PDE')
    #ax1.plot(true_flux,label='cumulative flux true')

    #ax2.plot(np.abs(cumulative_flux-true_flux),label='diff')
    ax2.plot(np.diff(cumulative_flux+mass),label='d/dt(cumulative flux pde)')
    
    
    ax1.set_xlabel('t index')
    ax2.set_xlabel('t index')
    
    ax1.set_ylabel('cumulative flux + mass')
    ax2.set_ylabel('derivative')
    
    
    fname = DIR_TESTS+'flux_'+lib.fname_suffix(ftype='.png',**kwargs)
    #print('*\t saving plot to ',fname)
    
    plt.tight_layout
    plt.savefig(fname)
    
    if show:
        plt.show(block=True)
    
    plt.close()
    
    return cumulative_flux+mass


def test_u_fixed_ss(CFL=0.2,Nlist=10**np.arange(2,5,1),**kwargs):
    
    print()
    print('Testing Fixed U='+str(kwargs['U']))
    print(kwargs)

    exclude_params = ['A0', 'beta', 'irregular', 'ivp_method', 'source', 'type',
                      'source', 'gaussian_pars']
        
    # save to csv file
    fname_csv = (DIR_TESTS+'U_fixed_ss'
                 + lib.fname_suffix(exclude=exclude_params,
                                    ftype='.csv',**kwargs))
    
    with open(fname_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N','L1','diff max','L1 rel err'])

        for i in range(len(Nlist)):

            kwargs['N'] = Nlist[i]
            
            a = mpde(**kwargs)
            
            #print('irreg?',a.irregular)
            
            if kwargs['irregular']:
                kwargs['N2'] = Nlist[i]
                kwargs['dt'] = CFL * np.amin(a.dx) / np.abs(kwargs['U'])
            else:
                kwargs['N'] = Nlist[i]
                kwargs['dt'] = CFL * a.dx / np.abs(kwargs['U'])

            
            a.__init__(**kwargs)
            print('N',Nlist[i],'dt', kwargs['dt'])

            

            # lib.disp_params(a)  # display non-array parameters

            # t0 = time.time()
            a.run()
            # t1 = time.time()
            # print('*\t Run time',t1-t0)

            ground_truth_vals = lib.ground_truth(a)
            # lib.disp_norms(a,ground_truth_values)

            fig = plt.figure(figsize=(10, 5))
            ax11 = fig.add_subplot(121)
            ax12 = fig.add_subplot(122)

            #print(a.x[-20:])

            ax11.plot(a.x, np.abs(a.sol[-1, :] - ground_truth_vals),
                      label='|pde-(true)|')

            ax12.plot(a.x, a.sol[-1, :], label='pde')
            ax12.plot(a.x, ground_truth_vals, label='ground truth')

            ax12.scatter(a.x[a.A_idx], 0, label='x=A index')
            ax11.scatter(a.x[a.A_idx], 0, label='x=A index')

            ax12.set_title('final')
            ax11.set_title('final-(true)')

            # ax11.set_xlabel('x index')
            ax11.set_xlabel('t')
            ax12.set_xlabel('x')
 
            pad = (a.B - a.A) / 10
            # ax11.set_xlim(a.A-pad,a.B+pad)
            # ax12.set_xlim(a.A-pad,a.B+pad)

            ax11.legend()
            ax12.legend()

            plt.tight_layout()

            # a.plot(figure='final-diff')
            fname = (DIR_TESTS
                     + 'U_fixed_ss_'
                     + lib.fname_suffix(exclude=exclude_params, **kwargs))
            plt.savefig(fname)
            plt.close()

            true_area = np.sum(ground_truth_vals) * a.dx

            L1 = np.sum(np.abs(a.sol[-1, :] - ground_truth_vals)) * a.dx
            diff = np.amax(np.abs(a.sol[-1, :] - ground_truth_vals)) * a.dx
            rel_err = np.sum(
                np.abs(a.sol[-1, :] - ground_truth_vals)) * a.dx / true_area

            a = Nlist[i]
            b = L1
            c = diff
            d = rel_err

            writer.writerow([a, b, c, d])
    


def test_U_fixed_dynamics(CFL=0.5,dt_list=5.**np.arange(-4,-9,-1),**kwargs):
    
    
    print()
    print('Testing Fixed U='+str(kwargs['U']))
    print(kwargs)

    diff_list = []
    
    kwargs['N'] = 100
    
    for dt in dt_list:
        
        #kwargs['dx'] = np.abs(kwargs['U'])*dt/CFL
        #kwargs['N'] = int((kwargs['B']-kwargs['A0'])/kwargs['dx'])
        kwargs['dt'] = dt
        diff = run_U_fixed_dynamics(show=False,**kwargs)
        
        diff_list.append(diff)
        
    # save to csv file
    fname_csv = (DIR_TESTS+'U_fixed_dynamics'
                 + lib.fname_suffix(exclude=['N'],ftype='.csv',**kwargs))
    
    with open(fname_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N','diff max'])
        print('dt\t err')
        for i in range(len(dt_list)):
            writer.writerow([dt_list[i],diff_list[i]])
            print(dt_list[i],'\t',diff_list[i])
    
    


def run_U_fixed_dynamics(**kwargs):
    """
    Run simulation for a given set of parameter values
    and generate relevant plots
    """
    # Steady state checks
    #print('============================== U fixed, U='+str(kwargs['U']))
    
    a = mpde(**kwargs)
    #lib.disp_params(a)  # display non-array parameters
    
    #t0 = time.time()
    a.run()
    #t1 = time.time()
    #print('*\t Run time',t1-t0)
    
    initial_mass = np.sum(a.sol[0,:])*a.dx
    mass_true = lib.mass_fn(a.t,initial_mass,**kwargs)
    #lib.disp_norms(a,ground_truth_values)
    
    fig = plt.figure(figsize=(10,5))
    ax11 = fig.add_subplot(121)
    ax12 = fig.add_subplot(122)
    
    mass_pde = np.sum(a.sol,axis=1)*a.dx
    
    ax11.plot(a.t,mass_true,label='mass true')
    ax11.plot(a.t,mass_pde,label='mass pde')
    
    ax12.plot(a.t,np.abs(mass_pde - mass_true),label='|pde-(true)|')
    
    ax11.set_title('mass over time')
    ax12.set_title('mass diff')
    
    ax11.set_xlabel('t')
    ax12.set_xlabel('t')
    

    ax11.legend()
    ax12.legend()
    
    plt.tight_layout()
    
    # include dt
    kwargs = {**kwargs, **{'dt':a.dt}}
    
    fname = (DIR_TESTS
             + 'U_fixed_dynamics_'
             + lib.fname_suffix(**kwargs))

    plt.savefig(fname)
    plt.close()
    
    return np.amax(np.abs(mass_true - mass_pde))


def test_U_variable(**kwargs):
    
    a = mpde(**kwargs)
    
    a.run()
    
    mass = np.sum(a.sol,axis=1)*a.dx
    
    ss_mass = np.zeros(a.TN)
    for i in range(len(a.t)):
        
        if a.U_arr[i] > 0:
            x = a.x[a.x > a.A]
        else:
            x = a.x[a.x <= a.A]
        
        ss_mass[i] = np.sum(lib.phi(x,a.U_arr[i],a))*a.dx
    
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=100, metadata=metadata)
    
    fig = plt.figure(figsize=(8,8))
    
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    image, = ax1.plot([], [])
    image2, = ax1.plot([], [])

    
    # get bounds for velocity function
    maxvel = np.amax(a.U(a.t,vel=kwargs['fn_vel'],
                         option=kwargs['fn_test_option']))
    minvel = np.amin(a.U(a.t,vel=kwargs['fn_vel'],
                         option=kwargs['fn_test_option']))
    
    ax2.plot(a.t[a.t>.05],a.U(a.t,vel=kwargs['fn_vel'],
                              option=kwargs['fn_test_option'])[a.t>.05])
    vel_text = ax2.text(0.05,(maxvel+minvel)/2,'U=%.2f' % a.U(0))
    vel_pos = ax2.scatter(a.t[0],a.U(a.t[0]))
    
    ax2.plot([a.t[0],a.t[-1]],[0,0],ls='--',color='gray')

    ax3.plot(a.t[a.t>.05],mass[a.t>.05])
    ax3.plot(a.t[a.t>.05],ss_mass[a.t>.05],
             ls='--',color='gray',label='ss mass')
    #ax3.plot([a.t[0],a.t[-1]],[],ls='--',color='gray',label='True mass U>0')
    #ax3.plot([a.t[0],a.t[-1]],[],ls='-',color='k',label='True mass U<0')
    start_idx = np.argmin(np.abs(a.t-0.05))
    mass_pos = ax3.scatter(a.t[start_idx],mass[start_idx])
    
    ax1.set_xlim(a.A0,a.B)
    ax1.set_ylim(0,.2)
    
    ax2.set_ylim(minvel,maxvel)

    ax2.set_xlim(0.05,a.T)
    ax3.set_xlim(0.05,a.T)

    
    ax1.set_xlabel('x')
    ax1.set_ylabel(r'$\phi$')
    
    ax2.set_xlabel('t')
    ax2.set_ylabel('U')
    
    ax3.set_ylabel('mass')
    ax3.set_xlabel('t')

    #ax2.set_xlim(.05, a.T)
    #ax3.set_xlim(.05, a.T)



    ax3.legend()
    
    plt.tight_layout()
    
    fname = (DIR_TESTS
             + 'U_exogenous_'
             + lib.fname_suffix(**kwargs))
    print(fname,os.getcwd())
    plt.savefig(fname)
    #plt.close()

    sim_len = len(a.sol[:,0])

    skipn = 1
    
    if True:
        with writer.saving(fig,
                           DIR_TESTS
                           + 'U_exogenous_'
                           + lib.fname_suffix(**kwargs,ftype='.mp4'),dpi=120):
            j = 0
            while j < sim_len/skipn:
                i = j*skipn
                if j%int(sim_len/10/skipn) == 0:
                    print(i/sim_len)
                image.set_data(a.x,a.sol[i,:])

                if a.U_arr[i] > 0:
                    x = a.x[a.x > a.A]
                else:
                    x = a.x[a.x <= a.A]

                image2.set_data(x,lib.phi(x,a.U_arr[i],a))
                #print(np.sum(a.sol[i,:]),np.sum(lib.phi(x,a.U_arr[i],a)))
                ax1.set_ylim(0, .8)

                vel_pos.set_offsets([a.t[i],a.U(a.t[i],option=kwargs['fn_test_option'])])
                mass_pos.set_offsets([a.t[i],mass[i]])
                
                vel_text.set_text('U=%.2f' % a.U(a.dt*i,option=kwargs['fn_test_option']))
                
                writer.grab_frame()

                j += 1


def test_U_dynamics(**kwargs):
    """
    check that convergence holds in dt but dx.
    """
    
    # solver must be euler to accurately save velocity dynamics.
    assert(kwargs['ivp_method'] == 'euler')
    assert(kwargs['store_position'] is True)
    
    a = mpde(**kwargs)
    
    a.run()
    
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=100, metadata=metadata)
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    image, = ax1.plot([],[])
    
    ax2.plot(a.t,a.U_arr)
    vel_pos = ax2.scatter(a.t[0],a.U_arr[0])
    
    ax2.plot([a.t[0],a.t[-1]],[0,0],ls='--',color='gray')
    
    ax1.set_xlim(a.A0,a.B)
    ax1.set_ylim(0,.2)
    
    
    ax1.set_xlabel('x')
    ax1.set_ylabel(r'$\phi$')
    
    ax2.set_xlabel('t')
    ax2.set_ylabel('U')
    
    plt.tight_layout()
    
    if True:
        with writer.saving(fig,"test2.mp4",dpi=100):
            for i in range(len(a.sol[:,0])):
                image.set_data(a.x,a.sol[i,:])
                vel_pos.set_offsets([a.t[i],a.U_arr[i]])
                writer.grab_frame()

    fname = (DIR_TESTS
             + 'U_dynamics_'
             + lib.fname_suffix(**kwargs))

    plt.savefig(fname)
    plt.close()


def test_sampling(quick=True,**kwargs):
    """
    for a given distribution, produce random variables that fit the distribution.
    
    """
    
    
    
    np.random.seed(1)
    
    a = mpde(**kwargs)
    
    if kwargs['irregular']:
        kwargs['dt'] = 0.5*np.amin(a.dx)/np.abs(kwargs['U'])
    else:
        kwargs['dt'] = 0.5*a.dx/np.abs(kwargs['U'])

    a.__init__(**kwargs)
    a.run()
    
    # get sol near start and near end
    pdf1 = a.sol[int(a.TN/10),:]
    pdfx = a.sol[int(a.TN/3),:]
    pdf2 = a.sol[-1,:]
    
    if quick:
        testing_numbers = 1000000
    else:
        testing_numbers = 200000000
    
    rvs1 = lib.inverse_transform_sampling(a,pdf1,testing_numbers)
    rvsx = lib.inverse_transform_sampling(a,pdfx,testing_numbers)
    rvs2 = lib.inverse_transform_sampling(a,pdf2,testing_numbers)
    
    fig = plt.figure(figsize=(8,8))
    
    gs = fig.add_gridspec(3, 2)
    
    ax11 = fig.add_subplot(gs[0,0])
    ax21 = fig.add_subplot(gs[1,0])
    ax31 = fig.add_subplot(gs[2,0])
    
    ax12 = fig.add_subplot(gs[0,1])
    ax22 = fig.add_subplot(gs[1,1])
    ax32 = fig.add_subplot(gs[2,1])
    
    
    # FIRST TIME POINT
    
    
    normalize1 = np.sum(pdf1[:-1]*a.dx)
    pdf1n = pdf1/normalize1
    
    n1, bins1, _ = ax11.hist(rvs1,density=True,bins=200,label='sampled')
    #bins1 = (bins1[:-1]+bins1[1:])/2
    hist1 = interp1d(bins1[1:],n1,fill_value=0.,bounds_error=False)
    ax11.clear()
    ax11.plot(a.x,hist1(a.x),label='sampled')
    ax11.plot(a.x,pdf1n,label='pde (conditioned on attachment)')
    
    #pad = bins1[1]-bins1[0]
    idx_pad = 10
    lower = a.A_idx+idx_pad
    upper = -idx_pad
    #print(idx_pad,bins1[1]-bins1[0],a.dx)
    
    # err
    ax12.plot(a.x[lower:upper],
              hist1(a.x)[lower:upper]-pdf1n[lower:upper])
    ax12.set_title('err excluding 10 indices left and right ends')
    
    # SECOND TIME POINT
    normalizex = np.sum(pdfx[:-1]*a.dx)
    pdfxn = pdfx/normalizex
    nx, binsx, _ = ax21.hist(rvsx,density=True,bins=200,label='sampled')
    #binsx = (binsx[:-1]+binsx[1:])/2
    histx = interp1d(binsx[1:],nx,fill_value=0.,bounds_error=False)
    ax21.clear()
    ax21.plot(a.x,histx(a.x),label='sampled')
    ax21.plot(a.x,pdfxn,label='pde (conditioned on attachment)')
    
    # err
    ax22.plot(a.x[lower:upper],
              histx(a.x)[lower:upper]-pdfxn[lower:upper])
    
    # THIRD TIME POINT
    normalize2 = np.sum(pdf2[:-1]*a.dx)
    pdf2n = pdf2/normalize2
    n2, bins2, _ = ax31.hist(rvs2,density=True,bins=200,label='sampled')
    #bins2 = (bins2[:-1]+bins2[1:])/2
    hist2 = interp1d(bins2[1:],n2,fill_value=0,bounds_error=False)
    ax31.clear()
    ax31.plot(a.x,hist2(a.x),label='sampled')
    ax31.plot(a.x,pdf2n,label='pde (conditioned on attachment)')
    
    # err
    ax32.plot(a.x[lower:upper],
              hist2(a.x)[lower:upper]-pdf2n[lower:upper])
    
    ax11.set_xlabel('x')
    ax21.set_xlabel('x')
    ax31.set_xlabel('x')
    
    ax11.set_ylabel('Distribution')
    ax21.set_ylabel('Distribution')
    ax31.set_ylabel('Distribution')
    
    ax11.legend()
    ax21.legend()
    ax31.legend()
    
    plt.tight_layout()

    #print(a.U)

    if a.U > 0:
        dx_min = np.amin(a.dx)
        ax11.set_xlim(a.A-dx_min*10,a.B+dx_min*10)
        ax21.set_xlim(a.A-dx_min*10,a.B+dx_min*10)
        ax31.set_xlim(a.A-dx_min*10,a.B+dx_min*10)
        
        ax12.set_xlim(a.A-dx_min*10,a.B+dx_min*10)
        ax22.set_xlim(a.A-dx_min*10,a.B+dx_min*10)
        ax32.set_xlim(a.A-dx_min*10,a.B+dx_min*10)
        
    fname = (DIR_TESTS
             + 'test_sampling_'
             + lib.fname_suffix(**kwargs))
    #plt.show(block=True)
    
    plt.savefig(fname)
    plt.close()
    
    
    
    #print(np.amax(np.abs(hist1(a.x)[lower:upper]-pdf1n[lower:upper])))
    #print(np.amax(np.abs(histx(a.x)[lower:upper]-pdfxn[lower:upper])))
    #print(np.amax(np.abs(hist2(a.x)[lower:upper]-pdf2n[lower:upper])))

def test_sample_call(quick=True,**kwargs):
    """
    for a given distribution, produce random variables that fit the distribution.
    
    """
    
    np.random.seed(1)
    
    # get sol near start and near end
    dx = 1

    x = np.arange(0,4,dx)
    
    pdf1 = np.exp(x-10)
    pdfx = x
    pdf2 = x
    
    if quick:
        testing_numbers = 1000000
    else:
        testing_numbers = 200000000
    
    rvs1 = inverse_transform_sampling(x,pdf1,testing_numbers,dx)
    #print(rvs1)
    #rvsx = inverse_transform_sampling(x,pdfx,testing_numbers,dx)
    #rvs2 = inverse_transform_sampling(x,pdf2,testing_numbers,dx)
    
    fig = plt.figure(figsize=(4,8))
    
    gs = fig.add_gridspec(3, 1)
    
    ax11 = fig.add_subplot(gs[0,0])
    ax21 = fig.add_subplot(gs[1,0])
    ax31 = fig.add_subplot(gs[2,0])
    
    # FIRST TIME POINT
    normalize1 = np.sum(pdf1*dx)
    pdf1n = pdf1/normalize1
    
    n1, bins1, _ = ax11.hist(rvs1,density=True,bins=200,label='sampled')
    #bins1 = (bins1[:-1]+bins1[1:])/2
    #hist1 = interp1d(bins1[1:],n1,fill_value=0.,bounds_error=False)
    #ax11.clear()
    #ax11.plot(x,hist1(x),label='sampled')
    ax11.plot(x,pdf1n,label='pde (conditioned on attachment)')
    
    # SECOND TIME POINT
    #normalizex = np.sum(pdfx[:-1]*dx)
    #pdfxn = pdfx/normalizex
    #nx, binsx, _ = ax21.hist(rvsx,density=True,bins=200,label='sampled')
    #binsx = (binsx[:-1]+binsx[1:])/2
    #histx = interp1d(binsx[1:],nx,fill_value=0.,bounds_error=False)
    #ax21.clear()
    #ax21.plot(x,histx(x),label='sampled')
    #ax21.plot(x,pdfxn,label='pde (conditioned on attachment)')
    
    
    ax11.set_xlabel('x')
    ax21.set_xlabel('x')
    ax31.set_xlabel('x')
    
    ax11.set_ylabel('Distribution')
    ax21.set_ylabel('Distribution')
    ax31.set_ylabel('Distribution')
    
    ax11.legend()
    ax21.legend()
    ax31.legend()
    
    plt.tight_layout()

    #print(a.U)

    if kwargs['U'] > 0:
        dx_min = np.amin(a.dx)
        ax11.set_xlim(a.A-dx_min*10,a.B+dx_min*10)
        ax21.set_xlim(a.A-dx_min*10,a.B+dx_min*10)
        ax31.set_xlim(a.A-dx_min*10,a.B+dx_min*10)
        
        ax12.set_xlim(a.A-dx_min*10,a.B+dx_min*10)
        ax22.set_xlim(a.A-dx_min*10,a.B+dx_min*10)
        ax32.set_xlim(a.A-dx_min*10,a.B+dx_min*10)
        
    fname = (DIR_TESTS
             + 'test_sampling_'
             + lib.fname_suffix(**kwargs))
    #plt.show(block=True)
    
    plt.savefig(fname)
    plt.close()
    
    
    
    #print(np.amax(np.abs(hist1(a.x)[lower:upper]-pdf1n[lower:upper])))
    #print(np.amax(np.abs(histx(a.x)[lower:upper]-pdfxn[lower:upper])))
    #print(np.amax(np.abs(hist2(a.x)[lower:upper]-pdf2n[lower:upper])))


    
def test_source(show=False,**kwargs):

    #print(kwargs)
    a = mpde(**kwargs)  # initialize
    #lib.disp_params(a)  # display non-array parameters
    
    #t0 = time.time()
    a.run()  # run
    #t1 = time.time()
    
    #sig = kwargs['init_pars']['pars']
    ground_truth = np.zeros(len(a.x))
    #shifted_x = kwargs['U']*a.TN*a.dt - (a.x-a.A_idx*a.dx)
    shifted_x = kwargs['U']*kwargs['T'] - (a.x-a.A)
    
    #shifted_idx = a.A_idx*a.dx
    nonzero_idxs = (a.x-a.A < kwargs['dt']*a.TN)*(a.x-a.A >= 0)
    first_nonzero = np.where(nonzero_idxs==1)[0][0]
    last_nonzero = np.where(nonzero_idxs==1)[0][-1]

    nonzero_idxs[first_nonzero[0]] = 0
    ground_truth[nonzero_idxs] = source_fn(shifted_x[nonzero_idxs])
    
    fig = plt.figure(figsize=(10,5))
    ax11 = fig.add_subplot(221)
    ax12 = fig.add_subplot(222)
    ax21 = fig.add_subplot(223)
    ax22 = fig.add_subplot(224)
    
    ax11.plot(a.sol[-1,:],label='pde final')
    ax11.plot(ground_truth,label='ground truth final')
    ax11.scatter(a.A_idx,0,label='x=A index')
    
    ax12.plot(a.sol[-1,:],label='pde final')
    ax12.plot(ground_truth,label='ground truth final')
    ax12.scatter(a.A_idx,0,label='x=A index')
    
    
    abs_diff = np.abs(a.sol[-1,:]-ground_truth)
    ax21.plot(abs_diff,label='||pde-(true)||\_1')
    ax22.plot(abs_diff,label='||pde-(true)||\_1')
    
    ax12.set_xlim(a.A_idx-10,a.A_idx+10)
    ax22.set_xlim(a.A_idx-10,a.A_idx+10)
    
    
    ax11.set_title('full')    
    ax12.set_title('zoom at A\_idx')
    
    ax21.set_title('diff')
    ax22.set_title('diff zoom  at A\_idx')

    
    #ax11.set_xlabel('x index')
    ax11.set_xlabel('x index')
    ax12.set_xlabel('x index')
    ax21.set_xlabel('x index')
    ax22.set_xlabel('x index')

    ax11.legend()
    ax12.legend()
    ax21.legend()
    ax22.legend()

    
    plt.tight_layout()
    
    # a.plot(figure='final-diff')
    fname = DIR_TESTS+'source_'+lib.fname_suffix(ftype='.png',**kwargs)
    #print('*\t saving to ',fname)
    
    plt.savefig(fname)
    
    # diff in mass init vs final
    diff_mass = np.abs((np.sum(a.init) - np.sum(a.sol[-1,:])))*a.dx
        
    # diff in final function vs ground truth
    diff_final = np.sum(np.abs(a.sol[-1,:]-ground_truth))*a.dx
    
    print(diff_final)
    
    if show:
        plt.show(block=True)
    
    plt.close()
    
    return diff_mass, diff_final
    

def source_fn(t):
    return np.sin(t)

def fn_test(t,vel=10,om=20,option='root_cos'):
    #print(np.cos(om*t))
    #print(np.cos(om*t)**(1/3))

    if option == 'root_cos':
        sign = np.sign(np.cos(om*t))
    
        return vel*sign*np.abs(np.cos(om*t))**(1/10)

    elif option == 'square':
        return vel*square(om*t)

    elif option == 'sawtooth':
        return np.mod(om*t,1)

def main():
    
    quick_test = True
    
    if quick_test:
        # for fast debugging
        Nlist = 6**np.arange(3,5,1)
        Nlist2 = Nlist
        dt_list = 5.**np.arange(-4,-8,-1)
    else:
        # test to machine epsilon
        Nlist = 10**np.arange(2,5,1)
        Nlist2 = 10**np.arange(3,6,1)
        dt_list = 10.**np.arange(-4,-10,-1)
    
    # Tests where ground truth is known. set if True to run.
    if False:
        # CONSERVATION CHECK with Gaussian init
        opt = {'U':100, 'A0':0, 'A':.25, 'B':0.5,
               'T':0.001,'beta':0,'N2':1000,'irregular':True,
               'ivp_method':'euler','store_position':True,
               'source':False,'init_pars':{'type':'gaussian','pars':0.01}}
        
        test_conservation(Nlist=5**np.arange(3,5,1),**opt)

    if False:
        # FLUX CHECK with Gaussian init
        opt = {'U':200, 'A0':0, 'A':0, 'B':0.5,
               'T':0.002,'beta':0,
               'ivp_method':'euler','store_position':True,
               'source':False,'init_pars':{'type':'gaussian','pars':0.01}}
        test_flux(Nlist=5**np.arange(2,5,1),**opt)
        
    if False:
        # FULL CHECK SS with source (U>0)
        # iterate over N2 since N2 is the mesh for z > A
        opt = {'U':121, 'A0':4.95, 'A':5, 'B':5.05,
                  'T':0.1,'alpha':14,'beta':126,'irregular':True,
                  'ivp_method':'euler','store_position':False}
        
        test_u_fixed_ss(Nlist=[50,200],**opt)
    
    if False:
        # FULL CHECK SS with source (U<0)
        opt = {'U':-121, 'A0':0, 'A':5, 'B':5.05,
               'alpha':14,'beta':126,'T':0.1,
               'N':100,'N2':100,'irregular':True,
               'ivp_method':'euler','store_position':False}
        
        test_u_fixed_ss(Nlist=[50,200],**opt)
    
    
    if False:
        # FULL CHECK SS with source (U>0)
        opt = {'U':64, 'A0':0, 'A':5, 'B':5.1,
                  'alpha':14,'beta':126,'T':0.2,
                  'irregular':False,
                  'ivp_method':'euler','store_position':False}
        
        test_u_fixed_ss(Nlist=10**np.arange(1,4),**opt)
        
    if False:
        # FULL CHECK SS with source (U<0)
        opt = {'U':-10,'B':5.1,
                  'alpha':14,'beta':126,'T':0.2,
                  'irregular':False,
                  'ivp_method':'euler','store_position':False}
        
        test_u_fixed_ss(Nlist=10**np.arange(1,5),**opt)

        
    if False:
        # FULL CHECK DYNAMIC + FIXED VELOCITY
        opt = {'U':-10, 'A0':0, 'A':5, 'B':5.01,
                  'T':0.02,'beta':126,'alpha':14,
                  'ivp_method':'euler','store_position':True}
        
        test_U_fixed_dynamics(dt_list=dt_list,**opt)
        
    # Tests where ground truth is not known
    
    if True:
        # VARIABLE (EXOGENOUS) VELOCITY CHECK
        # FULL CHECK (U<0)
        opt = {'U':fn_test,'fn_test_option':'square', 'fn_vel':30,
                  'A0':0, 'A':5, 'B':5.1,
                  'T':1,'beta':126,'alpha':14,'N':200,
                  'ivp_method':'euler','store_position':True}
        
        t = np.linspace(0,opt['T'],1000)
        max_U = np.amax(np.abs(fn_test(t,option=opt['fn_test_option'])))
        CFL = 0.25
        dx = (opt['B']-opt['A0'])/opt['N']
        opt['dt'] = CFL*dx/np.abs(max_U)
        
        test_U_variable(**opt,exclude=['U'])
    
    if False:
        # RV POSITION CHECK
        # NOT YET IMPLEMENTED.
        
        opt = {'U':'dynamic', 'A0':-2, 'A':5, 'B':5.5,
               'T':.1,'beta':126,'alpha':14,'N':200,
               'ivp_method':'euler','store_position':True}
        
        max_U = 100
        CFL = 0.5
        dx = (opt['B']-opt['A0'])/opt['N']
        opt['dt'] = CFL*dx/np.abs(max_U)
        
        test_U_dynamics(**opt)

    if False:
        # test inverse sampling
        
        opt = {'U':100, 'A0':0, 'A':5, 'B':5.5,
               'T':.01,'beta':126,'alpha':14,'N':500,'N2':500,
               'irregular':True,
               'ivp_method':'euler','store_position':True}
        
        CFL = 0.5
        dx = (opt['B']-opt['A0'])/opt['N']
        opt['dt'] = CFL*dx/np.abs(opt['U'])
        
        #test_sampling(**opt)
        
        opt['U'] = -100
        test_sampling(**opt)

        test_sample_call(**opt)
        
        
    if False:
        # test inverse sampling
        
        opt = {'U':-54, 'A0':-2, 'A':5, 'B':5.5,
                  'T':.02,'beta':150,'alpha':10,'N':100,
                  'margin':20,'N2':100,'irregular':True,
                  'ivp_method':'euler','store_position':True}
        
        CFL = 0.5
        dx = (opt['B']-opt['A0'])/opt['N']
        opt['dt'] = CFL*dx/np.abs(opt['U'])
        
        test_sampling(**opt)
        
        opt['U'] = -100
        test_sampling(**opt)

    if False:
        
        # test delta function source
        # Discretizing singular point sources in hyperbolic wave
        # propagation problems Petersson et al 2007
        
        # call upwinding scheme with custom source
        
        opt = {'U':1, 'A0':0, 'A':5, 'B':10,
                  'T':3,'beta':0,'alpha':0,'N':50000,
               'irregular':True,
               'N2':50000,
                  'ivp_method':'euler','store_position':True,
                  'init_pars':None,'source':source_fn,
                  'regularized':False}
        
        CFL = 0.5
        #dx = (opt['B']-opt['A0'])/opt['N']
        N = np.max([opt['N'],opt['N2']])
        _,dx = np.linspace(opt['A0'],opt['B'],N,
                           endpoint=False,retstep=True)

        opt['dt'] = CFL*dx/np.abs(opt['U'])

        
        test_source(**opt)
        
        
    if False:
        
        
        # test delta function source with irregular mesh
        # Discretizing singular point sources in hyperbolic wave
        # propagation problems Petersson et al 2007
        
        # call upwinding scheme with custom source
                
        opt = {'U':1, 'A0':0, 'A':5, 'B':10,
                  'T':3,'beta':0,'alpha':0,'N':1000,
                  'ivp_method':'RK45','store_position':True,
                  'init_pars':None,'source':source_fn,
                  'regularized':False,
                  'irregular':True}
        
        CFL = 0.5
        #dx = (opt['B']-opt['A0'])/opt['N']
        _,dx = np.linspace(opt['A0'],opt['B'],opt['N'],
                           endpoint=False,retstep=True)

        opt['dt'] = CFL*dx/np.abs(opt['U'])

        
        test_source(**opt)



if __name__ == "__main__":
    main()

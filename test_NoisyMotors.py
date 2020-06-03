# -*- coding: utf-8 -*-
"""
import and test pde2.py
"""

from NoisyMotors import NoisyMotors as nm
import libNoisyMotors as lib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

import csv
import os
#import time
#import inspect

import matplotlib
matplotlib.use("Agg")


# user modules


# create test directory for convergence results and plots
DIR_TESTS = 'tests/'

if not(os.path.exists(DIR_TESTS)):
    os.makedirs(DIR_TESTS)



def test_conservation(CFL=0.5,Nlist=10**np.arange(2,5,1),**kwargs):
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
    
    for N in Nlist:
        kwargs['N'] = N
        dx = (kwargs['B']-kwargs['A0'])/kwargs['N']
        kwargs['dt'] = CFL*dx/np.abs(kwargs['U'])
        
        diff_mass, diff_final = run_conservation(show=False,**kwargs)
        
        diff_mass_list.append(diff_mass)
        diff_final_list.append(diff_final)
        
    # save to csv file
    fname_csv = DIR_TESTS+'conservation_'+lib.fname_suffix(exclude=['N'],ftype='.csv',**kwargs)
    with open(fname_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N','diff mass', 'diff final (max)'])
        
        print('N\t diff mass\t diff final (max err)')
        for i in range(len(Nlist)):
            writer.writerow([Nlist[i],diff_mass_list[i],diff_final_list[i]])
            print(Nlist[i],'\t',diff_mass_list[i],'\t',diff_final_list[i])
    
    

def run_conservation(show=False,**kwargs):
    """
    Run simulation for a given set of parameter values
    and generate relevant plots
    """

    #print(kwargs)
    a = nm(**kwargs)  # initialize
    #lib.disp_params(a)  # display non-array parameters
    
    #t0 = time.time()
    a.run()  # run
    #t1 = time.time()
    
    sig = kwargs['init_pars']['pars']
    shifted_domain = a.x-(a.A0+a.B)/2-kwargs['U']*kwargs['T']
    ground_truth_values = lib.gauss(shifted_domain,sig=sig)
    #lib.disp_norms(a,ground_truth_values)
    
    fig = plt.figure(figsize=(10,5))
    ax11 = fig.add_subplot(221)
    ax12 = fig.add_subplot(222)
    ax21 = fig.add_subplot(223)
    ax22 = fig.add_subplot(224)
    
    ax11.plot(a.init,label='init')
    ax12.plot(a.sol[-1,:],label='pde')
    ax12.plot(ground_truth_values,label='ground truth')

    
    if kwargs['use_storage']:
        # get total mass per time
        #print(np.shape(a.sol))
        mass = np.sum(a.sol,axis=1)*a.dx
        #print(np.shape(mass))
        ax21.plot(mass,label='mass')
    
        ax21.set_title('Mass')
        ax21.set_xlabel('t index')
        #ax21.set_title('Timecourse at specific points')
    
    abs_diff = np.abs(a.sol[-1,:]-ground_truth_values)
    ax22.plot(a.x,abs_diff,label='||pde-(true)||_1')
    
    #print('max diff',np.amax(np.abs(a.sol[-1,:]-ground_truth_values)))

    
    ax11.scatter(a.A_idx,0,label='x=A index')
    ax12.scatter(a.A_idx,0,label='x=A index')
    
    
    ax12.set_title('initial')
    ax12.set_title('final')
    
    ax22.set_title('final-(true)')

    #ax11.set_xlabel('x index')
    ax11.set_xlabel('t index')
    ax12.set_xlabel('x index')

    ax11.legend()
    ax12.legend()
    ax21.legend()
    ax22.legend()
    
    plt.tight_layout()
    
    # a.plot(figure='final-diff')
    fname = DIR_TESTS+'conservation_'+lib.fname_suffix(ftype='.png',**kwargs)
    #print('*\t saving to ',fname)
    
    plt.savefig(fname)
    
    # diff in mass init vs final
    diff_mass = np.abs((np.sum(a.init) - np.sum(a.sol[-1,:])))*a.dx
        
    # diff in final function vs ground truth
    diff_final = np.amax(np.abs(a.sol[-1,:]-ground_truth_values))
    
    if show:
        plt.show(block=True)
    
    plt.close()
    
    return diff_mass, diff_final



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

    assert(kwargs['use_storage'] is True)
    
    a = nm(**kwargs)
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
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(cumulative_flux)
    
    fname = DIR_TESTS+'flux_'+lib.fname_suffix(ftype='.png',**kwargs)
    #print('*\t saving plot to ',fname)
    
    plt.savefig(fname)
    
    if show:
        plt.show(block=True)
    
    plt.close()
    
    return cumulative_flux


def test_U_fixed_ss(CFL=0.5,Nlist=10**np.arange(2,5,1),**kwargs):
    
    print()
    print('Testing Fixed U='+str(kwargs['U']))
    print(kwargs)
    
    diff_list = []
    diff_list = []
    
    for N in Nlist:
        kwargs['N'] = N
        dx = (kwargs['B']-kwargs['A0'])/kwargs['N']
        kwargs['dt'] = CFL*dx/np.abs(kwargs['U'])
        
        run_U_fixed_ss(show=False,**kwargs)
        diff = run_U_fixed_ss(show=False,**kwargs)
        
        diff_list.append(diff)
        
    # save to csv file
    fname_csv = (DIR_TESTS+'U_fixed_ss'
                 + lib.fname_suffix(exclude=['N'],ftype='.csv',**kwargs))
    
    with open(fname_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N','diff max'])
        print('N\t err')
        for i in range(len(Nlist)):
            writer.writerow([Nlist[i],diff_list[i]])
            print(Nlist[i],'\t',diff_list[i])
    
    

def run_U_fixed_ss(**kwargs):
    """
    Run simulation for a given set of parameter values
    and generate relevant plots
    """
    # Steady state checks
    #print('============================== U fixed, U='+str(kwargs['U']))
    
    a = nm(**kwargs)
    #lib.disp_params(a)  # display non-array parameters
    
    #t0 = time.time()
    a.run()
    #t1 = time.time()
    #print('*\t Run time',t1-t0)
    
    ground_truth_values = lib.ground_truth(a)
    #lib.disp_norms(a,ground_truth_values)
    
    fig = plt.figure(figsize=(10,5))
    ax11 = fig.add_subplot(121)
    ax12 = fig.add_subplot(122)
    
    ax11.plot(a.x,np.abs(a.sol[-1,:]-ground_truth_values),label='|pde-(true)|')
    
    ax12.plot(a.x,a.sol[-1,:],label='pde')
    ax12.plot(a.x,ground_truth_values,label='ground truth')
    
    ax12.scatter(a.A_idx,0,label='x=A index')
    ax11.scatter(a.A_idx,0,label='x=A index')
    
    ax12.set_title('final')
    ax11.set_title('final-(true)')

    #ax11.set_xlabel('x index')
    ax11.set_xlabel('t')
    ax12.set_xlabel('x')
    
    pad = (a.B-a.A)/10
    ax11.set_xlim(a.A-pad,a.B+pad)
    ax12.set_xlim(a.A-pad,a.B+pad)


    ax11.legend()
    ax12.legend()
    
    plt.tight_layout()
    
    # a.plot(figure='final-diff')
    fname = (DIR_TESTS
             + 'U_fixed_ss_'
             + lib.fname_suffix(**kwargs))
    plt.savefig(fname)
    plt.close()
    
    return np.amax(np.abs(a.sol[-1,:]-ground_truth_values))


def test_U_fixed_dynamic(CFL=0.5,Nlist=10**np.arange(2,5,1),**kwargs):
    
    print()
    print('Testing Fixed U='+str(kwargs['U']))
    print(kwargs)
    
    diff_list = []
    diff_list = []
    
    for N in Nlist:
        kwargs['N'] = N
        dx = (kwargs['B']-kwargs['A0'])/kwargs['N']
        kwargs['dt'] = CFL*dx/np.abs(kwargs['U'])
        
        run_U_fixed_dynamic(show=False,**kwargs)
        diff = run_U_fixed_dynamic(show=False,**kwargs)
        
        diff_list.append(diff)
        
    # save to csv file
    fname_csv = (DIR_TESTS+'U_fixed_dynamics'
                 + lib.fname_suffix(exclude=['N'],ftype='.csv',**kwargs))
    
    with open(fname_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N','diff max'])
        print('N\t err')
        for i in range(len(Nlist)):
            writer.writerow([Nlist[i],diff_list[i]])
            print(Nlist[i],'\t',diff_list[i])
    
    


def run_U_fixed_dynamic(**kwargs):
    """
    Run simulation for a given set of parameter values
    and generate relevant plots
    """
    # Steady state checks
    #print('============================== U fixed, U='+str(kwargs['U']))
    
    a = nm(**kwargs)
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
    
    fname = (DIR_TESTS
             + 'U_fixed_dynamics_'
             + lib.fname_suffix(**kwargs))

    plt.savefig(fname)
    plt.close()
    
    return np.amax(np.abs(mass_true - mass_pde))


def test_U_variable(**kwargs):
    
    a = nm(**kwargs)
    
    a.run()
        
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=100, metadata=metadata)
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    
    image, = ax1.plot([],[])
    
    # get bounds for velocity function
    maxvel = np.amax(a.U(a.t))
    minvel = np.amin(a.U(a.t))
    
    ax2.plot(a.t,a.U(a.t))
    vel_text = ax2.text(0,(maxvel+minvel)/2,'U=%.2f' % a.U(0))
    vel_pos = ax2.scatter(a.t[0],a.U(a.t[0]))
    
    ax2.plot([a.t[0],a.t[-1]],[0,0],ls='--',color='gray')
    
    ax1.set_xlim(a.A0,a.B)
    ax1.set_ylim(0,.2)
    
    ax2.set_ylim(minvel,maxvel)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel(r'$\phi$')
    
    ax2.set_xlabel('t')
    ax2.set_ylabel('U')
    
    plt.tight_layout()
    
    with writer.saving(fig,"test.mp4",dpi=100):
        for i in range(len(a.sol[:,0])):
            image.set_data(a.x,a.sol[i,:])
            vel_pos.set_offsets([a.t[i],a.U(a.t[i])])
            
            vel_text.set_text('U=%.2f' % a.U(a.dt*i))
            
            writer.grab_frame()


def fn_test(t,om=10):
    #print(np.cos(om*t))
    #print(np.cos(om*t)**(1/3))
    
    sign = np.sign(np.cos(om*t))
    
    return 100*sign*np.abs(np.cos(om*t))**(1/10)


def main():
    
    # Tests where ground truth is known. set if True to run.
    if False:
        # CONSERVATION CHECK
        kwargs = {'U':100, 'A0':0, 'A':0, 'B':0.5,
                  'T':0.002,'beta':0,
                  'ivp_method':'euler','use_storage':True,
                  'source':False,'init_pars':{'type':'gaussian','pars':0.01}}
        
        test_conservation(Nlist=5**np.arange(2,5,1),**kwargs)
        
        # FLUX CHECK 
        kwargs = {'U':200, 'A0':0, 'A':0, 'B':0.5,
                  'T':0.002,'beta':0,
                  'ivp_method':'euler','use_storage':True,
                  'source':False,'init_pars':{'type':'gaussian','pars':0.01}}
        
        test_flux(Nlist=5**np.arange(2,5,1),**kwargs)
        
        # FULL CHECK SS (U>0)
        kwargs = {'U':100, 'A0':4.95, 'A':5, 'B':5.5,
                  'T':0.1,'beta':126,
                  'ivp_method':'euler','use_storage':False}
        
        test_U_fixed_ss(Nlist=5**np.arange(2,5,1),**kwargs)
    
        
        # FULL CHECK SS (U<0)
        kwargs = {'U':-100, 'A0':-10, 'A':5, 'B':5.5,
                  'T':0.1,'beta':126,
                  'ivp_method':'euler','use_storage':False}
        
        test_U_fixed_ss(Nlist=5**np.arange(2,5,1),**kwargs)
    
        # FULL CHECK DYNAMIC + FIXED VELOCITY
        kwargs = {'U':-100, 'A0':-10, 'A':5, 'B':5.5,
                  'T':0.1,'beta':126,'alpha':14,
                  'ivp_method':'euler','use_storage':True}
        
        test_U_fixed_dynamic(Nlist=5**np.arange(2,5,1),**kwargs)
        
    # Tests where ground truth is not known
    if True:
        # VARIABLE VELOCITY CHECK
        # FULL CHECK (U<0)
        kwargs = {'U':fn_test, 'A0':-2, 'A':5, 'B':5.5,
                  'T':1,'beta':126,'alpha':14,'N':200,
                  'ivp_method':'RK45','use_storage':True}
        

        t = np.linspace(0,kwargs['T'],100)
        
        max_U = np.amax(np.abs(fn_test(t)))
        CFL = 0.5
        dx = (kwargs['B']-kwargs['A0'])/kwargs['N']
        kwargs['dt'] = CFL*dx/np.abs(max_U)
        
        test_U_variable(Nlist=5**np.arange(2,5,1),**kwargs)
    
    
if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
test functions for agents
"""



#DIR = 'D:/Dropbox/thomas-youngmin/vesicles/yptgf2/code_and_data/'



import argparse

import os
import time,datetime
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import uniform
from math import log10, floor
#from lubrication import lubrication as LB
#from Master import Master as M
import lib.libMotorPDE as lib
from Master import Master
from agents import Agents
import parsets as pset

from test_MotorPDE import fn_test


import matplotlib.animation as manimation
from scipy.signal import square

import matplotlib as mpl
#mpl.rcParams['text.usetex'] = False
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#mpl.use("Agg")


import platform
from pathlib import Path

# operating system
sys = platform.system()

if sys  == 'Darwin':
    home = str(Path.home())
    DIR = home+'/tests/'
    #print(os.path.isdir(DIR))
elif sys == 'Windows':
    DIR = 'C:/tests/'
elif sys == 'Linux':
    home = str(Path.home())
    DIR = home+'/tests/'

if not(os.path.isdir(DIR)):
    print('Missing tests folder at', DIR)
    input('Press enter to create and proceed')
    os.mkdir(DIR)
    
pi = np.pi
exp = np.exp
sqrt = np.sqrt

opt = {'T':1,
       'V0':0,
       'dt':1e-5,
       'seed':9,
       'U':None,
       'force_pos_type':'lin',
       'store_position':False,
       'ext':True}

opt = dict(opt,**pset.parset('figure1'))


def steady_state(v,update_kwargs={},return_data=False,**kwargs):
    """
    compute steady-state distribution given velocity v.
    """

    for key in update_kwargs:
        kwargs[key] = update_kwargs[key]

    kwargs['store_position'] = True
    kwargs['U'] = v

    a = Agents(**kwargs)
    start_time = time.time()
    a.run_agents()
    current_time = time.time()
    total_time = current_time - start_time
    print('total sim time', total_time)


    fig_pos_distrib = plt.figure(figsize=(6, 3))

    ax_X = fig_pos_distrib.add_subplot(211)
    ax_Y = fig_pos_distrib.add_subplot(212)

    # ignore first 1/5th of data
    X1 = a.X1[int(a.TN / 5):, :]
    X2 = a.X2[int(a.TN / 5):, :]

    # rearrange all possible motor positions
    X_motor_positions = np.reshape(X1, len(X1[:, 0]) * len(X1[0, :]))
    Y_motor_positions = np.reshape(X2, len(X2[:, 0]) * len(X2[0, :]))

    # skip x=A for now because it could go either way
    X = X_motor_positions[(X_motor_positions < a.A)
                          * (X_motor_positions != 0)
                          + (X_motor_positions > a.A)]
    # X_above_A = X_motor_positions[X_motor_positions>a.A]

    Y = Y_motor_positions[(Y_motor_positions < a.A)
                          * (Y_motor_positions != 0)
                          + (Y_motor_positions > a.A)]
    # Y_above_A = Y_motor_positions[Y_motor_positions>a.A]

    # total attached motor ratios
    UX = len(X) / len(X_motor_positions != 0)  # np.sum(X_below_A_attached_bool)
    UY = len(Y) / len(Y_motor_positions != 0)  # np.sum(Y_below_A_attached_bool)

    print('UX', UX)
    print('UY', UY)


    """
    if np.amax(X)>(a.A+(a.B-a.A)/2.):
        x = np.linspace(a.A,a.B,100)
        ax_X.plot(x,a.u_non_pref._pdf(x,a.A,a.B,a.alpha,a.beta,np.abs(a.V0)))
    else:
        x = np.linspace(np.amin(X),a.A,100)
        ax_X.plot(x,a.u_pref._pdf(x,a.A,a.B,a.alpha,a.beta,np.abs(a.V0)))

    if np.amax(Y)>(a.A+(a.B-a.A)/2.):
        x = np.linspace(a.A,a.B,100)
        ax_Y.plot(x,a.u_non_pref._pdf(x,a.A,a.B,a.alpha,a.beta,np.abs(a.V0)))
    else:
        x = np.linspace(np.amin(Y),a.A,100)
        ax_Y.plot(x,a.u_pref._pdf(x,a.A,a.B,a.alpha,a.beta,np.abs(a.V0)))
    """

    

    kwargs['A0'] = -6
    a2 = Master(N=2000,**kwargs)

    ground_truth_values = lib.ground_truth(a2)

    # get ground truth for other direction
    a2.U = -a2.U
    ground_truth_values2 = lib.ground_truth(a2)

    
    # generate histogram data
    X_hist, X_bins = np.histogram(X, bins=50)

    X_freq = UX * X_hist / sum(np.diff(X_bins) * X_hist)
    
    Y_hist, Y_bins = np.histogram(Y, bins=10)
    Y_freq = UY * Y_hist / sum(np.diff(Y_bins) * Y_hist)
    
    

    if return_data:
        pack = {'x':a2.x, 'trueX':ground_truth_values,
                'trueY':ground_truth_values2,
                'X_bins':X_bins,
                'X_freq':X_freq,
                'Y_bins':Y_bins,
                'Y_freq':Y_freq}
        
        return pack
    
    else:

        # plot
        X_hist, X_bins, _ = ax_X.hist(X_bins[:-1],
                                      X_bins,
                                      weights=X_freq)

        Y_hist, Y_bins, _ = ax_Y.hist(Y_bins[:-1], Y_bins, weights=Y_freq)

        
        ax_X.plot(a2.x,ground_truth_values)
        ax_Y.plot(a2.x,ground_truth_values2)

        print("area under histogram X", sum(np.diff(X_bins) * X_hist))
        print("area under histogram Y", sum(np.diff(Y_bins) * Y_hist))

        ax_X.set_xlim(4.97,5.12)
        ax_Y.set_xlim(-6,5.05)

        plt.savefig(DIR+'agents_steady_state'
                    + lib.fname_suffix(exclude=['V0','force_pos_type'],**kwargs))


def mean_force(kwargs):
    a = Agents(**kwargs)
    a.run_agents()

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(a.force_at_numberX[a.force_at_numberX > 0])
    ax2.plot(a.force_at_numberY[a.force_at_numberY > 0])

    print('mean force at number X',
          np.nanmean(a.force_at_numberX[a.force_at_numberX > 0]))
    print('mean force at number Y',
          np.nanmean(a.force_at_numberY[a.force_at_numberY > 0]))
    plt.savefig(DIR+'temp_figures/mean_force_agents.png')


def mean_pos(kwargs):
    kwargs['store_position'] = True
    a = Agents(**kwargs)
    a.run_agents()


    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(122)

    t = np.linspace(0, a.T, a.TN)

    # ax1.plot(t[::skipn],a.U1[::skipn],alpha=.4)
    # ax1.plot(t[::skipn],a.U2[::skipn],alpha=.4)

    meanposX = np.zeros(a.TN)
    meanposY = np.zeros(a.TN)

    for i in range(a.TN):
        positionsX = a.X1[i, :]
        positionsY = a.X2[i, :]

        meanposX[i] = np.nanmean(positionsX[positionsX > 0])
        meanposY[i] = np.nanmean(positionsY[positionsY > 0])

    ax1.plot(t, meanposX, alpha=.4, label='X')
    ax1.plot(t, meanposY, alpha=.4, label='Y')

    ax1.legend()

    plt.savefig(DIR + 'temp_figures/mean_pos_agents.png')


def velocity(kwargs):
    kwargs['store_position'] = True
    a = Agents(**kwargs)

    start_time = time.time()
    a.run_agents()
    current_time = time.time()
    total_time = current_time - start_time
    print('total sim time', total_time)    

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(122)

    t = np.linspace(0, a.T, a.TN)


    ax1.plot(t, a.V, alpha=.4)

    # plot mean velocity lines
    ax1.plot([t[0], t[-1]], [a.switch_v, a.switch_v], color='gray')
    ax1.plot([t[0], t[-1]], [-a.switch_v, -a.switch_v], color='gray')

    #ax1.legend()

    plt.savefig(DIR + 'temp_figures/agents_velocity.png')


def generate_pos_mov(kwargs):
    kwargs['store_position'] = False
    kwargs['store_velocity'] = True

    a = Agents(**kwargs)
    a.run_agents()

    fig = plt.figure(figsize=(3, 6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    # ax2 = fig.add_subplot(122)

    t = np.linspace(0, a.T, a.TN)
    plot_counter = 0
    for i in range(a.TN):
        if i % int((a.TN) / 1000) == 0:
            print(round(i / a.TN,4))

            positionsX = a.X1[i, a.X1[i, :] != 0]
            positionsY = a.X2[i, a.X2[i, :] != 0]

            positionsX = np.append(positionsX, 0)
            # positionsX = np.append(positionsX,5.1)

            positionsY = np.append(positionsY, 0)
            # positionsY = np.append(positionsY,5.1)

            ax1.hist(positionsX, bins=200, density=False)
            ax2.hist(positionsY, bins=200, density=False)

            time_label = "{:02.3f}".format(t[i])
            v_label = "{:02.3f}".format(a.V[i])
            x_label = "{:02d}".format(int(a.U1[i]))
            y_label = "{:02d}".format(int(a.U2[i]))

            ax1.set_title('agent-based model. t=' + time_label
                          + '; v=' + v_label
                          + '; X=' + x_label
                          + '; Y=' + y_label)

            ax1.set_ylabel('Position Count X')

            ax2.set_xlabel('Attached Motor Head Position')
            ax2.set_ylabel('Position Count Y')

            ax1.set_xlim(1, 5.1)
            ax1.set_ylim(0, 5)

            ax2.set_xlim(1, 5.1)
            ax2.set_ylim(0, 5)

            plt.savefig('C:/temp2/mov_agents/' + str(plot_counter)
                        + '.png', dpi=100)
            ax1.clear()
            ax2.clear()
            plot_counter += 1



def test_u_variable(update_kwargs,kwargs):

    for key in update_kwargs:
        kwargs[key] = update_kwargs[key]

    kwargs['store_position'] = True

    a = Agents(**kwargs)
    a.run_agents()



    # proportion attached at each time step
    thetaX = a.U1/a.nX
    #thetaY = a.U2/a.nY

    print(np.shape(a.X1))

    #massX = np.sum(a.X1, axis=1) * a.dx1
    massX = thetaX

    # domain
    domain,dx = np.linspace(-10,a.B,1000,retstep=True)

    ss_mass = np.zeros(a.TN)
    for i in range(a.TN):
        

        if a.V[i] > 0:
            x = domain[domain > a.A]
        else:
            x = domain[domain <= a.A]

        ss_mass[i] = np.sum(lib.phi(x, a.V[i], a)) * dx



    fig = plt.figure(figsize=(8, 8))

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    image, = ax1.plot([], [])
    image2, = ax1.plot([], [])

    # get bounds for velocity function
    maxvel = np.amax(a.U(a.t, option=kwargs['fn_test_option']))
    minvel = np.amin(a.U(a.t, option=kwargs['fn_test_option']))
    
    t0 = 0.0

    ax2.plot(a.t[a.t > t0],
             a.U(a.t, option=kwargs['fn_test_option'])[a.t > t0])
    vel_text = ax2.text(t0, (maxvel + minvel) / 2, 'U=%.2f' % a.U(0))
    vel_pos = ax2.scatter(a.t[0], a.U(a.t[0]))

    ax2.plot([a.t[0], a.t[-1]], [0, 0], ls='--', color='gray')

    ax3.plot(a.t[a.t > t0], massX[a.t > t0])
    ax3.plot(a.t[a.t > t0], ss_mass[a.t > t0],
             ls='--', color='gray', label='ss mass')
    # ax3.plot([a.t[0],a.t[-1]],[],ls='--',color='gray',label='True mass U>0')
    # ax3.plot([a.t[0],a.t[-1]],[],ls='-',color='k',label='True mass U<0')
    start_idx = np.argmin(np.abs(a.t - t0))
    mass_pos = ax3.scatter(a.t[start_idx], massX[start_idx])

    ax1.set_xlim(0, a.B)
    ax1.set_ylim(0, .2)

    ax2.set_ylim(minvel, maxvel)

    ax2.set_xlim(t0, a.T)
    ax3.set_xlim(t0, a.T)

    ax1.set_xlabel('x')
    ax1.set_ylabel(r'$\phi$')

    ax2.set_xlabel('t')
    ax2.set_ylabel('U')

    ax3.set_ylabel('mass')
    ax3.set_xlabel('t')

    # ax2.set_xlim(.05, a.T)
    # ax3.set_xlim(.05, a.T)

    ax3.legend()

    plt.tight_layout()

    fname = (DIR + 'agents_U_exogenous_'
             + lib.fname_suffix(exclude=['V0','force_pos_type',
                                         'store_position','U'],**kwargs))
    print(fname, os.getcwd())
    plt.savefig(fname)
    # plt.close()

    sim_len = a.TN

    skipn = 10

    if False:

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='U exogenous', artist='Youngmin Park',
                        comment='Testing functions')
        writer = FFMpegWriter(fps=100, metadata=metadata)

        with writer.saving(fig,
                           DIR
                           + 'agents_U_exogenous_'
                           + lib.fname_suffix(exclude=['V0','force_pos_type'],
                                              **kwargs, ftype='.mp4'), dpi=120):
            j = 0
            while j <= sim_len / skipn:
                i = j * skipn
                if j % int(sim_len / 10 / skipn) == 0:
                    print(i / sim_len)
                image.set_data(a.x, a.sol[i, :])

                if a.V[i] > 0:
                    x = a.x[a.x > a.A]
                else:
                    x = a.x[a.x <= a.A]

                image2.set_data(x, lib.phi(x, a.U_arr[i], a))
                # print(np.sum(a.sol[i,:]),np.sum(lib.phi(x,a.U_arr[i],a)))
                ax1.set_ylim(0, .8)

                vel_pos.set_offsets(
                    [a.t[i], a.U(a.t[i], option=kwargs['fn_test_option'])])
                mass_pos.set_offsets([a.t[i], mass[i]])

                vel_text.set_text(
                    'U=%.2f' % a.U(a.t[i], option=kwargs['fn_test_option']))

                writer.grab_frame()

                j += 1


                
def mean_attach_time(update_kwargs,kwargs):

    for key in update_kwargs:
        kwargs[key] = update_kwargs[key]


    kwargs['store_position'] = True

    a = Agents(**kwargs)
    a.run_agents()

    

    # for each motor mark all attached time intervals
    # assume no motors start attached.
    for j in range(2):

        attach_times = []

        if j == 0:
            motor_bool_full = a.motor_bool_x1
        else:
            motor_bool_full = a.motor_bool_x2
            
        for i in range(a.nX):

            motor_bool = motor_bool_full[:,i]

            change_array = np.diff(motor_bool)
            on_idxs = np.where(change_array == 1)[0]
            off_idxs = np.where(change_array == -1)[0]

            on_times = a.t[on_idxs]
            off_times = a.t[off_idxs]

            #print()
            #print(on_times)
            #print(off_times)
            #print(motor_bool[0],motor_bool[-1])

            if on_idxs != [] and off_idxs != []:
                    # if motors start or finish attached, need to handle differently.
                if motor_bool[-1] == 1 and motor_bool[0] == 1:
                    on_times = on_times[:-1]
                    off_times = off_times[1:]

                elif motor_bool[-1] == 1 and motor_bool[0] == 0:
                    on_times = on_times[:-1]

                elif motor_bool[-1] == 0 and motor_bool[0] == 1:
                    off_times = off_times[1:]

                elif motor_bool[-1] == 0 and motor_bool[0] == 0:
                    pass

                #print(on_idxs,off_idxs)

                on_intervals = off_times-on_times

                attach_times.append(list(on_intervals))

        flat = np.concatenate(attach_times).flat

    
    
        if j == 0:
            print("Agents mean attach time nonpref",np.mean(flat))
            A = kwargs['A']
            B = kwargs['B']
            be = kwargs['beta']
            U = kwargs['U']
            print("Theory mean attach time nonpref", (1-np.exp((A-B)*be/np.abs(U)))/be )

        else:
            print("Agents mean attach time pref",np.mean(flat))
            print("Theory mean attach time pref", 1/be )
            


    
    

        
if False:
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.hist(a.U1,alpha=.4,bins=10,density=True)
    ax1.hist(a.U2,alpha=.4,bins=20,density=True)

    ax2.hist(a.F1,alpha=.4,bins=10,density=True)
    ax2.hist(a.F2,alpha=.4,bins=20,density=True)


def main():
    #steady_state(42,opt)
    #mean_force(opt)
    #mean_pos(opt)
    
    velocity(opt)
    #import cProfile
    #import re
    #cProfile.run('velocity(opt)')
    
    #generate_pos_mov(opt)
    #mean_attach_time({"U":50},opt)
    

    #test_u_variable({'fn_test_option':'square',
    #                 'U':fn_test,'T':1,'nX':1000,'nY':1000},opt)

    #a = Agents(**opt)
    #print('seed',a.seed)
    #a.run_agents()


if __name__ == "__main__":
    main()

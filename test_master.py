
from Master import Master
import libMotorPDE as lib
import numpy as np
import parsets as pset
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
#matplotlib.use('TkAgg')

import platform
from pathlib import Path

# operating system
sys = platform.system()

if sys  == 'Darwin':
    home = str(Path.home())
    DIR = home+'/Dropbox/thomas-youngmin/vesicles/yptgf2/code_and_data/tests/'
    #print(os.path.isdir(DIR))
elif sys == 'Windows':
    DIR = 'D:/Dropbox/thomas-youngmin/vesicles/yptgf2/code_and_data/tests/'


opt = {'T':.1,
       'dt':2e-5,
       'seed':0,
       'B':5.4,
       'U':None,
       'X0':10,'Y0':10,
       'A0':0,
       'N':1000,
       'source':True,
       'extension':True,
       'store_position':False,
       'store_draws':False}

opt = dict(opt,**pset.parset('3b'))


def steady_state(v,kwargs):
    """
    get both PDE steady-state and sampling steady-state

    Parameters
    ----------
    v
    kwargs

    Returns
    -------

    """
    kwargs['U'] = v
    kwargs['store_draws'] = True
    a = Master(**kwargs)
    a.run_master()

    ground_truth_values = lib.ground_truth(a)

    # get ground truth for other direction
    a.U = -a.U
    ground_truth_values2 = lib.ground_truth(a)

    drawsX = np.concatenate(a.pos_drawsX).ravel()
    drawsY = np.concatenate(a.pos_drawsY).ravel()

    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)

    ax1.plot(a.x,a.posX[-1,:],label='PDE/Master')
    ax1.plot(a.x,ground_truth_values,label='Ground Truth')

    ax2.plot(a.x,a.posY[-1, :],label='PDE/Master')
    ax2.plot(a.x,ground_truth_values2,label='Ground Truth')

    ax3.hist(drawsX[drawsX>=4.97], bins=50)
    ax4.hist(drawsY, bins=50)

    ax1.set_xlim(4.97,5.12)
    ax2.set_xlim(0,5.05)

    ax1.legend()
    ax2.legend()

    plt.savefig(DIR + 'temp_figures/steady_state_master.png')
    plt.close()


def mean_position(kwargs):

    kwargs['store_position'] = True
    a = Master(**kwargs)
    a.run_master()

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    t = np.linspace(0, a.T, a.TN)

    ax1.plot(t[t > .01], a.mean_posX[t > .01], alpha=.4, label='X')
    ax1.plot(t[t > .01], a.mean_posY[t > .01], alpha=.4, label='Y')

    ax1.legend()

    plt.savefig(DIR+'temp_figures/mean_pos_master.png')



def generate_pos_mov(update_kwargs,kwargs):

    for key in update_kwargs:
        kwargs[key] = update_kwargs[key]

    kwargs['store_position'] = True

    a = Master(**kwargs)
    a.run_master()

    fig = plt.figure(figsize=(3, 6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    # line, = ax.plot(a.x,a.posX[jj,:])
    # ax.set_xlim(a.A-(a.B-a.A),a.B)

    # ax2 = fig.add_subplot(122)

    time.sleep(10)
    t = np.linspace(0, a.T, a.TN)
    plot_counter = 0
    for i in range(a.TN):
        if i % int((a.TN) / 1000) == 0:
            print(round(i / a.TN,4),a.posX[i, :10])

            ax1.plot(a.x, a.posX[i, :])
            ax2.plot(a.x, a.posY[i, :])

            time_label = "{:02.3f}".format(t[i])
            v_label = "{:02.3f}".format(a.V[i])
            x_label = "{:02d}".format(a.X[i])
            y_label = "{:02d}".format(a.Y[i])
            ax1.set_title('master t=' + time_label
                          + '; v=' + v_label
                          + '; X=' + x_label
                          + '; Y=' + y_label)

            ax2.set_xlabel('Attached Motor Head Position')
            ax2.set_ylabel('Position Count Y')

            ax1.set_xlim(a.A0, a.B)
            ax1.set_ylim(0, .7)

            ax2.set_xlim(a.A0, a.B)
            ax2.set_ylim(0, .7)

            plt.savefig('C:/temp2/mov_master/' + str(plot_counter) + '.png',
                        dpi=100)
            ax1.clear()
            ax2.clear()

            plot_counter += 1


def velocity(kwargs):
    kwargs['store_position'] = True

    a = Master(**kwargs)
    a.run_master()

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(122)

    t = np.linspace(0, a.T, a.TN)


    ax1.plot(t, a.V, alpha=.4)

    # plot mean velocity lines
    ax1.plot([t[0], t[-1]], [a.switch_v, a.switch_v], color='gray')
    ax1.plot([t[0], t[-1]], [-a.switch_v, -a.switch_v], color='gray')

    #ax1.legend()

    plt.savefig(DIR + 'temp_figures/master_velocity.png')


def test_U_variable(**kwargs):

    a = Master(**kwargs)

    a.run_master()

    mass = np.sum(a.sol, axis=1) * a.dx

    ss_mass = np.zeros(a.TN)
    for i in range(len(a.t)):

        if a.U_arr[i] > 0:
            x = a.x[a.x > a.A]
        else:
            x = a.x[a.x <= a.A]

        ss_mass[i] = np.sum(lib.phi(x, a.U_arr[i], a)) * a.dx

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=100, metadata=metadata)

    fig = plt.figure(figsize=(8, 8))

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    image, = ax1.plot([], [])
    image2, = ax1.plot([], [])

    # get bounds for velocity function
    maxvel = np.amax(a.U(a.t, option=kwargs['fn_test_option']))
    minvel = np.amin(a.U(a.t, option=kwargs['fn_test_option']))

    ax2.plot(a.t[a.t > .05],
             a.U(a.t, option=kwargs['fn_test_option'])[a.t > .05])
    vel_text = ax2.text(0.05, (maxvel + minvel) / 2, 'U=%.2f' % a.U(0))
    vel_pos = ax2.scatter(a.t[0], a.U(a.t[0]))

    ax2.plot([a.t[0], a.t[-1]], [0, 0], ls='--', color='gray')

    ax3.plot(a.t[a.t > .05], mass[a.t > .05])
    ax3.plot(a.t[a.t > .05], ss_mass[a.t > .05],
             ls='--', color='gray', label='ss mass')
    # ax3.plot([a.t[0],a.t[-1]],[],ls='--',color='gray',label='True mass U>0')
    # ax3.plot([a.t[0],a.t[-1]],[],ls='-',color='k',label='True mass U<0')
    start_idx = np.argmin(np.abs(a.t - 0.05))
    mass_pos = ax3.scatter(a.t[start_idx], mass[start_idx])

    ax1.set_xlim(a.A0, a.B)
    ax1.set_ylim(0, .2)

    ax2.set_ylim(minvel, maxvel)

    ax2.set_xlim(0.05, a.T)
    ax3.set_xlim(0.05, a.T)

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

    fname = (DIR_TESTS
             + 'master_U_exogenous_'
             + lib.fname_suffix(**kwargs))
    print(fname, os.getcwd())
    plt.savefig(fname)
    # plt.close()

    sim_len = len(a.sol[:, 0])

    skipn = 10

    if False:
        with writer.saving(fig,
                           DIR_TESTS
                           + 'master_U_exogenous_'
                           + lib.fname_suffix(**kwargs, ftype='.mp4'), dpi=120):
            j = 0
            while j <= sim_len / skipn:
                i = j * skipn
                if j % int(sim_len / 10 / skipn) == 0:
                    print(i / sim_len)
                image.set_data(a.x, a.sol[i, :])

                if a.U_arr[i] > 0:
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
                    'U=%.2f' % a.U(a.dt * i, option=kwargs['fn_test_option']))

                writer.grab_frame()

                j += 1


def main():
    #steady_state(164,opt)
    #mean_position(opt)
    velocity(opt)
    #generate_pos_mov({'T':.1},opt)

    #opt['store_position'] = True
    #a = Master(**opt)
    #a.run_master()


if __name__ == "__main__":
    main()

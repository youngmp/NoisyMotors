"""

Figure generation code for yptgf2

"""

#import matplotlib
#matplotlib.use('Agg')

import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import parsets as pset
import numpy as np
import scipy as sp


from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

from get_mfpts import get_mfpts
import lib.test_agents as test_agents
import lib.libMotorPDE as lmpde
from lib.lubrication import lubrication
from agents import Agents
from Master import Master
import fit_langevin
import lib.telegraph

import matplotlib as mpl
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{siunitx}')

FIGWIDTH_PX = 850
MY_DPI = 96

if not os.path.exists('data'):
    os.makedirs('data')
 
size = 15

class Arrow3D(FancyArrowPatch):
    """
    A class for drawing arrows in 3d plots.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)



def cylinder():
    
    """
    TODO: add small cartoon axes to subplots.
    mmake dome head for spine figure
    
    """
    
    T1 = .1
    
    gs = gridspec.GridSpec(nrows=2,ncols=3,wspace=-.1,hspace=.5)
    fig = plt.figure(figsize=(5,4))
    ax11 = fig.add_subplot(gs[:,:2],projection='3d')
    ax12 = fig.add_subplot(gs[0,2])
    ax22 = fig.add_subplot(gs[1,2])
    
    
    a = lubrication(phi1=.57,Rp=0.96,Rc=1.22,base_radius=1.22,
                    pi3=1,pi4=4.7,pi5=0.1,pi6=10,
                    mu=1.2,T=T1,constriction='piecewise',U0=0.2,
                    dt=0.02,eps=1,
                    F0=50,method='euler')
    a.Z0 = -5/a.Rp
    
    z = np.linspace(-7,7,100)  # dimensional
    r = a.pi1(z)
    th = np.linspace(0,2*np.pi,100)
    
    radius_al = 0.25
    
    # draw arrow going into spine
    
    ar1 = Arrow3D([0,0],[0,0],[-5,-1],
                  mutation_scale=10, 
                  lw=2, arrowstyle="-|>", color="k")
    
    ax11.add_artist(ar1)

    # A
    # draw spine
    Z,TH = np.meshgrid(z,th)
    #Z,TH = np.mgrid[-7:7:.1, 0:2*np.pi:.1]
    X = np.zeros_like(Z)
    Y = np.zeros_like(Z)
    print(np.shape(Z))
    for i in range(len(Z[:,0])):
        X[i,:] = a.pi1(Z[i,:])*np.cos(TH[i,:])
        Y[i,:] = a.pi1(Z[i,:])*np.sin(TH[i,:])
    
    ax11.plot_surface(X,Y,Z,alpha=.25)
    
    shifts = np.array([3,-3,0])
    names = ['Z','Y','Z']
    size = 2
    
    
    for i in range(3):
        coords = np.zeros((3,2))
        
        coords[:,0] += shifts
        coords[:,1] += shifts
        
        coords[i][1] += size
        arx = Arrow3D(*list(coords),
                      mutation_scale=5, 
                      lw=2, arrowstyle="-|>", color="k")
    
        ax11.text(*list(coords[:,1]),names[i],horizontalalignment='center')
    
        ax11.add_artist(arx)
        
    

    # draw sphere for cap
    b = a.base_radius
    r = np.sqrt(b**2+7**2)
    th2 = np.linspace(0,np.arctan(b/7),100)
    phi = np.linspace(0,2*np.pi,100)
    
    TH2,PHI = np.meshgrid(th2,phi)
    X = r*np.sin(TH2)*np.cos(PHI)
    Y = r*np.sin(TH2)*np.sin(PHI)
    Z = r*np.cos(TH2)
    ax11.plot_surface(X,Y,Z,color='tab:blue',alpha=.5)

    
    # draw sphere vesicle
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    X = np.cos(u)*np.sin(v)
    Y = np.sin(u)*np.sin(v)
    Z = np.cos(v)
    ax11.plot_surface(X,Y,Z,color='gray',alpha=.5)
    
    # label spine head and base
    ax11.text(2.5,-5,6,r'\setlength{\parindent}{0pt}Spine Head\\(Closed End)')
    
    
    ax11.text(2.5,-5,-7,r'\setlength{\parindent}{0pt}Spine Base\\(Open End)')    
    
    # B
    # Draw vesicle and spine wall
    ax12.plot(a.Rc*np.cos(th),a.Rc*np.sin(th),color='tab:blue',alpha=.5)
    ax12.fill(a.Rp*np.cos(th),a.Rp*np.sin(th),color='gray',alpha=.5)
    
    ax12.text(-.6,.4,'Vesicle')
    ax12.text(.9,.9,'Spine Wall')
    
    # annotate Rp length
    ax12.annotate('', xy=(0, -a.Rc), xytext=(0, 0), xycoords='data',
               arrowprops=dict(arrowstyle='-|>',color='k', lw=1))
    ax12.text(-.35,-.5,r'$R_c$')

    # annotate Rc length channel
    ax12.annotate(r'', xy=(a.Rp, 0.0), xytext=(0, 0), xycoords='data', 
                  arrowprops=dict(arrowstyle='-|>', color='k',lw=1))
    ax12.text(.4,.2,r'$R_p$')
    
    # annotate Rc length channel

    ax12.annotate(r'$h(s)$', xy=(1.08,-.1), xytext=(1.08, -1.5),
                  xycoords='data',fontsize=10, ha='center', va='center',
                  arrowprops=dict(arrowstyle=('-[, widthB='+str(a.Rc-a.Rp+.2)
                                              +', lengthB=.5'), lw=.5))

    # draw axes
    shift = .75
    x_center, y_center = (-1.5,-1.5)

    ax12.annotate(r'X', xy=(x_center,y_center),
                  xytext=(x_center+shift,y_center),
                  va='center',
                  arrowprops=dict(mutation_scale=5, 
                                  arrowstyle='<|-', 
                                  color='k',lw=2),annotation_clip=False)
    
    ax12.annotate(r'Y', xy=(x_center,y_center),
                  xytext=(x_center,y_center+shift),
                  ha='center', 
                  arrowprops=dict(mutation_scale=5,
                                  arrowstyle='<|-',
                                  color='k',lw=2),annotation_clip=False)
    
    # C
    # draw molecular motors
    
    pad = .15
    ax22.fill(a.Rp*np.cos(th),a.Rp*np.sin(th),color='gray',alpha=.5)
    
    ax22.plot([a.Rc+pad,a.Rc+pad],[-2,2],color='tab:blue',alpha=.5)
    ax22.plot([-a.Rc-pad,-a.Rc-pad],[-2,2],color='tab:blue',alpha=.5)
    
    ax22.text(0,0,'Vesicle',ha='center',va='center')
    ax22.text(-a.Rc-3*pad,0,'Spine Wall',rotation=90)
    
    ax22.annotate(r'$Z$',xy=(a.Rp,0),xytext=(a.Rc+4*pad,0),
                  ha='center',va='center',rotation=-90,
                  arrowprops=dict(arrowstyle='-|>',
                                  color='k',lw=1),annotation_clip=False)
    
    # draw axes
    shift = .75
    x_center, y_center = (-1.5,-1.5)

    ax22.annotate(r'X,Y', xy=(x_center,y_center),
                  xytext=(x_center+shift,y_center),
                  va='center',
                  arrowprops=dict(mutation_scale=5, 
                                  arrowstyle='<|-', 
                                  color='k',lw=2),
                  annotation_clip=False)
    
    ax22.annotate(r'Z', xy=(x_center,y_center),
                  xytext=(x_center,y_center+shift),
                  ha='center', 
                  arrowprops=dict(mutation_scale=5,
                                  arrowstyle='<|-',
                                  color='k',lw=2),
                  annotation_clip=False)
    
    
    ax11.set_title(r'\textbf{A} Idealized Spine Geometry',loc='left',y=1.085)
    ax12.set_title(r'\textbf{B} Transverse Cross-section',loc='left',x=-.2)
    ax22.set_title(r'\textbf{C} Longitudinal Cross-section',loc='left',x=-.27)
    
    
    # set equal aspect ratios
    #ax11.set_aspect('auto') # only auto allowed??
    ax11.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
    ax12.set_aspect(1)
    ax22.set_aspect(1)
    
    ax11.set_axis_off()
    ax12.set_axis_off()
    ax22.set_axis_off()
    
    ax22.set_xticks([])
    ax22.set_yticks([])
    
    lo = -4.4
    hi = 4.4
    dx = -.5
    
    ax11.set_xlim(lo-dx,hi+dx)
    ax11.set_ylim(lo-dx,hi+dx)
    ax11.set_zlim(lo,hi)
    
    ax22.set_xlim(-1.4,1.4)
    ax22.set_ylim(-1.4-pad,1.4+pad)
    
    ax11.view_init(20,45)
    
    #plt.tight_layout()
    
    return fig


def cylinder_sideways():
    
    """
    sideways cylinder for poster
    """
    
    T1 = .1
    
    #gs = gridspec.GridSpec(nrows=2,ncols=3,wspace=-.1,hspace=.5)
    fig = plt.figure(figsize=(5,4))
    ax11 = fig.add_subplot(111,projection='3d')
    #ax12 = fig.add_subplot(gs[0,2])
    #ax22 = fig.add_subplot(gs[1,2])
    
    
    a = lubrication(phi1=.57,Rp=0.96,Rc=1.22,base_radius=1.22,
                    pi3=1,pi4=4.7,pi5=0.1,pi6=10,
                    mu=1.2,T=T1,constriction='piecewise',U0=0.2,
                    dt=0.02,eps=1,
                    F0=50,method='euler')
    a.Z0 = -5/a.Rp
    
    z = np.linspace(-7,7,100)  # dimensional
    r = a.pi1(z)
    th = np.linspace(0,2*np.pi,100)
    
    radius_al = 0.25
    
    # draw arrow going into spine
    
    ar1 = Arrow3D([-5,-1.5],[0,0],[0,0],
                  mutation_scale=10, 
                  lw=2, arrowstyle="-|>", color="k")
    
    ax11.add_artist(ar1)

    # A
    # draw spine
    Z,TH = np.meshgrid(z,th)
    #Z,TH = np.mgrid[-7:7:.1, 0:2*np.pi:.1]
    X = np.zeros_like(Z)
    Y = np.zeros_like(Z)
    print(np.shape(Z))
    for i in range(len(Z[:,0])):
        X[i,:] = a.pi1(Z[i,:])*np.cos(TH[i,:])
        Y[i,:] = a.pi1(Z[i,:])*np.sin(TH[i,:])
    
    ax11.plot_surface(Z,Y,X,alpha=.25)
    
    shifts = np.array([-6,0,-4])
    names = ['z','y','x']
    size = 2
    
    
    for i in range(3):
        coords = np.zeros((3,2))
        
        coords[:,0] += shifts
        coords[:,1] += shifts
        
        coords[i][1] += size
        arx = Arrow3D(*list(coords),
                      mutation_scale=5, 
                      lw=2, arrowstyle="-|>", color="k")
    
        ax11.text(*list(coords[:,1]),names[i],horizontalalignment='center')
    
        ax11.add_artist(arx)
        
    

    # draw sphere for cap
    b = a.base_radius
    r = np.sqrt(b**2+7**2)
    th2 = np.linspace(0,np.arctan(b/7),100)
    phi = np.linspace(0,2*np.pi,100)
    
    TH2,PHI = np.meshgrid(th2,phi)
    X = r*np.sin(TH2)*np.cos(PHI)
    Y = r*np.sin(TH2)*np.sin(PHI)
    Z = r*np.cos(TH2)
    ax11.plot_surface(Z,Y,X,color='tab:blue',alpha=.5)

    
    # draw sphere vesicle
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    X = np.cos(u)*np.sin(v)
    Y = np.sin(u)*np.sin(v)
    Z = np.cos(v)
    ax11.plot_surface(Z,Y,X,color='gray',alpha=.5)
    
    # label spine head and base
    ax11.text(7,0,-2,r'\setlength{\parindent}{0pt}Spine Head\\(Closed End)')
    ax11.text(-4,0,3,r'\setlength{\parindent}{0pt}Spine Base\\(Open End)')    
        
    # set equal aspect ratios
    #ax11.set_aspect('auto') # only auto allowed??
    ax11.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
    
    ax11.set_axis_off()
    
    lo = -4.4
    hi = 4.4
    
    dx = -.5
    
    ax11.set_xlim(lo-dx,hi+dx)
    ax11.set_ylim(lo-dx,hi+dx)
    ax11.set_zlim(lo,hi)
    
    ax11.view_init(20,65)

    
    
    return fig


def agent_example():
    """
    plot an example trajectory of the agent-based model.

    display velocity and position.
    """
    
    kwargs = {'T':1,
              'V0':0,
              'Z0':0,
              'dt':2e-6,
              'seed':2,
              'U':None,
              'force_pos_type':'lin',
              'store_position':False,
              'ext':True}

    kwargs = dict(kwargs,**pset.parset('figure1'))
    
    kwargs['store_position'] = True
    a = Agents(**kwargs)
    a.run_agents()

    fig = plt.figure(figsize=(8,3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    t = np.linspace(0, a.T, a.TN)

    # skip plotting subset of points (so pdfs dont die)
    skipn = 10
    
    

    # plot mean velocity lines
    ax1.plot([t[0], t[-1]], [a.switch_v, a.switch_v], color='gray')
    ax1.plot([t[0], t[-1]], [-a.switch_v, -a.switch_v], color='gray',label="QSS")

    # vesicle velocity and position
    ax1.plot(t[::skipn], a.V[::skipn], alpha=1)
    ax2.plot(t[::skipn][:-1], a.Z[::skipn][:-1], alpha=1,lw=2)
    
    #
    ax1.set_ylabel(r'Velocity V (\si{\nano\meter/\second})',fontsize=size)
    ax2.set_ylabel(r'Position Z (\si{\nano\meter})',fontsize=size)
    
    ax1.set_xlabel('Time (\si{\second})',fontsize=size)
    ax2.set_xlabel('Time (\si{\second})',fontsize=size)

    ax1.set_title(r'\textbf{A}',loc='left',fontsize=size)
    ax2.set_title(r'\textbf{B}',loc='left',fontsize=size)
    #plt.suptitle('Agent-Based Model',fontsize=size)

    ax1.set_xlim(0,a.t[-1])
    ax2.set_xlim(0,a.t[-1])
    
    ax1.tick_params(axis='both',which='major', labelsize=size)
    ax2.tick_params(axis='both',which='major', labelsize=size)

    ax1.legend()

    plt.tight_layout()

    return fig

    #plt.savefig(DIR + 'temp_figures/agents_velocity.png')


def fpDimLin(U,kwargs):

    al = kwargs['alpha']
    be = kwargs['beta']
    A = kwargs['A']
    B = kwargs['B']
    gamma = kwargs['gamma']
    p1 = kwargs['p1']
    n0 = kwargs['nX']

    prefix = -p1*gamma*n0*al
    
    if U >= 0:
        ee = np.exp((A-B)*be/U)

        return prefix*(U+A*be-ee*(U+B*be))/(be*(al-ee*al+be))
    
    else:
        return prefix*(U+A*be)/(be*(al+be))



def run_langevin(t,switch_vel=None,kwargs={}):
    
    switch_times = []
    side = 0
    width = 42

    Rp = 0.96
    mu = 1.2
    
    zeta = kwargs['zeta']
    
    zeta_nd = zeta/(6*np.pi*Rp*mu)
    # get the mean-field force-velocity curves
    
    
    # set up langevin simulation
    dt = t[-1]-t[-2]

    U = np.zeros(len(t))
    U[0] = 121 # explicit for clarity

    for i in range(len(t)-1):
        
        # Euler step
        w = np.random.normal(0,1)

        motor_forces = fpDimLin(U[i],kwargs) - fpDimLin(-U[i],kwargs)
        #print(motor_forces)
        total_force =  motor_forces - zeta*U[i]
        
        U[i+1] = U[i] + dt*total_force + np.sqrt(dt)*width*w

        # track switch times
        if switch_vel != None:
                        
            if (side == 0) and (U[i+1] >= switch_vel):
                side = 1
                switch_times.append(t[i])

            if (side == 1) and (U[i+1] <= -switch_vel):
                side = 0
                switch_times.append(t[i])
    
    return U,switch_times

def langevin_vs_agents(recompute_agents=False,
                       recompute_lan=False):

    fig = plt.figure(figsize=(8,5))
    ax11 = fig.add_subplot(221) # solution distribution agents
    ax12 = fig.add_subplot(222) # solution distribution langevin
    ax21 = fig.add_subplot(223) # solution + MFPT dots agents
    ax22 = fig.add_subplot(224) # solution + MFPT dots agents

    # agent simulation
    kwargs = {'T':11,
              'V0':0,
              'Z0':0,
              'dt':2e-6,
              'seed':9,
              'U':None,
              'force_pos_type':'lin',
              'store_position':False,
              'ext':True}

    kwargs = dict(kwargs,**pset.parset('figure1'))
    kwargs['store_position'] = True

    f_switch_agents = 'data/switch_agents.txt'
    file_does_not_exist = not(os.path.isfile(f_switch_agents))

    if recompute_agents or file_does_not_exist:
        a = Agents(**kwargs)
        a.run_agents()
        
        switch_agents = a.switch_times
        switch_agents = np.asarray(switch_agents)
        t_agents = a.t
        V = a.V
        Z = a.Z
        
        np.savetxt(f_switch_agents,a.switch_times)
        np.savetxt('data/t.txt',a.t)
        np.savetxt('data/V.txt',a.V)
        np.savetxt('data/Z.txt',a.Z)
        
    else:
        switch_agents = np.loadtxt(f_switch_agents)
        t_agents = np.loadtxt('data/t.txt')
        V = np.loadtxt('data/V.txt')
        Z = np.loadtxt('data/Z.txt')

    mfpt_agents = np.mean(np.diff(switch_agents))

    f_switch_lan = 'data/switch_lan.txt'
    file_does_not_exist = not(os.path.isfile(f_switch_lan))
    
    # langevin simulation
    if recompute_lan or file_does_not_exist:
        dt = 1e-2
        tfinal = 15000
        t_lan = np.linspace(0,tfinal,int(tfinal/dt))

        np.random.seed(1)
        sol_langevin,switch_times = run_langevin(t_lan,
                                                 switch_vel=kwargs['switch_v'],
                                                 kwargs=kwargs)

        
        V_lan = sol_langevin
        
        np.savetxt(f_switch_lan,switch_times)
        np.savetxt('data/t_lan.txt',t_lan)
        np.savetxt('data/V_lan.txt',sol_langevin)
        #np.savetxt('Z.txt',a.Z)
        
    else:
        switch_times = np.loadtxt(f_switch_lan)
        t_lan = np.loadtxt('data/t_lan.txt')
        V_lan = np.loadtxt('data/V_lan.txt')

    
    mfpt_lan = np.mean(np.diff(switch_times))
    
    # plot solutions

    # agents  cutting parts of solutions to make symmetric histogram
    cut1 = int(len(V)/2)
    ax11.plot(t_agents[:cut1],V[:cut1])

    t_last = t_agents[cut1]

    ax11.scatter(switch_agents[switch_agents<t_last],
                 np.zeros(len(switch_agents[switch_agents<t_last])),
                 color='tab:red',alpha=0.7,s=30)

    # plot agents histogram from longer simulations
    cut_s = int(0)
    cut_f = -int(1)
    counts, bins, bars = ax21.hist(V[cut_s:cut_f],bins=40,
                                   density=True,label='Density')
    np.savetxt('data/counts.csv',counts)
    np.savetxt('data/bins.csv',bins)

    # get steady-state langevin and plot
    bins = np.loadtxt('data/bins.csv')[:-1]
    counts = np.loadtxt('data/counts.csv')

    out = curve_fit(fit_langevin.ps,bins,counts,50)
    sigma = out[0]
    x = np.linspace(-350,350,100000)
    ax21.plot(x,fit_langevin.ps(x,sigma),color='tab:orange',
              label=r'$p_s$')

    # langevin
    cut2 = -int(14*len(V_lan)/15)
    ax12.plot(t_lan[:cut2],V_lan[:cut2])
    ax12.scatter(switch_times,np.zeros(len(switch_times)),
                 color='tab:red',alpha=0.7,s=30)
    counts2, bins2, bars2 = ax22.hist(V_lan[:],bins=40,
                                      density=True,
                                      label='Density')

    ax22.plot(x,fit_langevin.ps(x,sigma),color='tab:orange',
              label=r'$p_s$')
    
    np.savetxt('data/counts_lan.csv',counts2)
    np.savetxt('data/bins_lan.csv',bins2)

    mfpt1 = np.round(mfpt_agents,2)
    mfpt2 = np.round(mfpt_lan,2)
    ax11.set_title(r'\textbf{A} Agent-Based (Switch=\SI{'+str(mfpt1)+'}{\s})',
                   fontsize=size,loc='left')
    ax12.set_title(r'\textbf{B} Langevin (Switch=\SI{'+str(mfpt2)+'}{\s})',
                   fontsize=size,loc='left')
    ax21.set_title(r'\textbf{C} Agent-Based',fontsize=size,loc='left')
    ax22.set_title(r'\textbf{D} Langevin',fontsize=size,loc='left')

    ax11.set_xlabel('Time (\si{s})',fontsize=size)
    ax12.set_xlabel('Time (\si{s})',fontsize=size)
    ax21.set_xlabel('Velocity (V \si{nm/s})',fontsize=size)
    ax22.set_xlabel('Velocity (V \si{nm/s})',fontsize=size)
    
    ax11.set_ylabel('Velocity (V \si{nm/s})',fontsize=size)
    ax12.set_ylabel('Velocity (V \si{nm/s})',fontsize=size)
    ax21.set_ylabel('Probability',fontsize=size)
    ax22.set_ylabel('Probability',fontsize=size)

    ax11.tick_params(axis='both',which='major', labelsize=size)
    ax12.tick_params(axis='both',which='major', labelsize=size)
    ax21.tick_params(axis='both',which='major', labelsize=size)
    ax22.tick_params(axis='both',which='major', labelsize=size)

    #ax21.legend()
    #ax22.legend()

    print('agents final time', t_agents[:cut_f][-1])
    ax11.set_xlim(0,t_agents[:cut1][-1])
    ax12.set_xlim(0,t_lan[:cut2][-1])


    plt.tight_layout()
    
    return fig

def motor_distributions(recompute=False):

    gs = gridspec.GridSpec(nrows=1,ncols=5,wspace=0,hspace=0)
    
    fig = plt.figure(figsize=(8,2))
    ax11 = fig.add_subplot(gs[0,0])
    ax12 = fig.add_subplot(gs[0,2])
    ax13 = fig.add_subplot(gs[0,4])
    
    opt = {'T':2,
           'dt':1e-6,
           #'T':.01,
           #'dt':1e-5,
           'V0':0,
           'seed':0,
           'U':None,
           'force_pos_type':'lin',
           'store_position':False,
           'ext':True}

    opt = dict(opt,**pset.parset('figure1'))

    a = Agents(**opt)
    x1 = np.linspace(0,5,100)
    x2 = np.linspace(5,5.05,20)

    ground_truthX = lmpde.phi(x1,-opt['switch_v'],a)
    ground_truthY = lmpde.phi(x2,opt['switch_v'],a)

    fname_xbins = 'data/xbins.txt'
    fname_ybins = 'data/ybins.txt'
    fname_xfreq = 'data/xfreq.txt'
    fname_yfreq = 'data/yfreq.txt'

    file_does_not_exist = not(os.path.isfile(fname_xbins))

    if recompute or file_does_not_exist:
        
        pack = test_agents.steady_state(-opt['switch_v'],
                                        opt,return_data=True)
        xbins = pack['X_bins']
        ybins = pack['Y_bins']
        xfreq = pack['X_freq']
        yfreq = pack['Y_freq']

        np.savetxt(fname_xbins,xbins)
        np.savetxt(fname_ybins,ybins)
        np.savetxt(fname_xfreq,xfreq)
        np.savetxt(fname_yfreq,yfreq)
        
        
    else:

        xbins = np.loadtxt(fname_xbins)
        ybins = np.loadtxt(fname_ybins)
        xfreq = np.loadtxt(fname_xfreq)
        yfreq = np.loadtxt(fname_yfreq)
        

    #x, true1, true2, X_hist, X_bins, Y_hist, Y_bins
    ax11.hist(xbins[:-1], xbins, weights=xfreq)
    ax13.hist(ybins[:-1], ybins, weights=yfreq)

    ax11.set_title(r"\textbf{A}",loc='left',fontsize=size)
    ax12.set_title(r"\textbf{B}",loc='left',fontsize=size)
    ax13.set_title(r"\textbf{C}",loc='left',fontsize=size)
    
    ax11.plot(x1,ground_truthX)
    ax13.plot(x2,ground_truthY)

    ax12.scatter(5,0,color='white')
    ax12.scatter(4,0,color='white')
    ax12.scatter(5.05,.1,color='white')

    ax11.set_xlim(0,5)
    ax13.set_xlim(4.99,5.06)

    ax11.yaxis.tick_right()
    ax11.spines['left'].set_visible(False)
    ax11.spines['top'].set_visible(False)

    
    ax12.spines['left'].set_position(('data',5))
    ax12.spines['right'].set_visible(False)
    ax12.spines['top'].set_visible(False)

    ax12.yaxis.set_ticks_position('none')
    ax12.axes.get_yaxis().set_ticks([])

    ax13.spines['right'].set_visible(False)
    ax13.spines['top'].set_visible(False)

    ax11.tick_params(axis='both',which='major', labelsize=size)
    ax12.tick_params(axis='both',which='major', labelsize=size)
    ax13.tick_params(axis='both',which='major', labelsize=size)

    ax11.set_xlabel(r'Local Position $z$',fontsize=size)
    ax12.set_xlabel(r'Local Position $z$',fontsize=size)
    ax13.set_xlabel(r'Local Position $z$',fontsize=size)

    ax11.set_ylabel(r'Probability',fontsize=size)
    ax12.set_ylabel(r'Probability',fontsize=size)
    ax13.set_ylabel(r'Probability',fontsize=size)

    ax12.yaxis.labelpad = 100

    plt.tight_layout()
    return fig


def master_vs_agents(recompute_agents=False,
                     recompute_master=False):

    #fig = plt.figure(figsize=(FIGWIDTH_PX/MY_DPI,850*5/8/MY_DPI))
    fig = plt.figure(figsize=(8,5))
    
    ax11 = fig.add_subplot(221) # solution distribution agents
    ax12 = fig.add_subplot(222) # solution distribution langevin
    ax21 = fig.add_subplot(223) # solution + MFPT dots agents
    ax22 = fig.add_subplot(224) # solution + MFPT dots agents

    # agent simulation
    kwargs = {'T':11,
              'V0':0,
              'Z0':0,
              'dt':2e-6,
              'seed':9,
              'U':None,
              'force_pos_type':'lin',
              'store_position':True,
              'ext':True}

    parset_name = 'figure1'
    kwargs = dict(kwargs,**pset.parset(parset_name))

    f_switch_agents = 'data/switch_agents_'+parset_name+'.txt'
    f_counts_agents = 'data/counts_'+parset_name+'.csv'
    file_does_not_exist = not(os.path.isfile(f_counts_agents))
    
    if recompute_agents or file_does_not_exist:
        a = Agents(**kwargs)
        a.run_agents()
        
        switch_agents = a.switch_times
        t_agents = a.t[::10]
        V = a.V[::10]
        Z = a.Z[::10]
        
        np.savetxt(f_switch_agents,a.switch_times)
        np.savetxt('data/t_agents_'+parset_name+'.txt',t_agents)
        np.savetxt('data/V_agents_'+parset_name+'.txt',V)
        #np.savetxt('Z_agents_'+parset_name+'.txt',a.Z)
        
        cut_s = int(0)
        cut_f = -int(1)
        counts, bins, bars = ax21.hist(V[cut_s:cut_f],bins=40,
                                       density=True,label='Density')
        np.savetxt(f_counts_agents,counts)
        np.savetxt('data/bins_'+parset_name+'.csv',bins)

        
    else:
        switch_agents = np.loadtxt(f_switch_agents)
        t_agents = np.loadtxt('data/t_agents_'+parset_name+'.txt')
        V = np.loadtxt('data/V_agents_'+parset_name+'.txt')
        #Z = np.loadtxt('Z_agents_'+parset_name+'.txt')
        counts = np.loadtxt(f_counts_agents)
        bins = np.loadtxt('data/bins_'+parset_name+'.csv')[:-1]

    mfpt_agents = np.mean(np.diff(switch_agents))

    opt = {'T':5,
           'dt':3e-6,
           'seed':4,
           'U':None,
           'X0':10,'Y0':1,
           'A0':1,
           'N':41,
           'N2':41,
           'source':True,
           'irregular':True,
           'store_position':True,
           'store_draws':False}
    
    opt = dict(opt,**pset.parset(parset_name))

    f_switch_master = 'data/switch_master_'+parset_name+'.txt'
    f_counts_master = 'data/counts_master'+parset_name+'.txt'
    
    file_does_not_exist = not(os.path.isfile(f_counts_master))
    
    # master simulation
    if recompute_master or file_does_not_exist:

        am = Master(**opt)
        am.run_master()

        switch_master = am.switch_times
        t_master = am.t
        V_master = am.V
        
        np.savetxt(f_switch_master,switch_master)
        np.savetxt('data/t_master_'+parset_name+'.txt',t_master[::20])
        np.savetxt('data/V_master_'+parset_name+'.txt',V_master[::20])
        #np.savetxt('Z.txt',a.Z)

        cut1 = int(len(t_master)/2)
        counts2, bins2, bars2 = ax22.hist(V_master[cut1:],bins=40,
                                          density=True,
                                          label='Density')

        np.savetxt(f_counts_master,counts)
        np.savetxt('data/bins_master_'+parset_name+'.csv',bins)
        
    else:
        switch_master = np.loadtxt(f_switch_master)
        t_master = np.loadtxt('data/t_master_'+parset_name+'.txt')
        V_master = np.loadtxt('data/V_master_'+parset_name+'.txt')

        cut1 = int(len(t_master)/2)
        counts2, bins2, bars2 = ax22.hist(V_master[cut1:],bins=40,
                                          density=True,
                                          label='Density')

        counts2 = np.loadtxt(f_counts_master)
        bins2 = np.loadtxt('bins_master_'+parset_name+'.csv')[:-1]


    mfpt_master = np.mean(np.diff(switch_master))
    
    # plot solutions

    # agents  cutting parts of solutions to make symmetric histogram
    cut1 = int(len(V)/2)
    ax11.plot(t_agents[:cut1],V[:cut1])

    t_last = t_agents[cut1]
    ax11.scatter(switch_agents[switch_agents<t_last],
                 np.zeros(len(switch_agents[switch_agents<t_last])),
                 color='tab:red',alpha=0.7,s=30)

    # plot agents histogram from longer simulations
    #counts, bins, bars
    ax21.hist(V,bins=40,density=True)

    # get steady-state langevin and plot
    #bins = np.loadtxt('bins.csv')[:-1]
    #counts = np.loadtxt('counts.csv')

    

    # master
    cut1 = int(0)
    cut2 = int(-1)#-int(14*len(V_master)/15)
    ax12.plot(t_master[cut1:cut2],V_master[cut1:cut2])
    ax12.scatter(switch_master,np.zeros(len(switch_master)),
                 color='tab:red',alpha=0.7,s=30)
    

    #ax22.hist(V_master,bins=40,density=True)


    #ax22.plot(x,fit_langevin.ps(x,sigma),color='tab:orange',
    #          label=r'$p_s$')
    
    #np.savetxt('counts_master.csv',counts2)
    #np.savetxt('bins_master.csv',bins2)


    ax11.set_xlim(0,5)
    ax12.set_xlim(0,5)
    
    mfpt1 = np.round(mfpt_agents,2)
    mfpt2 = np.round(mfpt_master,2)
    ax11.set_title(r'\textbf{A} Agent-Based (Switch=\SI{0.19}{\s})',
                   loc='left',fontsize=size)
    ax12.set_title(r'\textbf{B} Master (Switch=\SI{0.19}{\s})',
                   loc='left',fontsize=size)
    ax21.set_title(r'\textbf{C} Agent-Based',loc='left',fontsize=size)
    ax22.set_title(r'\textbf{D} Master',loc='left',fontsize=size)

    ax11.set_xlabel('Time (\si{s})',fontsize=size)
    ax12.set_xlabel('Time (\si{s})',fontsize=size)
    ax21.set_xlabel('Velocity (V \si{nm/s})',fontsize=size)
    ax22.set_xlabel('Velocity (V \si{nm/s})',fontsize=size)
    
    ax11.set_ylabel('Velocity (V \si{nm/s})',fontsize=size)
    ax12.set_ylabel('Velocity (V \si{nm/s})',fontsize=size)
    ax21.set_ylabel('Probability',fontsize=size)
    ax22.set_ylabel('Probability',fontsize=size)

    ax11.tick_params(axis='both',which='major', labelsize=size)
    ax12.tick_params(axis='both',which='major', labelsize=size)
    ax21.tick_params(axis='both',which='major', labelsize=size)
    ax22.tick_params(axis='both',which='major', labelsize=size)

    #ax21.legend()
    #ax22.legend()

    #print('agents final time', t_agents[:cut_f][-1])
    #ax11.set_xlim(0,t_agents[:cut1][-1])
    #ax12.set_xlim(0,t_master[:cut2][-1])


    plt.tight_layout()
    
    return fig



def get_times(path,options):
    """
    collect simulation time data
    """
        
    fnames = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if all(x in file for x in options):
                fname = os.path.join(root,file)
                fnames.append(fname)

                
                #print(fname)
        
    master_list = []
    agents_list = []

    cutoff1 = 0
    cutoff2 = -1
    
    print('Total seeds =', len(fnames[cutoff1:cutoff2]), 'opts', options)

    
    for fname in fnames[cutoff1:cutoff2]:
        
        time_data = np.loadtxt(fname)
        master_list.append(time_data[0])
        agents_list.append(time_data[1])
        
    return master_list, agents_list


def mva_time():
    
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)

    nx_list = [100]
    for i in range(1,11):
        nx_list.append(i*1000)
    #nx_list = [100,500,1000,1500,2000,
    #           2500,3000,3500,4000,4500,
    #           5000]
    print(nx_list)

    path = "../cluster_data/figure1_times"
    #options = ['nX=100','dt=1e-06','T=10','.txt']

    mean_master = []
    mean_agents = []

    std_master = []
    std_agents = []
    
    for nx in nx_list:
        
        options = ['nX='+str(nx)+'_']
        times_master, times_agents = get_times(path,options)

        mean_master.append(np.mean(times_master))
        mean_agents.append(np.mean(times_agents))

        std_master.append(np.std(times_master))
        std_agents.append(np.std(times_agents))

    #ax.plot(nx_list,mean_agents,color='tab:orange')
    #ax.errorbar(nx_list,mean_agents,
    #            yerr=std_agents/np.sqrt(len(std_agents)),
    #            color='tab:orange',marker='^',label='Agents',ls='',
    #            markersize=8,clip_on=False,elinewidth=2,capsize=5)


    yerr_a = std_agents/np.sqrt(len(std_agents))
    ax.fill_between(nx_list,mean_agents-yerr_a,mean_agents+yerr_a,
                    alpha=.5,color='gray')
    ax.plot(nx_list,mean_agents,label='Agents',lw=2,
            marker='s',color='black',clip_on=False)

    
    #ax.plot(nx_list,mean_master,color='tab:blue')
    #ax.errorbar(nx_list,mean_master,
    #            yerr=std_master/np.sqrt(len(std_master)),
    #            color='tab:blue',marker='s',label='Master',ls='',
    #            markersize=8,clip_on=False,elinewidth=2,capsize=5)

    yerr_m = std_master/np.sqrt(len(std_master))
    ax.fill_between(nx_list,mean_master-yerr_m,mean_master+yerr_m,
                    alpha=.5,color='tab:blue')
    ax.plot(nx_list,mean_master,label='Master',lw=2,
            marker='^',color='tab:blue',clip_on=False)

    #ax.plot(nx_list,mean_master)
    #ax.plot(nx_list,mean_agents)

    
    ax.set_xlabel(r'Motor Number $N$',fontsize=size)
    ax.set_ylabel(r'Wall Time (\si{s})',fontsize=size)
    ax.tick_params(axis='both',which='major', labelsize=size)
    ax.legend(fontsize=12)

    plt.tight_layout()
    
    return fig


def velocity_mfpts():
    """
    plot of convergence. MFPT to switch velocity
    agents vs master.

    quantities obtained from julia code and master equation
    """

    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)
    
    dt_a = [1.00E-03,1.00E-04,1.00E-05,3.00E-06,1.00E-06,
            7.50E-07,5.00E-07,2.50E-07,1.00E-07]
    mfpt_a = [0.16397,0.18870,0.19168,0.18818,0.19291,
              0.19173,0.19394,0.19223,0.19500]

    mfpt_m = [0.22247,0.21818,0.21056,0.20433,0.20455457257189363,0.19717]
    dt_m = [3.00E-06,1.00E-06,7.50E-07,5.00E-07,2.50E-07,1e-07]
    
    #ax.errorbar(dt_a,mfpt_a,yerr=mfpt_a/np.sqrt(40),label='Agents')
    #ax.errorbar(dt_m,mfpt_m,yerr=mfpt_m/np.sqrt(40),label='Master')

    yerr_a = mfpt_a/np.sqrt(40)
    ax.fill_between(dt_a,mfpt_a-yerr_a,mfpt_a+yerr_a,
                    alpha=.5,color='gray')
    ax.plot(dt_a,mfpt_a,label='Agents',lw=2,
            marker='s',color='black',clip_on=False)

    yerr_m = mfpt_m/np.sqrt(50)
    ax.fill_between(dt_m,mfpt_m-yerr_m,mfpt_m+yerr_m,alpha=.4)
    ax.plot(dt_m,mfpt_m,label='Master',lw=2,
            marker='^',clip_on=False,markersize=10)
    
    #ax.errorbar(dt_m,mfpt_m,yerr=mfpt_m/np.sqrt(40),label='Master')


    ax.set_xlabel(r'\texttt{dt}',fontsize=size)
    ax.set_ylabel(r'Time (\si{\s})',fontsize=size)
    ax.set_title(r'Mean Time to Switch Velocity',fontsize=size)

    #ax.ticklabel_format(axis='x',style='scientific',scilimits=(0,0),)
    ax.set_xscale('log')
    ax.tick_params(axis='both',which='major', labelsize=size)

    ax.set_xlim(dt_a[0],dt_a[-1])

    ax.legend(fontsize=12,loc='upper left')
    
    plt.tight_layout()
    return fig
    
    

def switch_distributions():
    """
    MFPT to switch velocity
    """
        
    from scipy.stats import kstest

    # load agents MFPT
    #path = '../julia/mfpt1a'
    path = '../julia/mfptfigure1'
    options = ['nX=100','dt=1.0e-07','.txt']
    diffs = False

    mfpts_agents = get_mfpts(path,options,diffs)
    
    # load master MFPT
    #path = "../cluster_data/p1/a"
    path = "../cluster_data/figure1"
    #options = []
    options = ['nX=100','dt=1e-06','T=10','.txt']
    diffs = True

    mfpts_master = get_mfpts(path,options,diffs)

    dist_names = ['genexpon','expon']
    full_names = ['Gen. Exp.', 'Exp.']
    
    #fig = plt.figure(figsize=(FIGWIDTH_PX/MY_DPI,250/MY_DPI),dpi=MY_DPI)
    fig = plt.figure(figsize=(8,2.5))
    
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    na, binsa, _ = ax1.hist(mfpts_agents,bins=50,density=True,alpha=0.5)
    nm, binsm, _ = ax2.hist(mfpts_master,bins=50,density=True,alpha=0.5)
    
    #ax2.hist(mfpts_master[::2],bins=50,density=True,alpha=0.5)
    #ax2.hist(mfpts_master[1::2],bins=50,density=True,alpha=0.5)

    x = np.linspace(0,.75,1000)
    
    for i,dist_name in enumerate(dist_names):
        dist = getattr(sp.stats, dist_name)
        #param_a = dist.fit(mfpts_agents,loc=0)
        #param_m = dist.fit(mfpts_master,loc=0)

        param_a = dist.fit(mfpts_agents)
        param_m = dist.fit(mfpts_master)

        print(param_a,param_m)

        pdf_fitted_a = dist.pdf(x, *param_a[:-2],
                                loc=param_a[-2], scale=param_a[-1])
        pdf_fitted_m = dist.pdf(x, *param_m[:-2],
                                loc=param_m[-2], scale=param_m[-1])

        if i == 0:
            dashes = (5,0)
            ls = '-'
            lw = 5
        else:
            ls=''
            dashes = (2,2)
            lw = 3

        
        ax1.plot(x[pdf_fitted_a>0],pdf_fitted_a[pdf_fitted_a>0],
                 label=full_names[i],lw=lw,ls=ls,dashes=dashes)

        # fits make loc nonzero for master distribution.
        # so force positive points only.
        ax2.plot(x[pdf_fitted_m>0],pdf_fitted_m[pdf_fitted_m>0],
                 label=dist_name,lw=lw,dashes=dashes)

        #print(dist_name,i,pdf_fitted_a[:10])
        #print(dist_name,i,pdf_fitted_m[:10])

        test_data_a = kstest(mfpts_agents,dist_name,N=100,
                             args=tuple(param_a),
                             alternative='two-sided')

        test_data_m = kstest(mfpts_master,dist_name,N=100,
                             args=tuple(param_m),
                             alternative='two-sided')

        print('agents ks =',test_data_a[0],'p =',test_data_a[1],dist_name)
        print('master ks =',test_data_m[0],'p =',test_data_m[1],dist_name)

    ax1.set_title(r'\textbf{A} Agent-Based Model',fontsize=size,loc='left')
    ax2.set_title(r'\textbf{B} Master Equation',fontsize=size,loc='left')

    ax1.set_xlabel('Time (\si{\s})',fontsize=size)
    ax2.set_xlabel('Time (\si{\s})',fontsize=size)

    ax1.set_ylabel('Probability',fontsize=size)
    ax2.set_ylabel('Probability',fontsize=size)
    
    
    ax1.tick_params(axis='both',which='major', labelsize=size)
    ax2.tick_params(axis='both',which='major', labelsize=size)

    ax1.legend(fontsize=12)


    #ax1.set_xlim(0,x[-1])
    #ax2.set_xlim(0,x[-1])

    ax1.set_xlim(0,.4)
    ax2.set_xlim(0,.4)

    maxes = [np.amax(na),np.amax(nm)]
    
    ax1.set_ylim(0,np.amax([maxes])+1.5)
    ax2.set_ylim(0,np.amax([maxes])+1.5)

    plt.tight_layout()
    
    return fig



def get_translocations(path,options,args=3,skipn=1):
    
    fnames = []
    
    for root, dirs, files in os.walk(path):
        
        for file in files:
            #print(options)
            #print(file,all(x in file for x in options))

            if all(x in file for x in options):
                fname = os.path.join(root,file)

                fnames.append(fname)

    if 'L=800' in options:
        cutoff1 = 10000
        cutoff2 = len(fnames)
    else:
        cutoff1 = 0
        cutoff2 = len(fnames)
        
    print('Total seeds =', len(fnames[cutoff1:cutoff2]))

    translocation_data = np.zeros((len(fnames[cutoff1:cutoff2:skipn]),args))
    
    for i,fname in enumerate(fnames[cutoff1:cutoff2:skipn]):
        
        translocation_data[i,:] = np.loadtxt(fname)
                
    return translocation_data


def mfpt_translocation():

    fig = plt.figure(figsize=(8,5))
    ax11 = fig.add_subplot(221)
    ax12 = fig.add_subplot(222)
    
    ax21 = fig.add_subplot(223)
    ax22 = fig.add_subplot(224)

    x_list = np.linspace(0,200,10)

    # collect data from cluster 
    z0_list = [0,50,100,150,175]

    r_mean_list = []
    r_std_list = []
    r_prob_list = []

    # top row figure data
    for z0 in z0_list:
        # load all files into single array for each Z0
        data = get_translocations('../cluster_data/figure1_translocation',
                                  ['Z0='+str(z0),'dt=3e-06'])

        # average time to cross rigth
        #print(data)
        cr_idx = np.where(data[:,-1]==200)[0]
        r_mean = np.mean(data[cr_idx,1])
        r_std = np.std(data[cr_idx,1])

        cl_idx = np.where(data[:,-1]==0)[0]
        l_mean = np.mean(data[cl_idx,1])

        #if z0 == 0:
        #    print(cr_idx[:10])
        #    print(data[cr_idx[:10],:])

        assert(len(cl_idx) + len(cr_idx) == len(data[:,-1]))

        mean_tloc_both = np.mean(data[:,1])
        mean_switch_number_both = np.mean(data[:,0])

        mean_tloc_r = np.mean(data[cr_idx,1])
        mean_switch_number_r = np.mean(data[cr_idx,0])

        mean_tloc_l = np.mean(data[cl_idx,1])
        mean_switch_number_l = np.mean(data[cl_idx,0])

        mean_vel_mfpt = mean_tloc_both/mean_switch_number_both
        mean_vel_mfpt_r = mean_tloc_r/mean_switch_number_r
        mean_vel_mfpt_l = mean_tloc_l/mean_switch_number_l

        r_mean_list.append(r_mean)
        r_std_list.append(r_std)
        r_prob_list.append(len(cr_idx)/len(data[:,-1]))
        
        print('z0=',z0, 'mfpt to switch vel=',mean_vel_mfpt)
        #print('\t z0=',z0, 'mfpt to switch vel conditioned on right=',mean_vel_mfpt_r)
        #print('\t z0=',z0, 'mfpt to switch vel conditioned on left=',mean_vel_mfpt_l)
        #ax12.errorbar(z0,r_mean,yerr=r_std,color='gray',marker='o')
        
        print('\t prob exit left', len(cl_idx)/len(data[:,0]) , 'mean left', l_mean)
        print('\t prob exit right',len(cr_idx)/len(data[:,0])  ,'mean right', r_mean)


        
    
    # plot probability to escape
    ax11.plot(x_list,telegraph.pp(x_list,121,lam=1/.2,L=200),label='Analytical')
    ax11.plot(z0_list,r_prob_list,
             label='Numerical',color='gray',marker='s',ls='',
             markersize=8,clip_on=False)

    # plot MFPT
    ax12.plot(x_list,telegraph.tp(x_list,121,lam=1/.2,L=200),label='Analytical')

    ax12.errorbar(z0_list,r_mean_list,
                 yerr=r_std_list/np.sqrt(len(r_std_list)),
                 color='gray',marker='s',label='Numerical',ls='',
                 markersize=8,clip_on=False,elinewidth=2,capsize=5)


    L_list = [200,400,600,800]
    L_domain = np.linspace(200,800,1000)


    r_mean_list = []
    r_std_list = []
    r_prob_list = []

    print('========================')
    
    # bottom row figure data
    for L in L_list:
        # load all files into single array for each Z0
        if L == 200:
            data = get_translocations('../cluster_data/figure1_translocation',
                                      ['Z0=0','dt=3e-06'])

        else:
            data = get_translocations('../cluster_data/figure1_translocation',
                                      ['L='+str(L),'dt=3e-06'],args=5,
                                      skipn=1)

        # average time to cross rigth
        #print(data)
        #print(data[:,-1])
        cr_idx = np.where(data[:,2]==L)[0]
        r_mean = np.mean(data[cr_idx,1])
        r_std = np.std(data[cr_idx,1])

        cl_idx = np.where(data[:,2]==0)[0]
        l_mean = np.mean(data[cl_idx,1])

        #if z0 == 0:
        #    print(cr_idx[:10])
        #    print(data[cr_idx[:10],:])

        #print(len(cl_idx), len(cr_idx), len(data[:,-1]))
        
        #assert(len(cl_idx) + len(cr_idx) == len(data[:,-1]))
        

        mean_tloc_both = np.mean(data[:,1])
        mean_switch_number_both = np.mean(data[:,0])

        mean_tloc_r = np.mean(data[cr_idx,1])
        mean_switch_number_r = np.mean(data[cr_idx,0])

        mean_tloc_l = np.mean(data[cl_idx,1])
        mean_switch_number_l = np.mean(data[cl_idx,0])

        mean_vel_mfpt = mean_tloc_both/mean_switch_number_both
        mean_vel_mfpt_r = mean_tloc_r/mean_switch_number_r
        mean_vel_mfpt_l = mean_tloc_l/mean_switch_number_l

        r_mean_list.append(r_mean)
        r_std_list.append(r_std)
        r_prob_list.append(len(cr_idx)/len(data[:,-1]))
        
        #print('L=',L, 'mfpt to switch vel=',mean_vel_mfpt)
        #print('\t z0=',z0, 'mfpt to switch vel conditioned on right=',mean_vel_mfpt_r)
        #print('\t z0=',z0, 'mfpt to switch vel conditioned on left=',mean_vel_mfpt_l)
        #ax12.errorbar(z0,r_mean,yerr=r_std,color='gray',marker='o')

        print('L=',L)
        print('\t prob exit left', len(cl_idx)/len(data[:,0]) , 'mean left', l_mean)
        print('\t prob exit right',len(cr_idx)/len(data[:,0])  ,'mean right', r_mean)

    
    ax21.plot(L_domain,telegraph.e0p(L_domain,121,1/.22))

    ax21.plot(L_list,r_prob_list,
              label='Numerical',color='gray',marker='s',ls='',
              markersize=8,clip_on=False)


    ax22.plot(L_domain,telegraph.t0p(L_domain,121,1/.22))
    
    ax22.errorbar(L_list,r_mean_list,
                  yerr=r_std_list/np.sqrt(len(r_std_list)),
                  color='gray',marker='s',label='Numerical',ls='',
                  markersize=8,clip_on=False,elinewidth=2,capsize=5)

    
    ax11.legend(fontsize=12)
    ax12.legend(fontsize=12)
    #ax12.set_title('MFPT to exit right given initial positive velocity',fontsize=size)

    #ax11.set_title(r'\textbf{A} $z=\SI{200}{\nm}$',loc='left',fontsize=size)
    #ax12.set_title(r'\textbf{B} $z=\SI{200}{\nm}$',loc='left',fontsize=size)
    #ax21.set_title(r'\textbf{C} $z=L$',loc='left',fontsize=size)
    #ax22.set_title(r'\textbf{D} $z=L$',loc='left',fontsize=size)

    ax11.set_title(r'\textbf{A}',loc='left',fontsize=size)
    ax12.set_title(r'\textbf{B}',loc='left',fontsize=size)
    ax21.set_title(r'\textbf{C}',loc='left',fontsize=size)
    ax22.set_title(r'\textbf{D}',loc='left',fontsize=size)
    
    
    ax11.set_xlabel('Position (\si{nm})',fontsize=size)
    ax12.set_xlabel('Position (\si{nm})',fontsize=size)
    ax21.set_xlabel('Length (\si{nm})',fontsize=size)
    ax22.set_xlabel('Length (\si{nm})',fontsize=size)
    
    ax11.set_ylabel(r'Probability',fontsize=size)
    ax12.set_ylabel(r'MFPT',fontsize=size)
    ax21.set_ylabel(r'Probability',fontsize=size)
    ax22.set_ylabel(r'MFPT',fontsize=size)

    ax11.tick_params(axis='both',which='major', labelsize=size)
    ax12.tick_params(axis='both',which='major', labelsize=size)
    ax21.tick_params(axis='both',which='major', labelsize=size)
    ax22.tick_params(axis='both',which='major', labelsize=size)

    
    ax11.set_xlim(-5,200)
    ax12.set_xlim(-5,200)
    
    ax11.set_ylim(0,1)
    ax12.set_ylim(0,8)

    tickpos = [0,2,4,6,8]
    ax12.set_yticks(tickpos,tickpos)

    #plt.locator_params(axis="y", integer=True, tight=True)

    #import matplotlib.ticker as ticker
    #for axis in [ax2.yaxis]:
    #    axis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    #plt.show()
    #plt.subplots_adjust(wspace=2)
    plt.tight_layout()
    
    return fig

def generate_figure(function, args, filenames, dpi=MY_DPI):
    # workaround for python bug where forked processes use the same random 
    # filename.
    #tempfile._name_sequence = None;

    fig = function(*args)

    if type(filenames) == list:
        for name in filenames:
            fig.savefig(name,dpi=dpi)
    else:
        fig.savefig(filenames,dpi=dpi)
    
def main():
        
    # listed in order of Figures in paper
    figures = [
        #(cylinder_sideways,[],['f_cylinder_sidways.png']),
        #(cylinder,[],['f_cylinder.pdf']),
        #(agent_example,[],['f_agent_example.pdf']),
        (langevin_vs_agents,[],['f_langevin_vs_agents.pdf']),
        (motor_distributions,[],['f_motor_distribution.pdf']),
        (master_vs_agents,[],['f_master_vs_agents.pdf']),
        (mva_time,[],['f_mva_time.pdf']),
        (velocity_mfpts,[],['f_velocity_mfpts.pdf']),
        (switch_distributions,[],['f_switch_distributions.pdf']),
        (mfpt_translocation,[],['f_mfpt_translocation.pdf']),
        
    ]
    
    for fig in figures:
        generate_figure(*fig)

if __name__ == "__main__":
    main()

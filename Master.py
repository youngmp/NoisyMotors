"""
Modified master equation

"""

import argparse

import libMotorPDE as lib
#import libMaster as libm
from libMotorPDE import inverse_transform_sampling as inv_sample
from MotorPDE import MotorPDE


import os
#import sys
import time
import pickle

import numpy as np


#from scipy.interpolate import interp1d


pi = np.pi
exp = np.exp
sqrt = np.sqrt
Sqrt = np.sqrt


def fn_test(t,om=100):
    """
    velocity function used for testing.
    """
    #print(np.cos(om*t))
    #print(np.cos(om*t)**(1/3))
    
    sign = np.sign(np.cos(om*t))
    
    return 100*sign*np.abs(np.cos(om*t))**(1/10)


class Master(MotorPDE):
    
    def __init__(self,**kwargs):
        

        """

        Species X prefers negative vel.

        Parameters
        ----------
        timeout: save random state in this many seconds
        use_last: if true, search for saved random state and initialize.
        """
        
        defaults = {'alpha':14,
                    'beta':126,
                    'nX':100,
                    'nY':100,
                    'zeta':0.048,
                    'dt':0.01,
                    'switch_v':100,
                    'gamma':0.322,
                    'p1':4,
                    'X0':1,
                    'Y0':1,
                    'V0':0,
                    'Z0':0,
                    'A':5,
                    'B':5.5,
                    'T':5,
                    'seed':0,
                    'extension':True,
                    'store_position':False,
                    'store_draws':False,
                    'force_pos_type':'linear',
                    'U':None,
                    'source':True,
                    'irregular':False,
                    'timeout':False,
                    'use_last':False,
                    'terminate_on_boundary':False,
                    'L':200
                    }


        # define defaults if not defined in kwargs
        for (prop, default) in defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        # update dt keyword argument
        kwargs['dt'] = self.dt
        
        MotorPDE.__init__(self, **kwargs)
        
        # spatial discretization
        #self.dx = (self.B-self.A0)/self.N

        if self.irregular:
            print('CFL',self.switch_v*self.dt/np.amin(self.dx),'irregular')
            #print(np.amin(self.dx),np.amax(self.dx),self.x[1]-self.x[0])
        else:
            print('CFL',self.switch_v*self.dt/self.dx)
            

        #print('sdfasfdafd', self.irregular)
        # display parameters
        lib.disp_params(self)
        

        # define parameters for convenience/computational efficiency
        self.bAB = self.beta*(self.A-self.B)
        
        self.k = self.p1*self.gamma  # spring constant

        # if use storage, preallocate time data
        if self.store_position:
            self.store_state = True
            self.posX = np.zeros((self.TN,len(self.x)))
            self.posY = np.zeros((self.TN,len(self.x)))
            self.t = np.linspace(0,self.T,self.TN)
            
            self.mean_posX = np.zeros(self.TN)
            self.mean_posY = np.zeros(self.TN)
            size = self.TN
        else:
            self.store_state = False
            self.posX = np.zeros((1,len(self.x)))
            self.posY = np.zeros((1,len(self.x)))
            
            self.mean_posX = np.zeros(1)
            self.mean_posY = np.zeros(1)
            self.t = np.linspace(0,self.T,)
            size = 1

        self.pos_drawsX = []
        self.pos_drawsY = []

        # break initial symmetry:
        #print(self.x[self.A_idx:]-self.A)
        #print(np.exp((self.x[self.A_idx+1:]-self.A))/20)

        arr1 = self.x[:self.A_idx+1]
        al = self.alpha
        be = self.beta
        v0abs = np.abs(self.V0)
        
        pref_ss = al*be*np.exp((arr1-self.A)*be/v0abs)/(v0abs*(al+be))

        c = 1-np.exp(-be*(self.B-self.A)/v0abs)
        c1 = al*be/(v0abs*(al*c+be))

        arr2 = self.x[self.A_idx:]
        nonpref_ss = c1*np.exp(-be*(arr2-self.A)/v0abs)
            
        
        #self.posX[0,self.A_idx-1:] = np.exp(-(self.x[self.A_idx-1:]-self.A))/20
        #self.posY[0,:self.A_idx] = np.exp((self.x[:self.A_idx]-self.A))/10

        self.posX[0,self.A_idx:] = nonpref_ss
        self.posY[0,:self.A_idx+1] = pref_ss


        if False:
            import matplotlib.pyplot as plt
            
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax1.set_title('inits')
            groundt = lib.ground_truth(self)
            ax1.plot(self.x,self.posX[0,:])
            ax1.plot(self.x,groundt)
            
            ax2.plot(self.x,self.posY[0,:])
            
            plt.show()

        
        #self.posX[0,self.A_idx-1:] = np.random.rand(len(self.posX[0,self.A_idx-1:]))/10
        #self.posY[0,:self.A_idx] = 0

        
        self.X = np.zeros(size,dtype=int)
        self.Y = np.zeros(size,dtype=int)
        self.V = np.zeros(size)
        self.Z = np.zeros(size)
        self.FX = np.zeros(size)
        self.FY = np.zeros(size)
        
        self.varX = np.zeros(size)
        self.varY = np.zeros(size)
        
        # initialize with nonzero to get the sampling to work
        #self.posX[0,:] = np.linspace(0,1,self.N)*.1
        #self.posY[0,:] = np.linspace(0,1,self.N)*.1

        self.X[0] = self.X0
        self.Y[0] = self.Y0
        

        self.switch_times = []
        self.V[0] = self.V0
        self.Z[0] = self.Z0
        
            
    def p(self,x):
        """
        p1 = 4 # pN
        gamma = 0.322 # /nm

        assume linear for computational efficiency
        """

        #return x#/self.gamma
        
        #if (self.force_pos_type == 'linear') or (self.force_pos_type == 'lin'):
        #    return x*self.k
        return x*self.k
        
        #elif self.force_pos_type == 'exp':
        #    return self.p1*(exp(self.gamma*x)-1)
            
        #else:
        #    raise Exception('I dont recognize this type of force-position curve',
        #                    self.force_pos_type)
            
    def kp(self,i,n):
        """
        i: motor state
        n: total motor number
        """
        return self.alpha*(n-i)

    def si(self,i,v):
        """
        km(i) + gm(i)
        """
        
        if v <= 0:
            return self.beta*i
        else:
            return self.beta*i + i*v/self.B

    def gmHat(self,i):
        return i/(self.B-self.A)

    def detach(self,V,preferred_direction='down'):
        """
        Detachment rate.
        rate_extra becomes greater than 1 when in non-preferred direction.
        """
        if V != 0:
            
            if self.B > self.A:
                rate_extra = 1/(1-exp(self.bAB/np.abs(V)))
            else:
                rate_extra = 1#/self.dt/self.beta # just to make probability = 1
        else:
            rate_extra = 1

        # check direction of motion.
        if preferred_direction == 'down':
            # V > 0 means lam1 = 1. Else, lam1 = 0.
            if V > 0:
                lam = 1
            else:
                lam = 0
            #lam = np.heaviside(V, 0)
        else:
            if V > 0:
                lam = 0
            else:
                lam = 1
            #lam = np.heaviside(-V, 0)

            #print(self.beta*(1-lam))
        
        # p2 = self.X[j_prev]*(self.beta*((1-lam1)+rate_extra*lam1))*self.dt

        #if np.isnan(self.beta*((1-lam)+rate_extra*lam)):
        #    print(self.beta*((1-lam)+rate_extra*lam),self.beta,lam,rate_extra,rate_extra*lam)
        
        return self.beta*((1-lam)+rate_extra*lam)

    @staticmethod
    def update_state(tot, r, p1, p2):

        if r < p1:
            tot += 1
        elif (r >= p1) and (r < p1 + p2):
            tot -= 1

        #print(r,p1,p2)
            
        return tot



    def run_master(self):
        import warnings
        #np.errstate(runtime='raise')
        np.random.seed(self.seed)

        # random thing for plotting
        plot_counter = 0
        plot_counter2 = 0
        plot_counter3 = 0
        
        # track which side you are on after switch
        side = 0

        # track total real-world time
        start_time = time.time()
        
        
        self.i = 1
        
        while self.i < self.TN:

            #print(self.posX[self.i-1,:])
            #time.sleep(5)
            #print(self.X[0],self.Y[0],self.X0,self.Y0)
            #time.sleep(5)

            # check total run time of the simulation
            if self.i % 1000 == 0 and self.timeout:
                current_time = time.time()
                total_time = current_time - start_time
                #print(total_time)

                # if reach timeout, break.
                if total_time >= 60*60*20:
                    print('{:<10s}{:<12s}'.format('T', 'TN'))
                    print('{:<12.2E}{:<10d}'.format(self.i*self.TN,self.i))


                    break
                
            # indices are updated depending on store_position
            # if no store, then only save data from the previous index
            # if yes store, then save data from all indices.
            if self.store_state:
                jm = self.i-1
                j_current = self.i
            else:
                jm = 0
                j_current = jm

            if self.store_position:
                jj = jm
            else:
                jj = 0

            # update vesicle center of mass
            self.Z[j_current] = self.Z[jm] + self.dt*self.V[jm]

            if self.terminate_on_boundary:
                
                if self.Z[j_current] >= self.L:
                    self.cross = self.L
                    self.passage_time = self.i*self.dt
                    break
                    
                elif self.Z[j_current] < 0:
                    self.cross = 0
                    self.passage_time = self.i*self.dt
                    break

                else:
                    self.cross = -1
                    self.passage_time = -1

            
            # get motor positions
            if self.extension:
                
                # PDE motor head positon update
                if False:# and (self.i-1 % int(self.TN/1000) == 0):
                    print(jm)

                    import matplotlib.pyplot as plt
                    import matplotlib as mpl
                    mpl.rcParams['text.usetex'] = False
                    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
                    #matplotlib.use('TkAgg')

                    fig = plt.figure(figsize=(4, 5))
                    ax1 = fig.add_subplot(211)
                    ax2 = fig.add_subplot(212)

                    ax1.plot(self.x, self.posX[jm, :])
                    ax2.plot(self.x, self.posY[jm, :])


                    ax1.set_xlim(4, self.B)
                    #ax1.set_ylim(0, 1)

                    ax2.set_xlim(4, self.B)
                    #ax2.set_ylim(0, .7)

                    plt.tight_layout()
                    plt.savefig('/home/youngmp/mov_master2/' + str(plot_counter2) + '.png',
                                dpi=100)
                    plot_counter2 += 1

                
                #dX = self.upwind(self.t[jm],self.posX[jm,:],self.V[jm])
                #dY = self.upwind(self.t[jm],self.posY[jm,:],-self.V[jm])

                #print('dX',dX[-5:],self.posX[jm,-5:])

                self.posX[j_current,:] = self.posX[jm,:] + self.dt*self.upwind(self.t[jm],self.posX[jm,:],self.V[jm])
                self.posY[j_current,:] = self.posY[jm,:] + self.dt*self.upwind(self.t[jm],self.posY[jm,:],-self.V[jm])

                # draw positions from motor head distribution
                if np.add.reduce(self.posX[jj]) != 0:
                #if self.posX[jj].sum(axis=0) != 0:
                    
                    Xs = inv_sample(self,self.posX[jj,:],self.X[jm],
                                    spec='X',vel=self.V[jm])
                    
                    if self.store_draws:
                        self.pos_drawsX.append(Xs)
                    #Xs = np.ones(np.sum(self.X))*self.A
                else:
                    Xs = np.zeros(self.X[jm])
                    
                if np.add.reduce(self.posY[jj]) != 0:
                #if self.posY[jj].sum(axis=0) != 0:
                    Ys = inv_sample(self,self.posY[jj,:],self.Y[jm],
                                    spec='Y',vel=self.V[jm])
                    if self.store_draws:
                        self.pos_drawsY.append(Ys)
                    #Ys = np.ones(np.sum(self.Y))*self.A
                else:
                    Ys = np.zeros(self.Y[jm])                    
                

                #print(Xs,Ys)
                #time.sleep(3)
                #if self.store_position and Xs.size != 0:
                #    self.mean_posX[j_current] = np.mean(Xs)
                #elif self.store_position and Xs.size == 0:
                #    self.mean_posX[j_current] = np.nan
            
                #if self.store_position and Ys.size != 0:
                #    self.mean_posY[j_current] = np.mean(Ys)
                #if self.store_position and Ys.size == 0:
                #    self.mean_posY[j_current] = np.nan
                    
                    
                if self.X[jm] == 0 and False:
                    t = self.dt*self.i
                    print('t',t,'Xs',Xs,'mean',self.mean_posX[j_current])
                    
                    fig  = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(self.posX[j_current,:])
                    plt.show(block=True)
                    plt.close()
                    
                    time.sleep(1)
            else:
                Xs = np.ones(np.sum(self.X))*self.A
                Ys = np.ones(np.sum(self.Y))*self.A


            # add forces
            #FX = np.add.reduce(self.p(Xs))
            #FY = np.add.reduce(self.p(Ys))
            FX = np.add.reduce(Xs*self.k)
            FY = np.add.reduce(Ys*self.k)
            
            self.FX[j_current] = FX
            self.FY[j_current] = FY

            #print(FY,len(Ys),np.sum(self.Y))
            
            ## velocity update
            # update velocity based on forces
            if self.U is None:
                #print((FY - self.zeta*self.V[jm]),FY,self.V[jm],np.sum(self.X),np.sum(self.Y))
                #self.V[j_current] = self.V[jm] + .01*(-FX + FY - self.zeta*self.V[jm])
                self.V[j_current] = (-FX + FY)/self.zeta
                #time.sleep(.1)
            else:
                #print(t,U)
                if callable(self.U):
                    Uval = self.U(self.i*self.dt)
                elif isinstance(self.U,float) or isinstance(self.U,int):
                    Uval = self.U
                self.V[j_current] = Uval
            
            # draw random number
            # used to determine probability of attachment or detachment
            #r = np.random.rand()
            rx = np.random.rand()
            ry = np.random.rand()

            ## update X (down preferred)
            
            # X attachment probabilty
            attach = self.nX-self.X[jm]
            p1 = self.alpha*attach*self.dt

            # X detachment probability
            detach = self.detach(self.V[jm],preferred_direction='down')
            p2 = self.X[jm]*detach*self.dt

            ## update Y (up preferred)
            # Y attachment probabilty
            attach = self.nY-self.Y[jm]
            s1 = self.alpha*attach*self.dt

            # Y detachment probability
            detach = self.detach(self.V[jm],preferred_direction='up')            
            s2 = self.Y[jm]*detach*self.dt

            
            
            # update total
            """
            if r < p1:
                self.X[j_current] = self.X[jm] + 1
                self.Y[j_current] = self.Y[jm]
            elif (r >= p1) and (r < p1+p2):
                self.X[j_current] = self.X[jm] - 1
                self.Y[j_current] = self.Y[jm]
            elif (r >= p1+p2) and (r < p1+p2+s1):
                self.Y[j_current] = self.Y[jm] + 1
                self.X[j_current] = self.X[jm]
            elif (r >= p1+p2+s1) and (r < p1+p2+s1+s2):
                self.Y[j_current] = self.Y[jm] - 1
                self.X[j_current] = self.X[jm]
            """
            
            # update total X
            self.X[j_current] = self.update_state(self.X[jm],rx,p1,p2)
            
            # update total Y
            self.Y[j_current] = self.update_state(self.Y[jm],ry,s1,s2)

            if self.i % 100 == 0:
                #print(self.i,self.X[jm],self.Y[jm],self.V[jm])
                #print(np.add.reduce(self.posX[jj]))
                #print(Xs)
                pass

            if False:# and (self.i-1 % int(self.TN/1000) == 0):
                print(jm)
                
                import matplotlib.pyplot as plt
                import matplotlib as mpl
                mpl.rcParams['text.usetex'] = False
                mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
                #matplotlib.use('TkAgg')

                fig = plt.figure(figsize=(4, 5))
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                
                ax1.plot(self.x, self.posX[jm, :])
                ax2.plot(self.x, self.posY[jm, :],color='tab:green')

                #ax1.plot(self.x, dX)
                #ax2.plot(self.x, dY)

                time_label = "{:02.2f}".format(self.i*self.dt)
                v_label = "{:02.2f}".format(self.V[jm])
                x_label = "{:02d}".format(self.X[jm])
                y_label = "{:02d}".format(self.Y[jm])
                ax1.set_title('t=' + time_label
                              + '; v=' + v_label
                              + '; X=' + x_label
                              + '; Y=' + y_label)

                ax1.set_ylabel('Down Motors (Probability)')
                ax2.set_ylabel('Up Motors (Probability)')

                ax2.set_xlabel('Attached Motor Head Position (nm)')
                #ax2.set_ylabel('Position Count Y')

                ax1.set_xlim(4, self.B)
                #ax1.set_ylim(0, 1)

                ax2.set_xlim(4, self.B)
                #ax2.set_ylim(0, .7)

                plt.tight_layout()
                plt.savefig('/home/youngmp/mov_master/' + str(plot_counter) + '.png',
                            dpi=100)

                ax1.clear()
                ax2.clear()
                plot_counter += 1

            
            if False and(self.i % int((self.TN)/10) == 0):

                t = self.i*self.dt
                X = self.X[jm]
                Y = self.Y[jm]
                V = self.V[j_current]
                Z = self.Z[j_current]

                #print('{:<10s}{:<12s}'.format('T', 'TN'))
                #print('{:<12.2E}{:<10d}'.format(self.i * self.TN, self.i))

                times = np.array(self.switch_times)[np.array(self.switch_times)>.1]
                
                #print('t=%.3f,V=%.2f,X=%d,Y=%d,MFPT est.=%.4f'
                #      % (t,V,X,Y,np.mean(np.diff(times))))

                print('t=%.3f,V=%.2f,X=%d,Y=%d,Z=%.2f'
                      % (t,V,X,Y,Z))
                
                #print(self.posX[self.i, :10])
                #print(p2,self.t[i],self.X[i-1],self.V[i-1])
            
            if self.posX[jj,0] > 1000:
                raise ValueError("bounds exceeded")

            # record when velocity switches
            if (side == 0) and (self.V[jm] >= self.switch_v):
                side = 1
                #print(self.V[jm])
                self.switch_times.append(self.i*self.dt)

            if (side == 1) and (self.V[jm] <= -self.switch_v):
                side = 0
                #print(self.V[jm])
                self.switch_times.append(self.i*self.dt)

            #print(self.i)
            self.i += 1

            

            
        # end while loop
        times = np.array(self.switch_times)[np.array(self.switch_times)>.01]
        #print(times)
        self.mfpt = np.mean(np.diff(times))

        print('MPFT=',self.mfpt)
        print('Total number of switches:',len(self.switch_times))



def main():
    
    import matplotlib.pyplot as plt

    """
    parser = argparse.ArgumentParser(description='run the agent-based '
                                     'Myosin motor model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-s','--seed',default=0,type=int,
                        help='Set random number seed for simulation')
    #parser.add_argument('-z','--zeta',default=0,type=np.float64,help='Set viscous drag')

    parser.add_argument('-T','--Tfinal',default=10,type=np.float64,
                        help='Set total simulation time')
    parser.add_argument('-d','--dt',default=0.001,type=np.float64,
                        help='Set time step factor')

    
    #parser.add_argument('--save_switch',dest='switch',action='store_true',
    #                    help='If true, save switching rates')
    #parser.add_argument('--no-save_switch',dest='switch',action='store_false',
    #                    help='If true, save switching rates')
    #parser.set_defaults(switch=False)

    
    parser.add_argument('--storage',dest='storage',action='store_true',
                        help=('If true, store all trace'
                              'data to memory (for plotting)'))
    parser.add_argument('--no-storage',dest='storage',action='store_false')
    parser.set_defaults(storage=True)

    parser.add_argument('--ext',dest='ext',action='store_true',
                        help=('If true, use force extension '
                              'in preferred direction. '
                              'else do not udate position in '
                              'prefeerred direction'))
    parser.add_argument('--no-ext',dest='ext',action='store_false')
    parser.set_defaults(ext=True)

    parser.add_argument('-X','--nX',default=100,type=int,
                        help='Set total motor number (X left preferred)')
    parser.add_argument('-Y','--nY',default=100,type=int,
                        help='Set total motor number (Y right preferrred)')

    args = parser.parse_args()
    print('args',args)
    
    d_flags = vars(args)
    """
    
    # options not from terminal flags
    options = {'T':.5,
               'nX':400,'nY':400,
               'dt':1e-5,
               'seed':0,
               'zeta':0.7,
               'A':5,'B':5.1,
               'alpha':14,'beta':126,
               'p1':4/4,'gamma':0.322,
               'switch_v':164,'U':164,
               'X0':10,'Y0':10,
               'A0':0,
               'ivp_method':'euler',
               'N':200,
               'N2':150,
               'margin':2.4,
               'source':True,
               'extension':True,
               'store_position':True,
               'timeout':True}
    
    #kwargs = {**d_flags,**options}
    kwargs = options

    a = Master(**kwargs)

    # load MFPT data if it exists
    fname = 'data/mfpt_'+lib.fname_suffix(ftype='.txt',
                                          exclude=['store_state',
                                                   'store_position',
                                                   'X0','Y0',
                                                   'ivp_method',
                                                   'extension',
                                                   'U'],
                                          **kwargs)
    
    if os.path.isfile(fname) and False:
        switch_times = np.loadtxt(fname)
    
    else:
        t0 = time.time()
        a.run_master()
        t1 = time.time()
        print('*\t Run time',t1-t0)
        
        skipn = 1
        
        switch_times = a.switch_times
        np.savetxt(fname,switch_times)
        

        
    if False:
        diffs = np.diff(switch_times)
        means = np.cumsum(diffs)/np.arange(1,len(switch_times))
        
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        
        ax1.plot(switch_times[:-1],means)
        ax1.plot([0,switch_times[-2]],[means[-1],means[-1]],color='gray',
                 label='final mean value')
        ax1.legend()
    
    if False:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        ax1.plot(a.varX)
        ax2.plot(a.varY)
    
    if False:
        print('mean FX',np.nanmean(a.FX[skipn:]))
        print('mean FY',np.mean(a.FY[skipn:]))
    
        print('mean X',np.nanmean(a.X[skipn:]))
        print('mean Y',np.mean(a.Y[skipn:]))
        
        print('mean V',np.mean(a.V[skipn:]))
    
    if False:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        ax1.hist(a.X,alpha=.4,bins=10,density=True)
        ax1.hist(a.Y,alpha=.4,bins=20,density=True)
        
        ax2.hist(a.FX,alpha=.4,bins=10,density=True)
        ax2.hist(a.FY,alpha=.4,bins=20,density=True)
        
        
    #plt.show(block=True)
    #plt.close()
    
    # will only plot with --use-storage is enabled
    #libm.plot_traces(a)
    #libm.plot_heatmap(a)
    
    if False:#not(cluster):
        plt.show(block=True)


if __name__ == "__main__":
    main()

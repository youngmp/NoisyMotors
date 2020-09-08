# testing ground for master equation


import argparse

import libMotorPDE as libp
import libMaster as libm
from MotorPDE import MotorPDE

#import os
#import sys
import time
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#from scipy.interpolate import interp1d

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] 
#matplotlib.use('TkAgg')

pi = np.pi
exp = np.exp
sqrt = np.sqrt
Sqrt = np.sqrt


def fn_test(t,om=100):
    #print(np.cos(om*t))
    #print(np.cos(om*t)**(1/3))
    
    sign = np.sign(np.cos(om*t))
    
    return 100*sign*np.abs(np.cos(om*t))**(1/10)


class Master(MotorPDE):

    
    def __init__(self,**kwargs):

        
        defaults = {'alpha':14,
                    'beta':126,
                    'nX':100,
                    'nY':100,
                    'zeta':0.048,
                    'dt_factor':0.01,
                    'switch_v':100,
                    'gamma':0.322,
                    'p1':4,
                    'X0':1,
                    'Y0':1,
                    'A':5,
                    'B':5.5,
                    'T':5,
                    'seed':0,
                    'ext':True,
                    'use_storage':True,
                    'force_pos_type':'linear',
                    'U':None
                    }

        
        # define defaults if not defined in kwargs
        for (prop, default) in defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
        
        kwargs['dt'] = self.dt_factor/(self.alpha+self.beta+self.nX+self.nY)
        
        
        MotorPDE.__init__(self,**kwargs)
        
        self.dx = (self.B-self.A0)/self.N
        
        print('CFL',self.switch_v*self.dt/self.dx)
        
        libp.disp_params(self)
        
        self.k = self.p1*self.gamma  # spring constant
        
        if self.use_storage:
            self.t = np.linspace(0,self.T,self.TN)
        else:
            self.t = np.linspace(0,self.T,)
    
    def p(self,x):
        """
        p1 = 4 # pN
        gamma = 0.322 # /nm
        """

        #return x#/self.gamma
        if (self.force_pos_type == 'linear') or (self.force_pos_type == 'lin'):
            return x*self.p1*self.gamma
        
        elif self.force_pos_type == 'exp':
            return self.p1*(exp(self.gamma*x)-1)
            
        else:
            raise Exception('I dont recognize this type of force-position curve',
                            self.force_pos_type)
            
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
        
    
    def run_states(self):

        np.random.seed(self.seed)

        if self.use_storage:
            size = self.TN
        else:
            size = 1

        self.X = np.zeros(size,dtype=int)
        self.Y = np.zeros(size,dtype=int)
        self.V = np.zeros(size)
        self.FX = np.zeros(size)
        self.FY = np.zeros(size)
        
        #self.posX = np.zeros((self.TN,self.N))
        #self.posY = np.zeros((self.TN,self.N))
        
        self.posX = np.zeros(self.N)
        self.posY = np.zeros(self.N)
        
        # force at specific motor number
        self.force_at_numberX = np.zeros(size)
        self.force_at_numberY = np.zeros(size)
        
        # initialize with nonzero to get the sampling to work
        #self.posX[0,:] = np.linspace(0,1,self.N)*.1
        #self.posY[0,:] = np.linspace(0,1,self.N)*.1

        self.X[0] = self.X0
        self.Y[0] = self.Y0
        
        self.meanPosX = np.zeros(size)
        self.meanPosY = np.zeros(size)

        self.switch_times = []

        # keep track of which side you are on after switch
        side = 0
        
        self.V[0] = 0
        
        #v = -121
        
        self.i = 1
        while self.i < self.TN:

            
            if self.use_storage:
                j_prev = self.i-1
                j_current = self.i
            else:
                j_prev = 0
                j_current = j_prev


            # PDE positon update
            dX = self.upwind(self.t[j_prev],self.posX,self.V[j_prev])
            dY = self.upwind(self.t[j_prev],self.posY,-self.V[j_prev])
            
            self.posX = self.posX + self.dt*dX
            self.posY = self.posY + self.dt*dY
            
            # draw positions
            if np.sum(self.posX) != 0:
                Xs = libp.inverse_transform_sampling(self,self.posX,
                                                     self.X[j_prev])

            else:
                Xs = np.zeros(self.X[j_prev])
                
            if np.sum(self.posY) != 0:
                Ys = libp.inverse_transform_sampling(self,self.posY,
                                                     self.Y[j_prev])
            else:
                Ys = np.zeros(self.Y[j_prev])
                
            # save mean position
            #self.meanPosX[j_current] = np.mean(Xs)
            #self.meanPosY[j_current] = np.mean(Ys)
            
            # force update
            
            FX = np.sum(self.p(Xs))
            FY = np.sum(self.p(Ys))
            
            if self.X[j_prev] == 11:
                self.force_at_numberX[j_prev] = FX
                
            if self.Y[j_prev] == 11:
                self.force_at_numberY[j_prev] = FY
            
            
            if self.V[j_prev] < 0:
            
                FX2 = self.p(self.X[j_prev]*self.A)
                FY2 = np.sum(self.p(Ys))
                #print(FX)
            else:
                FX2 = np.sum(self.p(Xs))
                FY2 = self.p(self.Y[j_prev]*self.A)
            
            
            self.FX[j_current] = FX  # /self.X[j_prev]
            self.FY[j_current] = FY  # /self.Y[j_prev]
            
            
            # velocity update

            if self.U is None:
                self.V[j_current] = (-FX+FY)/(self.zeta)
            else:
                #print(t,U)
                if callable(self.U):
                    Uval = self.U(self.i*self.dt)
                elif isinstance(self.U,float) or isinstance(self.U,int):
                    Uval = self.U
                self.V[j_current] = Uval
            
            # draw random number
            rx = np.random.rand()
            ry = np.random.rand()

            # update X
            # decay rate
            p1 = self.alpha*(self.nX-self.X[j_prev])*self.dt
            #p2 = self.X[i-1]*(self.beta+np.abs(v)*np.heaviside(v,0)/(self.B-self.A))*self.dt #

            if self.V[j_prev] == 0:
                rate_extra = 1
            else:
                rate_extra = 1/(1-exp(self.beta*(self.A-self.B)/np.abs(self.V[j_prev])))

            lam1 = np.heaviside(self.V[j_prev],0)
            
            # decay rate
            p2 = self.X[j_prev]*(self.beta*((1-lam1)+rate_extra*lam1))*self.dt

            if (rx < p1):
                self.X[j_current] = self.X[j_prev] + 1
            elif (rx >= p1) and (rx < p1+p2):
                self.X[j_current] = self.X[j_prev] - 1
            else:
                self.X[j_current] = self.X[j_prev]

            # update Y
            s1 = self.alpha*(self.nY-self.Y[j_prev])*self.dt
            
            # decay rate
            #s2 = self.Y[i-1]*(self.beta+np.abs(v)*np.heaviside(-v,0)/(self.B-self.A))*self.dt 
            if self.V[j_prev] == 0:
                rate_extra = 1
            else:
                rate_extra = 1/(1-exp(self.beta*(self.A-self.B)/np.abs(self.V[j_prev])))

            lam2 = np.heaviside(-self.V[j_prev],0)
            
            # decay rate
            s2 = self.Y[j_prev]*(self.beta*((1-lam2) + rate_extra*lam2))*self.dt

            if True and(self.i % int((self.TN)/20) == 0):
                
                if False:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(self.x,self.posX)
                    ax.set_xlim(3,5)
                    plt.show(block=True)
                    plt.close()
                    #time.sleep(1)
                #print('t=%.2f,p1=%.4f,p2=%.4f,X=%d'%(self.t[i],p1,p2,self.X[i-1]),end='\r')
                print('t=%.2f,FX=%.2f,FY=%.2f,s1=%.2f,s2=%.2f,X=%d,Y=%d,V=%.2f,FX2=%.2f'
                      % (self.i*self.dt,FX,FY,s1,s2,
                         self.X[j_prev],self.Y[j_prev],self.V[j_current],FX2))
                #print(p2,self.t[i],self.X[i-1],self.V[i-1])
            
            if self.posX[0] > 1000:
                raise ValueError("bounds exceeded")

            if False and self.i % 1000 == 0:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                
                v = self.V[j_prev]
                
                if v < 0:
                    xs = self.x[:self.A_idx]
                    px = self.posX[:self.A_idx]
                else:
                    xs = self.x[self.A_idx:]
                    px = self.posX[self.A_idx:]
                    
                ax.plot(xs,px)
                ax.plot(xs,libp.phi(xs,self.V[j_prev],self))
                
                print(self.V[j_prev],self.alpha,self.beta,self.A,self.B)
                
                #ax.plot(self.x,self.posY)
                plt.show(block=True)
                
                Xs_pde = np.sum(self.x*self.posX/np.sum(self.posX))
                Ys_pde = np.sum(self.x*self.posY/np.sum(self.posY))
                
                Xs_dis = libp.inverse_transform_sampling(self,self.posX,10)
                Ys_dis = libp.inverse_transform_sampling(self,self.posY,10)
                
                FX = np.sum(self.p(Xs_dis))
                FY = np.sum(self.p(Ys_dis))
                
            
                
                #print('posXmean distrib=%.4f,posYmean distrib=%.4f'
                #      'posXmean pde=%.4f,posYmean pde=%.4f'
                #      % (np.mean(Xs_dis),np.mean(Ys_dis),Xs_pde,Ys_pde))
                
                #print('force x distrib=%.4f,force y mean=%.4f,'
                #      'force x sim=%.4f,force y sim=%.4f X=%d,Y=%d'
                #      % (FX,FY,self.FX[j_current],self.FY[j_current],
                #         self.X[j_prev],self.Y[j_prev]))
                
                #print(Xs)
                
                time.sleep(3)
                plt.close()
                
            if (ry < s1):
                self.Y[j_current] = self.Y[j_prev] + 1
            elif (ry >= s1) and (ry < s1+s2):
                self.Y[j_current] = self.Y[j_prev] - 1
            else:
                self.Y[j_current] = self.Y[j_prev]

            
            if (side == 0) and (self.V[j_prev] >= self.switch_v):
                side = 1
                #print(self.V[j_prev])
                self.switch_times.append(self.i*self.dt)

            if (side == 1) and (self.V[j_prev] <= -self.switch_v):
                side = 0
                #print(self.V[j_prev])
                self.switch_times.append(self.i*self.dt)

            self.i += 1
            
        # end while loop
        self.mfpt = np.mean(np.diff(self.switch_times))

        print('MPFT=',self.mfpt)
        print('Total number of switches:',len(self.switch_times))



def main():

    """
    parser = argparse.ArgumentParser(description='run the agent-based Myosin motor model',
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
                        help=('If true, store all trace data to memory (for plotting)'))
    parser.add_argument('--no-storage',dest='storage',action='store_false')
    parser.set_defaults(storage=True)

    parser.add_argument('--ext',dest='ext',action='store_true',
                        help=('If true, use force extension in preferred direction. '
                              'else do not udate position in prefeerred direction'))
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
    options = {'X0':10,'Y0':10,
               'zeta':.5,'A':5,'B':5.1,
               'alpha':10,'beta':200,
               'T':50,'A0':0,
               'switch_v':54,
               'dt_factor':1e-2,
               'nX':100,'nY':100,
               'seed':0,
               'use_storage':True,
               'ivp_method':'euler',
               'U':None,
               'N':500}
    
    
    
    #kwargs = {**d_flags,**options}
    kwargs = options
    
    #a = Master(X0=0,Y0=0,seed=args.seed,nX=args.nX,nY=args.nY,T=args.Tfinal,
    #           dt_factor=args.dt,ze=2,B=5.1,al=25,be=100,switch_v=50.4,ext=args.ext,
    #           use_storage=args.storage)
    
    
    a = Master(**kwargs)

    t0 = time.time()
    a.run_states()
    t1 = time.time()
    print('*\t Run time',t1-t0)
    
    skipn = 1
    if False:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        ax1.plot(a.force_at_numberX[a.force_at_numberX > 0])
        ax2.plot(a.force_at_numberY[a.force_at_numberY > 0])
        
        print('mean force at number X',np.nanmean(a.force_at_numberX[a.force_at_numberX > 0]))
        print('mean force at number Y',np.nanmean(a.force_at_numberY[a.force_at_numberY > 0]))
    
    if False:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        #ax3 = fig.add_subplot(133)
        
        ax1.plot(a.t[::skipn],a.X[::skipn],alpha=.4)
        ax1.plot(a.t[::skipn],a.Y[::skipn],alpha=.4)
        
        ax2.plot(a.t[::skipn],a.meanPosX[::skipn],alpha=.4)
        ax2.plot(a.t[::skipn],a.meanPosY[::skipn],alpha=.4)
        
        #ax3.plot(a.t[::skipn],a.V[::skipn])
        
        #ax1.scatter(a.switch_times,np.zeros(len(a.switch_times)),color='tab:red',s=10)
        #ax2.scatter(a.switch_times,np.zeros(len(a.switch_times)),color='tab:red',s=10)
        #ax3.scatter(a.switch_times,np.zeros(len(a.switch_times)),color='tab:red',s=10)
    
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
        
        #ax3.plot(a.t[::skipn],a.V[::skipn])
        
        #ax1.scatter(a.switch_times,np.zeros(len(a.switch_times)),color='tab:red',s=10)
        #ax2.scatter(a.switch_times,np.zeros(len(a.switch_times)),color='tab:red',s=10)
        #ax3.scatter(a.switch_times,np.zeros(len(a.switch_times)),color='tab:red',s=10)
        
        
    #plt.show(block=True)
    #plt.close()
    
    # will only plot with --use-storage is enabled (should be enabled by default)
    #libm.plot_traces(a)
    #libm.plot_heatmap(a)
    
    plt.show(block=True)


if __name__ == "__main__":
    main()

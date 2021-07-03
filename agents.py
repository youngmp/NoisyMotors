# leaving note: dec 16, 2019 10:44pm: code works as expected when using analytic velocity, at least in the extreme case when no switching is expected (small zeta and stable velocity is far from origin according to mean field). euler velocity estimation results in extremely fast switching which is wrong... need a way to check this velocity calculation...

import argparse

import os
import time,datetime
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import uniform
from math import log10, floor
from lib.lubrication import lubrication as LB
from Master import Master as M

import matplotlib as mpl
#mpl.rcParams['text.usetex'] = False
#mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

pi=np.pi
exp=np.exp
sqrt=np.sqrt

import scipy.stats as st


# define probability distributions for the motor heads
class uPref(st.rv_continuous):

    def _pdf(self,x,A,B,alpha,beta,V0):
        """
        incoming V0 should be positive (it is made negative below)
        """
        return -exp((A-x)*beta/(-V0))*alpha*beta/((-V0)*(alpha+beta))


class uNonPref(st.rv_continuous):

    def _pdf(self,x,A,B,alpha,beta,V0):
        c = 1-exp(-beta*(B-A)/V0)
        return exp(-beta*(x-A)/V0)*alpha*beta/(V0*(alpha*c+beta))


class Agents(object):

    def __init__(self,**kwargs):
        """
        species 1 is down preferred.
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
                    'A':5,
                    'B':5.5,
                    'T':5,
                    'V0':0,  # nm/s shortening velocity for t > Tstart
                    'Z0':0,
                    'seed':0,
                    'ext':True,
                    'force_pos_type':'linear',
                    'U':None,
                    'store_position':False,
                    'fn_test_option':'root_cos'}
        
        # define defaults if not defined in kwargs
        for (prop, default) in defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        # need to multiply by factor of 2 to account for double forces.
        #self.zeta *= 2


        self.a1 = np.zeros(self.nX, dtype=bool)  # binary variable for motors
        self.x1 = np.zeros(self.nX)  # position for each motor

        self.a2 = np.zeros(self.nY, dtype=bool)
        self.x2 = np.zeros(self.nY)

        # initial condition

        self.a1[:self.X0] = 1
        self.a2[:self.Y0] = 1

        self.x1[:self.X0] = self.A
        self.x2[:self.Y0] = self.A

        self.TN = int(self.T / self.dt) + 1

        # total attached
        self.U1 = np.zeros(self.TN)
        self.U2 = np.zeros(self.TN)

        # force
        self.F1 = np.zeros(self.TN)
        self.F2 = np.zeros(self.TN)

        # position and velocity
        self.Z = np.zeros(self.TN)
        self.V = np.zeros(self.TN)

        # force at specific motor number
        self.force_at_numberX = np.zeros(self.TN)
        self.force_at_numberY = np.zeros(self.TN)

        if self.store_position:
            TN = self.TN
            self.X1 = np.zeros((TN, self.nX))
            self.X2 = np.zeros((TN, self.nY))
            
            self.dx1 = np.zeros(TN)
            self.dx2 = np.zeros(TN)

            self.motor_bool_x1 = np.zeros((TN, self.nX))
            self.motor_bool_x2 = np.zeros((TN, self.nX))

            self.t = np.linspace(0, self.T, self.TN)


        else:
            TN = 1
            self.X1 = np.zeros((TN, 2))
            self.X2 = np.zeros((TN, 2))
            self.dx1 = np.zeros(1)
            self.dx2 = np.zeros(1)

            self.motor_bool_x1 = np.zeros((1, self.nX),dtype=bool)
            self.motor_bool_x2 = np.zeros((1, self.nX),dtype=bool)

            self.t = np.linspace(0,self.T,)


        self.motor_bool_x1[0,:] = self.a1
        self.motor_bool_x2[0,:] = self.a2

        self.V[0] = self.V0
        self.Z[0] = self.Z0

        self.switch_times = []

    def p(self, x):
        """
        p1 = 4 # pN
        gamma = 0.322 # /nm
        """

        #return x#/self.gamma
        if (self.force_pos_type == 'linear') or (self.force_pos_type == 'lin'):
            return x*self.p1*self.gamma
        
        elif self.force_pos_type == 'exp':
            pass
            #return self.p1*(exp(self.gamma*x)-1)
            
        else:
            raise Exception('I dont recognize '
                            'this type of force-position curve',
                            self.force_pos_type)
        

    def run_agents(self):
        #print('test')
        print('force pos type',self.force_pos_type)
        
        np.random.seed(self.seed)
        
        side = 0

        k = 1
        while k*self.dt < self.T:

            t = k*self.dt
            
            # update according to new V
            if self.store_position:
                jm = k-1  # if use storage, save last entry
                j = k
            else:
                jm = 0
                j = jm

            self.Z[j] = self.Z[jm] + self.dt*self.V[jm]
            
            # if no extension, only extend in non-preferred direction.
            if self.V[jm] >= 0:
                self.dx1[jm] = self.V[jm]*self.dt
                self.dx2[jm] = self.ext*(-self.V[jm]*self.dt)
            else:
                self.dx1[jm] = self.ext*(self.V[jm]*self.dt)
                self.dx2[jm] = -self.V[jm]*self.dt

            # displace only attached crossbridges
            self.x1[self.a1 > 0] += self.dx1[jm]
            self.x2[self.a2 > 0] += self.dx2[jm]
            
            #print(self.x1[0])

            if self.store_position:
                self.X1[j,:] = self.x1
                self.X2[j,:] = self.x2
                #print(t,self.x1)
            
            # probability of changing state
            """
            if self.V[jm] < 0:
                attatch_p1 = (self.alpha*self.dt)
                attatch_p2 = 0
                
            elif self.V[jm] > 0:
                attatch_p1 = 0
                attatch_p2 = (self.alpha*self.dt)

            else:
                attach_p1 = (self.alpha*self.dt)
                attach_p2 = (self.alpha*self.dt)
            """
            
            attach_p1 = (self.alpha*self.dt)
            attach_p2 = attach_p1
            
            detatch_p = (self.beta*self.dt)
            
            prob1 = detatch_p*self.a1 + attach_p1*(1-self.a1)
            prob2 = detatch_p*self.a2 + attach_p2*(1-self.a2)

            # decide which crossbridges change state
            change1 = uniform(size=(self.nX,)) < prob1
            change2 = uniform(size=(self.nY,)) < prob2
            
            # detach crossbridges that pass x=B
            change1 = np.logical_or(change1,self.x1 > self.B)
            change2 = np.logical_or(change2,self.x2 > self.B)

            # detach crossbridges that pass x=B
            #change1 = np.logical_or(change1,self.x1<0)
            #change2 = np.logical_or(change2,self.x2<0)

            # change the state of those crossbridges
            self.a1 = self.a1 != change1
            self.a2 = self.a2 != change2

            self.motor_bool_x1[j,:] = self.a1
            self.motor_bool_x2[j,:] = self.a2

            # for newly attached bridges, set x=A
            self.x1[self.a1*change1] = self.A
            self.x2[self.a2*change2] = self.A

            # for all detached bridges, set x=0
            self.x1[np.logical_not(self.a1)] = 0 
            self.x2[np.logical_not(self.a2)] = 0

            # total motor number at time step j
            self.U1[j] = np.sum(self.a1)
            self.U2[j] = np.sum(self.a2)

            # net force 1
            self.F1[j] = np.sum(self.p(self.x1))
            self.F2[j] = np.sum(self.p(self.x2))
            
            if self.U1[j] == 11:
                self.force_at_numberX[j] = self.F1[j]
                
            if self.U2[j] == 11:
                self.force_at_numberY[j] = self.F2[j]


            if self.U is None:
                # factor of 2 is to correct for phi terms in mean field
                self.V[j] = (-self.F1[jm]+self.F2[jm])/(self.zeta)

            else:
                #print(t,U)
                if callable(self.U):
                    self.V[j] = self.U(t,option=self.fn_test_option)
                elif isinstance(self.U,float) or isinstance(self.U,int):
                    self.V[j] = self.U

            if True and (int(t/self.dt) % int((self.T/self.dt)/10) == 0):
                print('t=%.2f,V=%.2f, X=%d,Y=%d'
                      % (t,self.V[j],self.U1[j],self.U2[j]))
                

            if (side == 0) and (self.V[j] >= self.switch_v):
                side = 1
                self.switch_times.append(t)

            if (side == 1) and (self.V[j] <= -self.switch_v):
                side = 0
                self.switch_times.append(t)

            k += 1
            
        # save mfpt
        self.mfpt = np.mean(np.diff(self.switch_times))
        print('MPFT ignoring 0=',self.mfpt)


def round_to_n(x,n=3):
    return round(x, -int(floor(log10(x))) + (n - 1))


def main():
    #LUB = LB()
    # viscous drag

    
    parser = argparse.ArgumentParser(description='run the agent-based Myosin motor model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-s','--seed',default=0,type=int,
                        help='Set random number seed for simulation')
    parser.add_argument('-z','--zeta',default=0,type=np.float64,
                        help='Set viscous drag')
    parser.add_argument('-c','--convergence',action='store_true',
                        help='Set true to run convergence test')

    parser.add_argument('-T','--Tfinal',default=10,type=np.float64,help='Set total simulation time')
    parser.add_argument('-d','--dt',default=0.001,type=np.float64,help='Set time step factor')
    
    parser.add_argument('--save_switch',dest='switch',action='store_true',
                        help='If true, save switching rates')
    parser.add_argument('--no-save_switch',dest='switch',action='store_false',
                        help='If true, save switching rates')
    parser.set_defaults(switch=False)
    
    parser.add_argument('--no-storage',dest='storage',action='store_false',
                        help='If true, store all trace data to memory (for plotting)')
    parser.set_defaults(storage=True)

    parser.add_argument('--ext',dest='ext',action='store_true',
                        help='If true, store all trace data to memory (for plotting)')
    parser.add_argument('--no-ext',dest='ext',action='store_false',
                        help='If true, store all trace data to memory (for plotting)')
    parser.set_defaults(storage=True)

    parser.add_argument('-p','--eps',default=.01,type=np.float64,
                        help='Set epsilon for timescale separation')
    parser.add_argument('-X','--nX',default=100,type=int,
                        help='Set total motor number (X left preferred)')
    parser.add_argument('-Y','--nY',default=100,type=int,
                        help='Set total motor number (Y right preferrred)')

    parser.add_argument('-l','--use_last',default=False,action='store_true',
                        help=('If true, use last known saved data'
                              'and save final point of current data'))


    args = parser.parse_args()
    print('args',args)
    #print(args)

    a = Agents(store_position=True,Tf=args.Tfinal,V0=0,nX=args.nX,nY=args.nY,eps=args.eps,
               dt_factor=args.dt,seed=args.seed,zeta=args.zeta,
               force_pos_type='lin',B=5.1,alpha=14,beta=126,p1=4,ext=args.ext,switch_v=64)

    
    start_time = time.time()
    a.run_agents()
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()

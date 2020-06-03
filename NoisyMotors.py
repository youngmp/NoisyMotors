"""
TODO:
    -add ground truth steady-state distribution in phi
    -determine correct boudnary condition

Trying to apply upwind/downwind to our problem.

The equation I derived is ...see below
"""

#import time
import matplotlib

#import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import solve_ivp

import libNoisyMotors as lib

matplotlib.use('TkAgg')


class NoisyMotors(object):
    
    def __init__(self,**kwargs):

        defaults = {'N':100,
                    'dt':.0005,
                    'U':None,
                    'alpha':14,
                    'beta':126,
                    'A0':-1,
                    'A':5,
                    'B':5.5,
                    'B0':10,
                    'T':10,
                    'use_storage':True,
                    'ivp_method':'RK23',
                    'source':True,
                    'testing_ss':False,
                    'init_pars':None,
                    'domain':None}

        # define defaults if not defined in kwargs
        for (prop, default) in defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        
        assert(self.A < self.B)
        assert(self.A0 <= self.A)
        assert(self.U is not None)
        
        self.dx = (self.B-self.A0)/self.N
        
        self.x = np.linspace(self.A0,self.B,self.N)
        
        # index of position A
        self.A_idx = np.argmin(np.abs(self.x-self.A))
        
        # index of position B
        self.B_idx = np.argmin(np.abs(self.x-self.B))
        
        #self.CFL = np.abs(self.U)*self.dt/self.dx
        #rint("CFL=",np.abs(self.U)*self.dt/self.dx)
        
        self.TN = int(self.T/self.dt)  # time discretization
        self.t = np.linspace(0,self.T,self.TN)
        
        self.idx_full = np.arange(self.N)
    
    def boundary_left(self,U,sol):
        """
        left boundary condition changes depending on sign of U
        
        U < 0: Dirichlet or do nothing
        U > 0: Dirichlet +self.alpha*(1-self.theta)/U
        """
        
        if U < 0:
            return 0
        
        elif U > 0:
            return self.alpha*(1-self.theta_n)/U
    
    def boundary_right(self,U,sol):
        """
        Right boundary condition changes depending on sign of U
        
        U < 0: -self.alpha*(1-self.theta_n)/U
        U > 0: phi_x = phi_t
        """
        
        if U < 0:
            return -self.alpha*(1-self.theta_n)/U
        
        elif U > 0:
            return sol

    def run(self):
        """
        decides on which integration scheme to use based on user option (self.ivp_method)
        """

        if not(self.use_storage) and self.ivp_method == 'euler':
            self.t = np.array([0])
            self.sol = np.zeros((1,self.N))
        else:
            self.t = np.linspace(0,self.T,self.TN)
            self.sol = np.zeros((self.TN,self.N))
        
        # initial condition
        if self.init_pars is None:
            self.init = np.zeros_like(self.x)

        elif self.init_pars['type'] == 'gaussian':
            #self.init = lib.gauss(self.x-(self.B+self.A)/2,sig=1)
            self.init = lib.gauss(self.x-(self.A0+self.B)/2,sig=self.init_pars['pars'])

        self.sol[0,:] = self.init
        
        # indices of all points except appropriate boundary
        self.idx_except_last = self.idx_full[:-1]  # [0,1,2,3,4,5,6,7] to [0,1,2,3,4,5,6]
        self.idx_except_first = self.idx_full[1:]  # [0,1,2,3,4,5,6,7] to [1,2,3,4,5,6,7]
        self.idx_A2B = self.idx_full[self.A_idx:self.B_idx]
        
        self.roll_next = np.roll(self.idx_full,-1)[:-1]  # [0,1,2,3,4,5,6,7] to [1,2,3,4,5,6,7]
        self.roll_prev = np.roll(self.idx_full,1)[1:]  # [0,1,2,3,4,5,6,7] to [0,1,2,3,4,5,6]
        
        if self.ivp_method == 'euler':
            #assert (self.CFL < 1), "CFL condition not met for Euler method"
            
            i = 0
            while i < (self.TN-1):   
                #print(i)
                if self.use_storage:
                    k_next = i+1
                    k_current = i
                else:
                    k_next = 0
                    k_current = 0
                
                #print(k,self.use_storage)
                self.sol[k_next,:] = self.sol[k_current,:]+self.dt*(self.upwind(self.t[k_current],
                                                                    self.sol[k_current,:],
                                                                    self.U))

                i += 1
        
        else:
            obj_integrated = solve_ivp(self.upwind,[0,self.T],self.init,args=(self.U,),
                                       t_eval=self.t,
                                       method=self.ivp_method)
            self.sol = obj_integrated.y.T
        
    def upwind(self,t,sol,U):
        """
        Implementation of upwinding scheme to be used in Euler loop
        method of lines
        """
        #sol[0] = 0
        
        out = np.zeros_like(sol)
        
        if callable(U):
            Uval = U(t)
        else:
            Uval = U
        
        if Uval > 0:
            # boundaries
            idx_update = self.idx_except_first
            out[0] = -self.beta*sol[0]-sol[0]*Uval/self.dx
            
            
        else:
            # boundaries
            idx_update = self.idx_except_last
            out[-1] = -self.beta*sol[-1]+sol[-1]*Uval/self.dx
        
        U_minus = np.amin([Uval,0])
        U_plus = np.amax([Uval,0])
        
        p_plus = (sol[self.roll_next]-sol[self.idx_except_last])/self.dx
        p_minus = (sol[self.idx_except_first]-sol[self.roll_prev])/self.dx
        
        
        # update derivatives
        out[idx_update] = -self.beta*(sol[idx_update]) - p_plus*U_minus - p_minus*U_plus
        #out[-1] = - p_plus[-1]*U_minus - p_minus[-1]*U_plus
        
        
        self.theta_n = np.sum(sol)*self.dx
        #assert((self.theta_n <= 1) and (self.theta_n >= 0))
        
        # update input
        if self.source:
            out[self.A_idx] += self.alpha*(1-self.theta_n)/self.dx
        
        #out[-1] += self.alpha-self.theta_n*(self.alpha+self.beta)/self.dx
        
        return out
    
    
def main():
    
    pass
    

if __name__ == "__main__":
    main()

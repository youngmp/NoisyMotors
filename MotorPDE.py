"""
TODO:
    -add ground truth steady-state distribution in phi
    -determine correct boudnary condition

Trying to apply upwind/downwind to our problem.

The equation I derived is ...see below
"""

import time
#import matplotlib

#import matplotlib.pyplot as plt
import numpy as np


from scipy.integrate import solve_ivp
#from scipy.interpolate import interp1d
#from cumsumb import cumsum
import lib.libMotorPDE as lib

#matplotlib.use('TkAgg')


class MotorPDE(object):
    
    def __init__(self,**kwargs):

        defaults = {'N':100,
                    'N2':100,
                    'dt':.0005,
                    'U':None,
                    'alpha':14,
                    'beta':126,
                    'zeta':1,
                    'A0':0,
                    'A':5,
                    'B':5.1,
                    'T':10,
                    'fn_test_option':'root_cos',
                    'fn_vel':50,
                    'store_position':False,
                    'ivp_method':'euler',
                    'source':True,
                    'regularized':False,
                    'testing_ss':False,
                    'init_pars':None,
                    'domain':None,
                    'irregular':False}

        
        # define defaults if not defined in kwargs
        for (prop, default) in defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        #print('source type',self.source)
        assert(self.A <= self.B)
        assert(self.A0 <= self.A)
        #assert(self.U is not None)
        
        #self.dx = (self.B-self.A0)/self.N

        # center an index at z=A and build from there.

        if self.irregular:

            x_left = np.linspace(self.A0,self.A,self.N)
            x_right = np.linspace(self.A,self.B,self.N2)

            #print('xleft,xright',x_left,x_right)

            self.x = np.append(x_left,x_right[1:])
            self.dx = np.diff(self.x)
            #self.dx = np.append(self.dx[-1],self.dx[0])
            #self.dx = np.append(self.dx,self.dx[-1])

            # note that A_idx is chosen just right of z=A
            # this is because the mesh is finer on the right and
            # easier to manage.

            self.A_idx = len(x_left)
            self.B_idx = len(self.x)-1
            print(self.dx[self.A_idx-2])
            print(self.dx[self.A_idx-1])
            print(self.dx[self.A_idx])
            print(self.dx[self.A_idx+1])
            print(self.dx[self.A_idx+2])

            #print(self.x[self.A_idx])

        else:
            self.x,self.dx = np.linspace(self.A0,self.B,self.N,
                                         endpoint=False,retstep=True)

            # index of A
            self.A_idx = np.argmin(np.abs(self.x-(self.A)))

            # index of position B
            self.B_idx = np.argmin(np.abs(self.x-self.B))


        self.idx_full = np.arange(len(self.x))
        
        # indices of all points except appropriate boundary
        # [0,1,2,3,4,5,6,7] to [0,1,2,3,4,5,6]
        self.idx_except_last = self.idx_full[:-1]
        
        # [0,1,2,3,4,5,6,7] to [1,2,3,4,5,6,7]
        self.idx_except_first = self.idx_full[1:]
        #self.idx_A2B = self.idx_full[self.A_idx:self.B_idx]
        
        # [0,1,2,3,4,5,6,7] to [1,2,3,4,5,6,7]
        self.roll_next = np.roll(self.idx_full,-1)[:-1]

        # [0,1,2,3,4,5,6,7] to [0,1,2,3,4,5,6]
        self.roll_prev = np.roll(self.idx_full,1)[1:]
        
        self.TN = int(self.T/self.dt) # time discretization

        # preallocate output array for upwinding scheme here for efficiency.
        self.out = np.zeros_like(self.x)
        
        if not self.store_position and self.ivp_method == 'euler':
            TN = 1
        else:
            TN = self.TN

        self.t = np.linspace(0,self.T,TN)
        self.sol = np.zeros((TN,len(self.x)))
        self.U_arr = np.zeros(TN)

        if self.regularized:
            # regularized delta function
            s = (self.A-self.A0)/self.dx
            i = np.floor(s) # floor index of A
            r = s-i # distance from floor in index
            
            
            self.delta_idxs = np.mod(np.arange(i-2,i+1+1,1),self.N)+1
            self.delta_idxs = self.delta_idxs.astype(int)
            q = np.sqrt(1+4*r*(1-r))
            self.ws = np.array([(3-2*r-q)/8,
                                (3-2*r+q)/8,
                                (1+2*r+q)/8,
                                (1+2*r-q)/8])

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
        decides on which integration scheme to use 
        based on user option (self.ivp_method)
        """

        
        # initial condition
        if self.init_pars is None:
            self.init = np.zeros_like(self.x)

        elif self.init_pars['type'] == 'gaussian':
            self.init = lib.gauss(self.x-(self.A0+self.B)/2,
                                  sig=self.init_pars['pars'])

        self.sol[0,:] = self.init
        
        if self.ivp_method == 'euler':
            self.sol = self.run_euler()
        
        else:
            obj_integrated = solve_ivp(self.upwind,[0,self.T],
                                       self.init,args=(self.U,),
                                       t_eval=self.t,
                                       method=self.ivp_method)
            self.sol = obj_integrated.y.T
    
    def run_euler(self):
        #assert (self.CFL < 1), "CFL condition not met for Euler method"
        
        self.i = 0
        
        while self.i < (self.TN-1):

            
            if self.U == 'dynamic':
                # generate population distribution from sol
                # draw from population distribution
                if np.add.reduce(self.sol[self.i,:]) != 0:
                    xs = lib.inverse_transform_sampling(self,
                                                        self.sol[self.i,:],100)
                else:
                    xs = np.zeros(100)
                
                # get total force
                f_up = np.add.reduce(lib.force_position(xs))*self.dx
                #print(f_up)
                f_down = 0
                
                Uval = (-f_up + f_down)/(self.zeta)
                
                # update velocity
                #Uval = self.update_velocity(f_up,f_down,Uval)
                
                #print(Uval)
            else:
                Uval = self.U
            
            k_next, k_now = lib.get_time_index(self.store_position,self.i)
            
            sol_now = self.sol[k_now,:]
            
            
            self.sol[k_next,:] = sol_now + self.dt*(self.upwind(self.t[k_now],
                                                                sol_now,
                                                                Uval))

            self.i += 1
            
        return self.sol

    def upwind(self,t,sol,U):
        """
        Implementation of upwinding scheme to be used in Euler loop
        method of lines
        """

        if False:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            mpl.rcParams['text.usetex'] = False
            mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
            
            fig = plt.figure(figsize=(4, 5))
            ax1 = fig.add_subplot(111)
            ax1.set_title('input sol to upwind')
            
            ax1.plot(sol)
            
            plt.show()
            plt.close()
            time.sleep(2)

        if callable(U):
            Uval = U(t,vel=self.fn_vel,option=self.fn_test_option)
        elif isinstance(U,float) or isinstance(U,int):
            Uval = U
            
        if self.ivp_method == 'euler' and self.store_position:
            self.U_arr[self.i] = Uval
            
        if Uval > 0:
            # boundaries
            idx_update = self.idx_except_first
            if self.irregular:
                dx = self.dx[0]
            else:
                dx = self.dx
            self.out[0] = -self.beta*sol[0]-sol[0]*Uval/dx

            
            
        else:
            # boundaries
            idx_update = self.idx_except_last
            if self.irregular:
                dx = self.dx[-1]
            else:
                dx = self.dx
            self.out[-1] = -self.beta*sol[-1]+sol[-1]*Uval/dx

        
        
        if Uval <= 0:
            U_minus = Uval
            U_plus = 0
        else:
            U_minus = 0
            U_plus = Uval

        if self.irregular:
            dx = self.dx
        else:
            dx = self.dx

        p_plus = (sol[self.roll_next]-sol[self.idx_except_last])/dx
        p_minus = (sol[self.idx_except_first]-sol[self.roll_prev])/dx

        #if Uval > 0:
        #    print('p_plus',p_plus[-5:])
        #print('p_plus',p_plus[-5:])
        
        # update derivatives
        wind = p_plus*U_minus + p_minus*U_plus
        #print(self.i,'plus,minus',wind,U_minus,U_plus)
        self.out[idx_update] = -self.beta*(sol[idx_update]) - wind

        

        #print(dx,self.dx,self.irregular,self.alpha*(1-self.theta_n)/dx)
        #print()
        
        #print(self.out[self.A_idx])

        if self.irregular:
            self.theta_n = np.add.reduce(sol[:-1]*self.dx)
        else:
            self.theta_n = np.add.reduce(sol)*self.dx
        #assert((self.theta_n <= 1) and (self.theta_n >= 0))
        #print('thetan',self.theta_n)
        
        # update input
        if (self.source == True or self.source == 'motor')\
            and self.regularized == False:
            
            if self.irregular:
                dx = self.dx[self.A_idx]
                
            else:
                dx = self.dx
                
            #print(dx,self.dx,self.irregular,self.alpha*(1-self.theta_n)/dx)
            #print()
            #print(self.out[self.A_idx])
            self.out[self.A_idx] += self.alpha*(1-self.theta_n)/dx
            #print('out@A_idx',self.out[self.A_idx])
            #print(self.out[self.A_idx],dx,self.alpha,(1-self.theta_n))

        elif (self.source == True or self.source == 'motor')\
            and self.regularized == True:
            if self.irregular:
                dx = self.dx[self.delta_idxs]
            else:
                dx = self.dx

            self.out[self.delta_idxs] += self.ws*self.alpha*(1-self.theta_n)/dx

            # Gaussian source
            #sig = .3
            #k = self.alpha*(1-self.theta_n)/(sig*np.sqrt(2*np.pi))
            #out[self.idx_full] += k*np.exp(-0.5*(self.x-self.A)**2/sig**2)

        elif callable(self.source) and self.regularized == True:
            if self.irregular:
                dx = self.dx[self.A_idx]
            else:
                dx = self.dx
            self.out[self.delta_idxs] += self.ws*self.source(t)/dx

            

        elif callable(self.source) and self.regularized == False:
            if self.irregular:
                dx = self.dx[self.A_idx]
            else:
                dx = self.dx
            self.out[self.A_idx] += self.source(t)/dx

            
        
        elif self.source == False:
            pass
        
        else:
            raise ValueError('Unrecognized source option',self.source,
                             self.regularized)
            
        #elif self.source == 'regularized_custom':
        #    #print(self.source(t))
        #    out[self.delta_idxs] += self.ws*self.source(t)/self.dx

        #if Uval > 0:
        #    print('out',self.out[-5:])


        return self.out
    
    
def main():
    
    pass
    

if __name__ == "__main__":
    main()

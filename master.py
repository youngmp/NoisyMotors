# testing ground for master equation


import argparse


import os
import time
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] 

pi = np.pi
exp = np.exp
sqrt = np.sqrt
Sqrt = np.sqrt


class Master(object):

    def __init__(self,**kwargs):

        defaults = {'al':14,
                    'be':126,
                    'nX':100,
                    'nY':100,
                    'ze':0.048,
                    'dt_factor':0.01,
                    'switch_v':100,
                    'gm':0.322,
                    'p1':4,
                    'X0':1,
                    'Y0':1,
                    'A':5,
                    'B':5.5,
                    'T':5,
                    'seed':0,
                    'ext':True,
                    'use_storage':True
                    }

        # define defaults if not defined in kwargs
        for (prop, default) in defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
        
        self.k = self.p1*self.gm  # spring constant
        
        self.dt = self.dt_factor/(self.al+self.be+self.nX+self.nY)
        self.TN = int(self.T/self.dt)  # total anticipated time steps
        
        
        if self.use_storage:
            self.t = np.linspace(0,self.T,self.TN)
        else:
            self.t = np.linspace(0,self.T,)
    
    def u(self,x,v):

        c = 1-exp(-self.be*(self.B-self.A)/v)
        u0 = (self.al*self.be)/(v*(self.al*c+self.be))

        return u0*exp(-self.be*(x-self.A)/v)

    def kp(self,i,n):
        """
        i: motor state
        n: total motor number
        """
        return self.al*(n-i)

    def si(self,i,v):
        """
        km(i) + gm(i)
        """
        
        if v <= 0:
            return self.be*i
        else:
            #return self.be*i + i*v*self.u(self.B,v)
            return self.be*i + i*v/self.B

    def gmHat(self,i):
        return i/(self.B-self.A)
        

    def vel(self,nx,ny):

        #nx = np.asarray(nx)
        #ny = np.asarray(ny)
        #print('ze in vel in master',self.ze)

        #return self.k*self.A*(ny-nx)/self.ze

        #return -(((self.A*nx - self.A*ny)*self.be)/(nx + ny + self.be*self.ze))

        # see agents_velocity.nb. uses saturating approximation to true force-velocity

        if not(self.ext):
            if nx>=ny:
                return -(4*self.A*self.k*nx-2*self.A*self.k*ny-2*self.B*self.k*ny+self.A*self.be*self.ze-self.B*self.be*self.ze+Sqrt(-16*self.A*(self.A-self.B)*self.k*(nx-ny)*self.be*self.ze+(-4*self.A*self.k*nx+2*self.A*self.k*ny+2*self.B*self.k*ny-self.A*self.be*self.ze+self.B*self.be*self.ze)**2))/(8.*self.ze)
            else:
                return (-2*self.A*self.k*nx-2*self.B*self.k*nx+4*self.A*self.k*ny+self.A*self.be*self.ze-self.B*self.be*self.ze+Sqrt(16*self.A*(self.A-self.B)*self.k*(nx-ny)*self.be*self.ze+(2*self.A*self.k*nx+2*self.B*self.k*nx-4*self.A*self.k*ny-self.A*self.be*self.ze+self.B*self.be*self.ze)**2))/(8.*self.ze)
                
        else:
            if nx>= ny:
                return (-5*self.A*self.k*nx*self.be + self.B*self.k*nx*self.be + 2*self.A*self.k*ny*self.be + 2*self.B*self.k*ny*self.be - self.A*self.be**2*self.ze + self.B*self.be**2*self.ze - sqrt(self.be**2*(-16*self.A*(self.A - self.B)*self.k*(nx - ny)*(self.k*nx + self.be*self.ze) +(self.k*(-5*self.A*nx + self.B*nx + 2*self.A*ny + 2*self.B*ny) + (-self.A + self.B)*self.be*self.ze)**2)))/(8.*(self.k*nx + self.be*self.ze))

            else:
                return (-(self.k*(2*self.A*nx + 2*self.B*nx - 5*self.A*ny + self.B*ny)*self.be) + (self.A - self.B)*self.be**2*self.ze +sqrt(self.be**2*(16*self.A*(self.A - self.B)*self.k*(nx - ny)*(self.k*ny + self.be*self.ze) +(self.k*(2*self.A*nx + 2*self.B*nx - 5*self.A*ny + self.B*ny) + (-self.A + self.B)*self.be*self.ze)**2)))/(8.*(self.k*ny + self.be*self.ze))


    def vel_array(self,nx,ny):
        """
        array version of the above velocity equaiton
        """
        
        A = self.A
        B = self.B
        be = self.be
        ze = self.ze
        k = self.k
        
        assert(len(nx) == len(ny))

        out_vel = np.zeros(len(nx))

        return -(((A*nx - A*ny)*be)/(nx + ny + be*ze))
        
        # if Y > X then velocity is positive
        pos_bool = ny > nx 
        neg_bool = nx >= ny


        out_vel[pos_bool] = (2*A*k*ny[pos_bool]*be - B*k*(nx[pos_bool] + ny[pos_bool])*be + (A - B)*be**2*ze +sqrt(self.be**2*(4*self.A*(self.A - self.B)*self.k*(nx[pos_bool] - ny[pos_bool])*(self.k*ny[pos_bool] + self.be*self.ze) +(self.B*self.k*(nx[pos_bool] + ny[pos_bool]) + self.B*self.be*self.ze - self.A*(2*self.k*ny[pos_bool] + self.be*self.ze))**2)))/(2.*(self.k*ny[pos_bool] + self.be*self.ze))

        out_vel[neg_bool] = (-2*A*k*nx[neg_bool]*be + B*k*nx[neg_bool]*be + B*k*ny[neg_bool]*be - A*be**2*ze + self.B*self.be**2*self.ze -sqrt(self.be**2*(-4*self.A*(self.A - self.B)*self.k*(nx[neg_bool] - ny[neg_bool])*(self.k*nx[neg_bool] + self.be*self.ze) +(self.A*(2*self.k*nx[neg_bool] + self.be*self.ze) - self.B*(self.k*(nx[neg_bool] + ny[neg_bool]) + self.be*self.ze))**2)))/(2.*(self.k*nx[neg_bool] + self.be*self.ze))
        
        return out_vel
        
    
    def run_states(self):

        np.random.seed(self.seed)

        if self.use_storage:
            size = self.TN
        else:
            size = 1

        self.X = np.zeros(size)
        self.Y = np.zeros(size)
        self.V = np.zeros(size)

        self.X[0] = self.X0
        self.Y[0] = self.Y0

        self.switch_times = []

        # keep track of which side you are on after switch
        side = 0 

        i = 1
        while i < self.TN:

            if self.use_storage:
                jm = i-1
                j = i
            else:
                jm = 0
                j = jm

            v = self.vel(self.X[jm],self.Y[jm])
            #v = 500

            self.V[jm] = v
            
            # draw random number
            rx = np.random.rand()
            ry = np.random.rand()

            # update X
            
            # decay rate
            p1 = self.al*(self.nX-self.X[jm])*self.dt
            #p2 = self.X[i-1]*(self.be+np.abs(v)*np.heaviside(v,0)/(self.B-self.A))*self.dt #

            if v == 0:
                rate_extra = 1
            else:
                rate_extra = 1/(1-exp(self.be*(self.A-self.B)/np.abs(v)))

            lam1 = np.heaviside(v,0)
            
            # decay rate
            p2 = self.X[jm]*(self.be*((1-lam1)+rate_extra*lam1))*self.dt

            if (rx < p1):
                self.X[j] = self.X[jm] + 1
            elif (rx >= p1) and (rx < p1+p2):
                self.X[j] = self.X[jm] - 1
            else:
                self.X[j] = self.X[jm]

            # update Y
            s1 = self.al*(self.nY-self.Y[jm])*self.dt
            
            # decay rate
            #s2 = self.Y[i-1]*(self.be+np.abs(v)*np.heaviside(-v,0)/(self.B-self.A))*self.dt 
            if v == 0:
                rate_extra = 1
            else:
                rate_extra = 1/(1-exp(self.be*(self.A-self.B)/np.abs(v)))

            lam2 = np.heaviside(-v,0)
            
            # decay rate
            s2 = self.Y[jm]*(self.be*((1-lam2) + rate_extra*lam2))*self.dt

            if i % 10000 == 0:
                #print('t=%.2f,p1=%.4f,p2=%.4f,X=%d'%(self.t[i],p1,p2,self.X[i-1]),end='\r')
                print('t=%.2f,p1=%.4f,p2=%.4f,s1=%.4f,s2=%.4f,X=%d,Y=%d,v=%.4f'%(i*self.dt,p1,p2,s1,s2,self.X[jm],self.Y[jm],self.V[jm]))
                #print(p2,self.t[i],self.X[i-1],self.V[i-1])

            
            if (ry < s1):
                self.Y[j] = self.Y[jm] + 1
            elif (ry >= s1) and (ry < s1+s2):
                self.Y[j] = self.Y[jm] - 1
            else:
                self.Y[j] = self.Y[jm]

            
            if (side == 0) and (self.V[jm] >= self.switch_v):
                side = 1
                self.switch_times.append(i*self.dt)

            if (side == 1) and (self.V[jm] <= -self.switch_v):
                side = 0
                self.switch_times.append(i*self.dt)

            i += 1
            
        # end while loop
        self.mfpt = np.mean(np.diff(self.switch_times))

        print('MPFT=',self.mfpt)
        print('Total number of switches:',len(self.switch_times))
        

def plot_traces(obj):


    if obj.use_storage:

        fig = plt.figure(figsize=(8,3))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.plot(obj.t,obj.X,alpha=.5,label='X')
        ax1.plot(obj.t,obj.Y,alpha=.5,label='Y')

        ax2.plot(obj.t,obj.V,alpha=.5)
        ax2.scatter(obj.switch_times,np.zeros(len(obj.switch_times)),color='tab:red',s=10)
        
        ax1.set_xlabel('t (s)')
        ax1.set_title(r'$n_X=%d,n_Y=%d$'%(obj.nX,obj.nY))
        #ax1.set_title('Linear position-velocity curve')
        ax2.set_xlabel('t (s)')
        
        ax1.set_ylabel('Motor Number')
        ax2.set_ylabel('Velocity')
        
        plt.suptitle('Master Equation Solution with Linear Position-Velocity. MFPT='+str(obj.mfpt))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plt.show()


def plot_heatmap(obj):

    if obj.use_storage:

        fig2 = plt.figure(figsize=(4,4))
        ax_fig2 = fig2.add_subplot(111)

        # extract only transition points
        hist_data = ax_fig2.hist2d(obj.X,obj.Y,bins=[np.arange(0,obj.nX+1.5)-0.5,np.arange(0,obj.nY+1.5)-0.5])

        ax_fig2.set_title('raw histogram')
        
        fig3 = plt.figure(figsize=(4,4))
        ax_fig3 = fig3.add_subplot(111)
        
        t = np.arange(len(obj.X))
        #ax_fig3.scatter(obj.X,obj.Y,alpha=.1,label='X',c=t,cmap='viridis')
        th = 200

        # set numbers above threshold to threshold
        #th_mat = hist_data[0]*(hist_data[0]>th)
        #th_mat
        th_mat = hist_data[0]
        th_mat[th_mat > th] = th
        
        #th_mat = hist_data[0]*(hist_data[0]<th) + 
        ax_fig3.imshow(th_mat[:,:])#,vmin=0,vmax=th)
        ax_fig3.set_title('show all with threshold='+str(th))
        
        fig4 = plt.figure(figsize=(4,4))
        ax_fig4 = fig4.add_subplot(111)
        
        ax_fig4.set_title('show only below threshold='+str(th))
        a = hist_data[0]*(hist_data[0]<th)
        ax_fig4.imshow(a[:,:])

        
        fig5 = plt.figure(figsize=(4,4))
        ax_fig5 = fig5.add_subplot(111)
        
        
        fig6 = plt.figure(figsize=(4,4))
        ax_fig6 = fig6.add_subplot(111)
        
        #hist = ax_fig6.hist(obj.X,bins=np.arange(0,obj.nX+1.5)-0.5,alpha=.5)
        #hist = ax_fig6.hist(obj.Y,bins=np.arange(0,obj.nY+1.5)-0.5,alpha=.5)
        
        ax_fig6.hist(obj.X,bins=np.arange(0,obj.nX+1.5)-0.5,alpha=.5)
        ax_fig6.hist(obj.Y,bins=np.arange(0,obj.nY+1.5)-0.5,alpha=.5)
        
        
        fig7 = plt.figure(figsize=(4,4))
        ax_fig7 = fig7.add_subplot(111)
        
        #hist = ax_fig7.hist(obj.V,100,alpha=.5)
        ax_fig7.hist(obj.V,100,alpha=.5)
        
        plt.show()


def main():


    parser = argparse.ArgumentParser(description='run the agent-based Myosin motor model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-s','--seed',default=0,type=int,
                        help='Set random number seed for simulation')
    #parser.add_argument('-z','--zeta',default=0,type=np.float64,help='Set viscous drag')

    parser.add_argument('-T','--Tfinal',default=10,type=np.float64,
                        help='Set total simulation time')
    parser.add_argument('-d','--dt',default=0.001,type=np.float64,
                        help='Set time step factor')

    """
    parser.add_argument('--save_switch',dest='switch',action='store_true',
                        help='If true, save switching rates')
    parser.add_argument('--no-save_switch',dest='switch',action='store_false',
                        help='If true, save switching rates')
    parser.set_defaults(switch=False)
    """
    
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
    
    # options not from terminal flags
    options = {'X0':0,'Y0':0,
               'ze':2,'B':5.1,
               'al':25,'be':100,
               'T':1,
               'switch_v':50.4}
    
    kwargs = {**d_flags,**options}
    
    #a = Master(X0=0,Y0=0,seed=args.seed,nX=args.nX,nY=args.nY,T=args.Tfinal,
    #           dt_factor=args.dt,ze=2,B=5.1,al=25,be=100,switch_v=50.4,ext=args.ext,
    #           use_storage=args.storage)
    
    a = Master(**kwargs)

    a.run_states()

    # will only plot with --use-storage is enabled (should be enabled by default)
    plot_traces(a)
    plot_heatmap(a)


if __name__ == "__main__":
    main()

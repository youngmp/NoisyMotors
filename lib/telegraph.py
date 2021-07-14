"""
File for telegraph process

in this file, run telegraph process until escape through either boundary.

record escape side, time to escape, and seed in CSV

file name should include initial velocity, lambda, initial position, side length

make serial for now, parallelize later if needed.

run as euler.  gillespie later if needed.

"""

import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

A = 0 # left boundary
B = 200 # right boundary
lam = 1/.27 # switching rate

def e0p(L,U,lam):
    """
    probability of escape through right given different domain lengths L
    and starting position 0.
    """
    return U/(U+L*lam)

def t0p(L,U,lam):
    return (3*L*U**3 + 3*L**2*U**2*lam + L**3*U*lam**2)/(3.*U**3*(U + L*lam))

def pp(x0,U,lam,L):
    """
    pp means (p)robability to escape right with (p)ositive initial vel.
    """
    return (U + x0*lam)/(U + L*lam)

def tp(x0,U,lam=2,L=3):
    """
    tp means (t)ime to escape right with (p)ositive initial vel.
    """
    return ((2*(L - x0))/U + ((L - x0)*(L + x0)*lam)/U**2
            + L/(U + L*lam) - x0/(U + x0*lam))/3.

def run_sim(dt=.000001,seed=0,X0=0,V0=-1,
            return_data = False):
    """
    d: dict of parameters
    return_data: return simulation trajectory.
    """
    
    
    in_domain = True
    
    i = 0
    x_prev = X0
    V = V0
    sol = []

    np.random.seed(seed)
    
    while in_domain:
        #print(x_prev,dt*V)
        
        # update position
        x_next = x_prev + dt*V
        sol.append(x_next)

        # change direction?
        r = np.random.rand()
        if r < dt*lam:
            V *= -1

        # check if boundary hit
        if x_next <= A:
            #print('hit A')
            in_domain = False
            hit = 'L'
            break

        if x_next >= B:
            #print('hit B')
            in_domain = False
            hit = 'R'
            break
        
        x_prev = x_next
        i += 1

    total_time = i*dt    

    if return_data: 
        return hit, total_time, seed, sol
    else:
        return hit, total_time, seed
    
def main():

    # for x position, loop over 100 seeds.
    x_list = np.arange(A,B,(B-A)/4)
    seeds = np.arange(100)

    time_left = []
    time_right = []
    l_list = []
    r_list = []

    for i,x0 in enumerate(x_list):

        time_left_temp = []
        time_right_temp = []

        l_counter = 0
        r_counter = 0

        for j in seeds:

            hit, pt, seed, sol = run_sim(
                X0=x0,seed=j,V0=121,return_data = True)
            
            if hit == 'R' and x0 == B and False:
                print(hit,pt,seed,sol)
                print(time_right_temp)
            
            if hit == 'L':
                time_left_temp.append(pt)
                l_counter += 1
            else:
                time_right_temp.append(pt)
                r_counter += 1

            
            
            
            
        print(l_counter,r_counter)
        l_list.append(l_counter)
        r_list.append(r_counter)
        
        time_left.append(np.mean(time_left_temp))
        time_right.append(np.mean(time_right_temp))

    fig = plt.figure(figsize=(8,3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(x_list,np.array(r_list)/(np.array(l_list)+np.array(r_list)),
             label='simulation')
    ax1.plot(x_list,pp(x_list,121,lam=lam,L=B-A),label='theory')
    
    #ax.plot(x_list,time_left)
    ax2.plot(x_list,time_right,label='simulation')
    ax2.plot(x_list,tp(x_list,121,lam=lam,L=B-A),label='theory')

    


    
    ax2.legend()
    ax2.set_title('MFPT to exit right given initial positive velocity')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('MFPT')

    
    plt.show()
    
    

    
if __name__ == "__main__":
    main()

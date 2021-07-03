"""
script to run agent-based model for long times.
"""


import argparse

import numpy as np
import parsets as pset

from agents import Agents

def long_agents(seed,recompute=True):
    """
    simulate agent-based model for a long time with a fine time step
    for a given seed
    """

    # agent simulation
    kwargs = {'T':3,
              'V0':0,
              'Z0':0,
              'dt':5e-7,
              'seed':seed,
              'U':None,
              'force_pos_type':'lin',
              'ext':True}

    kwargs = dict(kwargs,**pset.parset('figure1'))
    kwargs['store_position'] = True

    a = Agents(**kwargs)
    a.run_agents()

    switch_agents = a.switch_times
    t_agents = a.t
    V = a.V
    Z = a.Z

    np.savetxt('long_switch_agents'+str(seed)+'.txt',a.switch_times)
    np.savetxt('long_t'+str(seed)+'.txt',a.t)
    np.savetxt('long_V'+str(seed)+'.txt',a.V)
    np.savetxt('long_Z'+str(seed)+'.txt',a.Z)


def main():

        
    parser = argparse.ArgumentParser(description='run the agent-based Myosin motor model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-s','--seed',default=0,type=int,
                        help='Set random number seed for simulation')

    args = parser.parse_args()

    long_agents(args.seed)
    

if __name__ == "__main__":
    main()

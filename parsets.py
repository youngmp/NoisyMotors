

def parset(num):
    """

    Parameters
    ----------
    num

    Returns
    -------
    dict of parameters

    """

    if num == 3:
        return {'nX': 100, 'nY': 100,
                'zeta': 0.7,
                'A': 5, 'B': 5.1,
                'alpha': 14, 'beta': 126,
                'p1': 4, 'gamma': 0.322,
                'switch_v': 64,
                }

    elif num == '3a':
        return {'nX':1000,'nY':1000,
                'zeta':1.4,
                'A':5,'B':5.1,
                'alpha':14,'beta':126,
                'p1':4/10,'gamma':0.322,
                'switch_v': 24,
                }

    elif num == '3b':
        return {'nX':100,'nY':100,
                'zeta':0.01,
                'A':5,'B':6,
                'alpha':14,'beta':126,
                'p1':4/20,'gamma':0.322,
                'switch_v':384,'X0':0,'Y0':0
                }

    if num == 13:
        return {'nX':400,'nY':400,
                'zeta':0.7,
                'A':5,'B':5.1,
                'alpha':14,'beta':126,
                'p1':4/4,'gamma':0.322,
                'switch_v':164,
                }

    elif num == 14:
        return {'nX':200,'nY':200,
                'zeta':1.4,
                'A':5,'B':5.1,
                'alpha':25,'beta':100,
                'p1':4/2,'gamma':0.322,
                'switch_v':121,
                }

    elif num == 15:
        return {'nX':400,'nY':400,
                'zeta':1.4,
                'A':5,'B':5.1,
                'alpha':50,'beta':10,
                'p1':4/2,'gamma':0.322,
                'switch_v':42,
                }

    elif num == 16:
        return {'nX':400,'nY':400,
                'zeta':1.4,
                'A':5,'B':5.1,
                'alpha':50,'beta':200,
                'p1':4/2,'gamma':0.322,
                'switch_v':42,
                }



    else:
        raise ValueError('Invalid option', num)
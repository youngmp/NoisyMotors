

def parset(num):
    """

    Parameters
    ----------
    num

    Returns
    -------
    dict of parameters

    """

    if num == '1a':
        return {'nX':100,'nY':100,
                'zeta':.2,
                'A':5,'B':5.05,
                'alpha':5,'beta':200,
                'p1':4/1,'gamma':0.322,
                'switch_v':61,'X0':0,'Y0':0
                }


    if num == '1b':
        return {'nX':500,'nY':500,
                'zeta':.2,
                'A':5,'B':5.05,
                'alpha':5,'beta':200,
                'p1':4/5,'gamma':0.322,
                'switch_v':61,'X0':0,'Y0':0
                }


    if num == '1c':
        return {'nX':1000,'nY':1000,
                'zeta':.2,
                'A':5,'B':5.05,
                'alpha':5,'beta':200,
                'p1':4/10,'gamma':0.322,
                'switch_v':61,'X0':0,'Y0':0
                }
    
    
    if num == '2':
        return {'nX':100,'nY':100,
                'zeta':.4,
                'A':5,'B':5.6,
                'alpha':14,'beta':126,
                'p1':4/1,'gamma':0.322,
                'switch_v':28,'X0':0,'Y0':0
                }


    if num == '1a_noext':
        return {'nX':100,'nY':100,
                'zeta':.4,
                'A':5,'B':5.00001,
                'alpha':10,'beta':150,
                'p1':4/1,'gamma':0.322,
                'switch_v':99,'X0':0,'Y0':0,
                'extension':False
                }


    if num == '1b_noext':
        return {'nX':100,'nY':100,
                'zeta':.4,
                'A':5,'B':5.01,
                'alpha':10,'beta':150,
                'p1':4/1,'gamma':0.322,
                'switch_v':99,'X0':0,'Y0':0,
                'extension':False
                }

    if num == '2a_noext':
        return {'nX':100,'nY':100,
                'zeta':.7,
                'A':5,'B':5.01,
                'alpha':14,'beta':300,
                'p1':4/1,'gamma':0.322,
                'switch_v':37,'X0':0,'Y0':0,
                'extension':False
                }

    if num == '2a':
        # use this to experiment with parameters before giving it a name
        return {'nX':100,'nY':100,
                'zeta':.7,
                'A':5,'B':5.01,
                'alpha':14,'beta':300,
                'p1':4/1,'gamma':0.322,
                'switch_v':36,'X0':0,'Y0':0,
                'extension':True
                }
                
            
    if num == 'figure1':
        return {'nX':100,'nY':100,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/1,'gamma':0.322,
                'switch_v':121,
                }

    if num == 'figure1.50':
        return {'nX':50,'nY':50,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/.5,'gamma':0.322,
                'switch_v':121,
                }

    if num == 'figure1.500':
        return {'nX':500,'nY':500,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/5,'gamma':0.322,
                'switch_v':121,
                }


    if num == 'figure1.1000':
        return {'nX':1000,'nY':1000,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/10,'gamma':0.322,
                'switch_v':121,
                }


    if num == 'figure1.1500':
        return {'nX':1500,'nY':1500,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/15,'gamma':0.322,
                'switch_v':121,
                }


    if num == 'figure1.2000':
        return {'nX':2000,'nY':2000,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/20,'gamma':0.322,
                'switch_v':121,
                }


    if num == 'figure1.2500':
        return {'nX':2500,'nY':2500,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/25,'gamma':0.322,
                'switch_v':121,
                }

    if num == 'figure1.3000':
        return {'nX':3000,'nY':3000,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/30,'gamma':0.322,
                'switch_v':121,
                }


    if num == 'figure1.3500':
        return {'nX':3500,'nY':3500,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/35,'gamma':0.322,
                'switch_v':121,
                }



    if num == 'figure1.4000':
        return {'nX':4000,'nY':4000,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/40,'gamma':0.322,
                'switch_v':121,
                }


    if num == 'figure1.4500':
        return {'nX':4500,'nY':4500,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/45,'gamma':0.322,
                'switch_v':121,
                }


    if num == 'figure1.5000':
        return {'nX':5000,'nY':5000,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/50,'gamma':0.322,
                'switch_v':121,
                }

    if num == 'figure1.6000':
        return {'nX':6000,'nY':6000,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/60,'gamma':0.322,
                'switch_v':121,
                }

    
    if num == 'figure1.7000':
        return {'nX':7000,'nY':7000,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/70,'gamma':0.322,
                'switch_v':121,
                }

    if num == 'figure1.8000':
        return {'nX':8000,'nY':8000,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/80,'gamma':0.322,
                'switch_v':121,
                }


    if num == 'figure1.9000':
        return {'nX':9000,'nY':9000,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/90,'gamma':0.322,
                'switch_v':121,
                }


    if num == 'figure1.10000':
        return {'nX':10000,'nY':10000,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/100,'gamma':0.322,
                'switch_v':121,
                }


    if num == 'figure1.11000':
        return {'nX':11000,'nY':11000,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/110,'gamma':0.322,
                'switch_v':121,
                }

    if num == 'figure1.12000':
        return {'nX':12000,'nY':12000,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/120,'gamma':0.322,
                'switch_v':121,
                }

    
    if num == 'figure2':
        return {'nX':100,'nY':100,
                'zeta':.4,
                'A':5,'B':5.05,
                'alpha':14,'beta':126,
                'p1':4/1,'gamma':0.322,
                'switch_v':50,'X0':0,'Y0':0
                }


    
    else:
        raise ValueError('Invalid option', num)

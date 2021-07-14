# -*- coding: utf-8 -*-
"""
create function-like object for interpolation.

use np.interp for speed as opposed to interp1d. Too much overhead in interp1d.
"""

import numpy as np
from scipy.interpolate import interp1d#, interp2d
from numpy.core.multiarray import interp as compiled_interp

class interp_basic(object):
    
    def __init__(self,x,y):
        
        self.x = x
        self.y = y
        
        #self.interp = interp1d(self.x,self.y,kind='linear')

    def __call__(self, x_new):
        #return self.interp(x_new)
        return compiled_interp(x_new, self.x, self.y,
                               left=False, right=False)

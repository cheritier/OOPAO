# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 08:44:13 2022

@author: cheritie
"""

from ..Telescope import Telescope
from ..Source import Source
from ..Atmosphere import Atmosphere


def get_fast_atmosphere(obj,param,speed_factor):
    # create the Telescope object
    tel_fast = Telescope(resolution          = param['resolution'],\
                         diameter            = param['diameter'],\
                         samplingTime        = obj.tel.samplingTime/speed_factor,\
                         centralObstruction  = param['centralObstruction'])

    # create the Source object
    ngs_fast = Source(optBand   = param['opticalBand'],\
               magnitude = 0)
    
    # combine the NGS to the telescope using '*' operator:
    ngs_fast*tel_fast
    
    # create the Atmosphere object
    atm_fast = Atmosphere(telescope     = tel_fast,\
                          r0            = param['r0'],\
                          L0            = param['L0'],\
                          windSpeed     = param['windSpeed'],\
                          fractionalR0  = param['fractionnalR0'],\
                          windDirection = param['windDirection'],\
                          altitude      = param['altitude'],\
                          param = param)
        
    # initialize atmosphere
    atm_fast.initializeAtmosphere(tel_fast)
    atm_fast.update()
    
    return tel_fast,atm_fast


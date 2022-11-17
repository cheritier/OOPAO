
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:41:21 2020

@author: cheritie
"""

from OOPAO.Telescope         import Telescope
from OOPAO.Source            import Source
from OOPAO.Atmosphere        import Atmosphere
from OOPAO.Pyramid           import Pyramid
from OOPAO.DeformableMirror  import DeformableMirror
from OOPAO.MisRegistration   import MisRegistration
from OOPAO.tools.set_paralleling_setup import set_paralleling_setup


def run_initialization_basic_AO(param):
    
    #%% -----------------------     TELESCOPE   ----------------------------------
    
    # create the Telescope object
    tel = Telescope(resolution          = param['resolution'],\
                    diameter            = param['diameter'],\
                    samplingTime        = param['samplingTime'],\
                    centralObstruction  = param['centralObstruction'])
    
    #%% -----------------------     NGS   ----------------------------------
    # create the Source object
    ngs=Source(optBand   = param['opticalBand'],\
               magnitude = param['magnitude'])
    
    # combine the NGS to the telescope using '*' operator:
    ngs*tel
    
    
    
    #%% -----------------------     ATMOSPHERE   ----------------------------------

    # create the Atmosphere object
    atm=Atmosphere(telescope     = tel,\
                   r0            = param['r0'],\
                   L0            = param['L0'],\
                   windSpeed     = param['windSpeed'],\
                   fractionalR0  = param['fractionnalR0'],\
                   windDirection = param['windDirection'],\
                   altitude      = param['altitude'],\
                   param         = param)
    
    # initialize atmosphere
    atm.initializeAtmosphere(tel)
    
    # pairing the telescope with the atmosphere using '+' operator
    tel+atm
    # separating the tel and atmosphere using '-' operator
    tel-atm
    
    #%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
    # mis-registrations object
    misReg = MisRegistration(param)
    # if no coordonates specified, create a cartesian dm
    dm=DeformableMirror(telescope    = tel,\
                        nSubap       = param['nSubaperture'],\
                        mechCoupling = param['mechanicalCoupling'],\
                        misReg       = misReg)
    
    
    #%% -----------------------     PYRAMID WFS   ----------------------------------
    
    # make sure tel and atm are separated to initialize the PWFS
    tel-atm
    try:
        print('Using a user-defined zero-padding for the PWFS:', param['zeroPadding'])
    except:
        param['zeroPadding']=0
        print('Using the default zero-padding for the PWFS')
    # create the Pyramid Object
    wfs = Pyramid(nSubap                = param['nSubaperture'],\
                  telescope             = tel,\
                  modulation            = param['modulation'],\
                  lightRatio            = param['lightThreshold'],\
                  pupilSeparationRatio  = param['pupilSeparationRatio'],\
                  calibModulation       = param['calibrationModulation'],\
                  psfCentering          = param['psfCentering'],\
                  edgePixel             = param['edgePixel'],\
                  unitCalibration       = param['unitCalibration'],\
                  extraModulationFactor = param['extraModulationFactor'],\
                  postProcessing        = param['postProcessing'],\
                  zeroPadding           = param['zeroPadding'])
    
    set_paralleling_setup(wfs, ELT = False)
    # propagate the light through the WFS
    tel*wfs

    class emptyClass():
        pass
    # save output as sub-classes
    simulationObject = emptyClass()
    
    simulationObject.tel    = tel
    simulationObject.atm    = atm
    simulationObject.ngs    = ngs
    simulationObject.dm     = dm
    simulationObject.wfs    = wfs
    simulationObject.param  = param
 

    return simulationObject



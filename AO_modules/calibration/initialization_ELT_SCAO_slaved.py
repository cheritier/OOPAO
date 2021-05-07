# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 15:29:11 2021

@author: cheritie
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:41:21 2020

@author: cheritie
"""
# commom modules
import matplotlib.pyplot as plt

# local modules 
from AO_modules.Telescope         import Telescope
from AO_modules.Source            import Source
from AO_modules.Atmosphere        import Atmosphere
from AO_modules.Pyramid           import Pyramid
from AO_modules.DeformableMirror  import DeformableMirror
from AO_modules.MisRegistration   import MisRegistration
from AO_modules.tools.set_paralleling_setup import set_paralleling_setup

# ELT modules
from AO_modules.M1_model.make_ELT_pupil                     import generateEeltPupilReflectivity
from AO_modules.M4_model.make_M4_influenceFunctions         import getPetalModes

def run_initialization_ELT_SCAO(param):
    plt.ioff()
    #%% -----------------------     TELESCOPE   ----------------------------------
    # create the pupil of the ELT
    M1_pupil_reflectivity = generateEeltPupilReflectivity(refl      = param['m1_reflectivityy'],\
                                              npt       = param['resolution'],\
                                              dspider   = param['spiderDiameter'],\
                                              i0        = param['m1_center'][0]+param['m1_shiftX'] ,\
                                              j0        = param['m1_center'][1]+param['m1_shiftY'] ,\
                                              pixscale  = param['pixelSize'],\
                                              gap       = param['gapSize'],\
                                              rotdegree = param['m1_rotationAngle'],\
                                              softGap   = True)
    
    # extract the valid pixels
    M1_pupil = M1_pupil_reflectivity>0
    
    # create the Telescope object
    tel = Telescope(resolution          = param['resolution'],\
                    diameter            = param['diameter'],\
                    samplingTime        = param['samplingTime'],\
                    centralObstruction  = param['centralObstruction'],\
                    pupilReflectivity   = M1_pupil_reflectivity,\
                    pupil               = M1_pupil)
    
    
    #%% -----------------------     NGS   ----------------------------------
    # create the Source object
    ngs=Source(optBand   = param['opticalBand'],\
               magnitude = param['magnitude'])
    
    # combine the NGS to the telescope using '*' operator:
    ngs*tel
    
    #%% -----------------------     ATMOSPHERE   ----------------------------------
#    
    # create the Atmosphere object
    atm=Atmosphere(telescope     = tel,\
                   r0            = param['r0'],\
                   L0            = param['L0'],\
                   windSpeed     = param['windSpeed'],\
                   fractionalR0  = param['fractionnalR0'],\
                   windDirection = param['windDirection'],\
                   altitude      = param['altitude'])
    
    # initialize atmosphere
    atm.initializeAtmosphere(tel)
    
    # pairing the telescope with the atmosphere using '+' operator
    tel+atm
    # separating the tel and atmosphere using '-' operator
    tel-atm
    
    #%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
    # mis-registrations object

    misReg=MisRegistration(param)
    
    # generate the M4 influence functions
    dm = DeformableMirror(telescope    = tel,\
                          nSubap       = param['nSubaperture'],\
                          misReg       = misReg,\
                          M4_param     = param)
    try:
        petals,petals_float = getPetalModes(tel,dm,[1,2,3,4,5,6])
    except:
        petals,petals_float = getPetalModes(tel,dm,[1])
    tel.index_pixel_petals = petals
    tel.isPetalFree =True

    from AO_modules.M4_model.get_slaved_m4 import get_slaved_m4
    
    dm = get_slaved_m4(dm)
    
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
    
    # propagate the light through the WFS
    tel*wfs
    set_paralleling_setup(wfs)

    class emptyClass():
        pass
    # save output as sub-classes
    simulationObject = emptyClass()
    
    simulationObject.tel             = tel
    simulationObject.atm             = atm
    simulationObject.ngs             = ngs
    simulationObject.dm              = dm
    simulationObject.wfs             = wfs
    simulationObject.param           = param
    simulationObject.misReg          = misReg


    return simulationObject



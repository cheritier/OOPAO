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

    misReg=MisRegistration(param)
    
    # generate the M4 influence functions
    dm = DeformableMirror(telescope    = tel,\
                          nSubap       = param['nSubaperture'],\
                          misReg       = misReg,\
                          M4_param     = param)
        
        
        
    from joblib import Parallel, delayed    
    import numpy as np
  
    def get_distance(x1,x2,y1,y2):
        r = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        return r 
    
    def joblib_distance():
        Q=Parallel(n_jobs=8,prefer='threads')(delayed(get_distance)(i,j,k,l) for i,j,k,l in zip(coord_x_1,coord_x_2,coord_y_1,coord_y_2))
        return Q 
    
    from AO_modules.M4_model.get_M4_coordinates                 import get_M4_coordinates
    coord, coord_m = get_M4_coordinates(M1_pupil,param['m4_filename'],nAct =param['nActuator'])

    dist = np.zeros([892,892])
    c = np.zeros([892,2])
    c[:,0] = np.flip(coord_m[:892,0])
    c[:,1] = np.flip(coord_m[:892,1])
    
    for i_act in range(892):
        coord_x_1       = c[:892,0] 
        coord_x_2       = np.tile(c[i_act,0],892)
        coord_y_1       = c[:892,1] 
        coord_y_2       = np.tile(c[i_act,1],892)
        dist[i_act,:]   = np.asarray(joblib_distance())
    
    dist[dist>0.8] =0
    
    index_coupling = []
    for i in range(892):
        index_coupling.append(np.where(dist[i,:] >0.1))
        
        
        
    dm_modes = np.copy(dm.modes)
    
    for i_petal in range(6):
        for i in range(892):
            print(i+892*i_petal)
            vect = index_coupling[i][0]
            for j in range(len(vect)):
                print('Coupling with actuator'+str(vect[j]+892*i_petal))
                dm_modes[:,i+892*i_petal]+=param['mechanicalCoupling']*dm.modes[:,i_petal*892+vect[j]]
            
          
    dm.modes = dm_modes
            
            
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



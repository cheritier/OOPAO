# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:36:02 2020

@author: cheritie
"""

import numpy as np
from os import path
from OOPAO.tools.tools  import createFolder
from numpy.random import RandomState


def initializeParameterFile():
    # initialize the dictionaries
    param = dict()
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['r0'                   ] = 0.15                                                                                                           # value of r0 @500 nm in [m]
    param['L0'                   ] = 30                                                                                                             # value of r0 @500 nm in [m]
    param['fractionnalR0'        ] = [0.5179,0.1892,0.0723,0.0579,0.0324,0.03802,0.0419,0.02897,0.02141]                                            # Cn2 profile
    
    N = 9
    # create a random state to ensure repeatability of the random generator
    randomState = RandomState(42)
    # SLOW
    param['windSpeed'            ] = list(randomState.randint(10,15,N))                                                                             # wind speed of the different layers in [m.s-1]
    param['windDirection'        ] = list(randomState.randint(0,360,N))                                                                             # wind direction of the different layers in [degrees]
    param['altitude'             ] = [  60.0872,  383.75,  1428.0275,  3512.6681,  6601.7719,  10083.8873, 13223.7837,  14710.7556,  17679.3437]    # altitude of the different layers in [degrees]
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['diameter'             ] = 8                                              # diameter in [m]
    param['nSubaperture'         ] = 30                                             # number of PWFS subaperture along the telescope diameter
    param['nPixelPerSubap'       ] = 4                                              # sampling of the PWFS subapertures
    param['resolution'           ] = param['nSubaperture']*param['nPixelPerSubap']  # resolution of the telescope driven by the PWFS
    param['sizeSubaperture'      ] = param['diameter']/param['nSubaperture']        # size of a sub-aperture projected in the M1 space
    param['samplingTime'         ] = 1/500                                         # loop sampling time in [s] == AO Loop frequency
    param['centralObstruction'   ] = 0                                              # central obstruction in percentage of the diameter
    param['nMissingSegments'     ] = 0                                              # number of missing segments on the M1 pupil
    param['m1_reflectivity'      ] = 1                                              # reflectivity of the pupil
          
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['magnitude'            ] = 1                                              # magnitude of the guide star
    param['opticalBand'          ] = 'I'                                            # optical band of the guide star
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nActuator'            ] = param['nSubaperture']+1                        # number of actuators 
    param['mechanicalCoupling'   ] = 0.45                                           # mechanical coupling between actuators
    param['isM4'                 ] = False                                          # tag for the deformable mirror class
    param['dm_coordinates'       ] = None                                           # using user-defined coordinates
    
    # mis-registrations                                                             
    param['shiftX'               ] = 0                                              # shift X of the DM in pixel size units ( tel.D/tel.resolution ) 
    param['shiftY'               ] = 0                                              # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
    param['rotationAngle'        ] = 0                                              # rotation angle of the DM in [degrees]
    param['anamorphosisAngle'    ] = 0                                              # anamorphosis angle of the DM in [degrees]
    param['radialScaling'        ] = 0                                              # radial scaling in percentage of diameter
    param['tangentialScaling'    ] = 0                                              # tangential scaling in percentage of diameter
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% KL MODAL BASIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['modal_basis_name'     ] ='M2C_'+ str(param['resolution'])+'_res_stage_1'         # name of the modal basis
    param['nModes'               ] = 600                                            # number of KL modes controlled 
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # PWFS
    param['modulation'            ] = 3                                             # modulation radius in ratio of wavelength over telescope diameter
    param['n_pix_separation'  ]     = 4                                           # separation ratio between the PWFS pupils
    param['psfCentering'          ] = False                                         # centering of the FFT and of the PWFS mask on the 4 central pixels
    param['calibrationModulation' ] = 50                                            # modulation radius used to select the valid pixels
    param['lightThreshold'        ] = 0.1                                           # light threshold to select the valid pixels
    param['edgePixel'             ] = 1                                             # number of pixel on the external edge of the PWFS pupils
    param['extraModulationFactor' ] = 0                                             # factor to add/remove 4 modulation points (one for each PWFS face)
    # param['postProcessing'        ] = 'fullFrame'                                 # post-processing of the PWFS signals 
    param['postProcessing'        ] = 'slopesMaps'                                  # post-processing of the PWFS signals 
    param['unitCalibration'       ] = False                                         # calibration of the PWFS units using a ramp of Tip/Tilt    
    
    # SHWFS
    param['lightThreshold'        ] = 0.1                                           # light threshold to select the valid pixels
    param['is_geometric'          ] = False                                         # post-processing of the PWFS signals 
        
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOOP PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    param['nLoop'                 ] = 5000                                         # number of iteration                             
    param['photonNoise'           ] = True                                         # Photon Noise enable  
    param['readoutNoise'          ] = 0.5                                          # Readout Noise value
    param['gainCL'                ] = 0.5                                          # integrator gain
    param['getProjector'          ] = True                                         # modal projector too get modal coefficients of the turbulence and residual phase

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # name of the system
    param['name'] = 'VLT_SPHERE_' +  param['opticalBand'] +'_band_'+ str(param['nSubaperture'])+'x'+ str(param['nSubaperture'])  
    
    # location of the calibration data
    param['pathInput'            ] = 'data_calibration/' 
    
    # location of the output data
    param['pathOutput'            ] = '/diskb/cheritier/psim/data_cl'
    

    print('Reading/Writting calibration data from ' + param['pathInput'])
    print('Writting output data in ' + param['pathOutput'])
    createFolder(param['pathInput'])
    createFolder(param['pathOutput'])
    
    return param
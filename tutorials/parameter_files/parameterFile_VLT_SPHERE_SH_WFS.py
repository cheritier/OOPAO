# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:36:02 2020

@author: cheritie
"""

import numpy as np
from os import path
from AO_modules.tools.tools  import createFolder


def initializeParameterFile():
    # initialize the dictionaries
    param = dict()
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['r0'                   ] = 0.15                                           # value of r0 in the visibile in [m]
    param['L0'                   ] = 30                                             # value of L0 in the visibile in [m]
    param['fractionnalR0'        ] = [0.45,0.1,0.1,0.25,0.1]                                            # Cn2 profile
    param['windSpeed'            ] = [10,12,11,15,20]                                           # wind speed of the different layers in [m.s-1]
    param['windDirection'        ] = [0,72,144,216,288]                                            # wind direction of the different layers in [degrees]
    param['altitude'             ] = [0, 1000,5000,10000,12000 ]                                         # altitude of the different layers in [m]
               
                              
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['diameter'             ] = 8.2                                             # diameter in [m]
    param['nSubaperture'         ] = 40                                             # number of PWFS subaperture along the telescope diameter
    param['nPixelPerSubap'       ] = 6                                            # sampling of the PWFS subapertures
    param['resolution'           ] = param['nSubaperture']*param['nPixelPerSubap']  # resolution of the telescope driven by the PWFS
    param['sizeSubaperture'      ] = param['diameter']/param['nSubaperture']        # size of a sub-aperture projected in the M1 space
    param['samplingTime'         ] = 1/1000                                         # loop sampling time in [s]
    param['centralObstruction'   ] = 0.12                                              # central obstruction in percentage of the diameter
    param['nMissingSegments'     ] = 0                                             # number of missing segments on the M1 pupil
    param['m1_reflectivity'      ] = 1                                   # reflectivity of the 798 segments
          
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['magnitude'            ] = 8                                              # magnitude of the guide star
    param['opticalBand'          ] = 'R'                                            # optical band of the guide star
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nActuator'            ] = param['nSubaperture']+1                                             # number of actuators 
    param['mechanicalCoupling'   ] = 0.4
    param['isM4'                 ] = False                                           # tag for the deformable mirror class
    param['dm_coordinates'       ] = None                                           # tag for the eformable mirror class
    
    # mis-registrations                                                             
    param['shiftX'               ] = 0                                              # shift X of the DM in pixel size units ( tel.D/tel.resolution ) 
    param['shiftY'               ] = 0                                              # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
    param['rotationAngle'        ] = 0                                              # rotation angle of the DM in [degrees]
    param['anamorphosisAngle'    ] = 0                                              # anamorphosis angle of the DM in [degrees]
    param['radialScaling'        ] = 0                                              # radial scaling in percentage of diameter
    param['tangentialScaling'    ] = 0                                              # tangential scaling in percentage of diameter
    

    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['modulation'            ] = 3                                             # modulation radius in ratio of wavelength over telescope diameter
    param['lightThreshold'        ] = 0.5                                           # light threshold to select the valid pixels
    param['unitCalibration'       ] = False                                         # calibration of the PWFS units using a ramp of Tip/Tilt    
    param['is_geometric'          ] = False
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOOP PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    param['nLoop'                 ] = 5000                                           # number of iteration                             
    param['photonNoise'           ] = True                                         # Photon Noise enable  
    param['readoutNoise'          ] = 0                                            # Readout Noise value
    param['gainCL'                ] = 0.5                                          # integrator gain
    param['nModes'                ] = 600                                          # number of KL modes controlled 
    param['nPhotonPerSubaperture' ] = 1000                                         # number of photons per subaperture (update of ngs.magnitude)
    param['getProjector'          ] = True                                         # modal projector too get modal coefficients of the turbulence and residual phase

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # name of the system
    param['name'] = 'VLT_' +  param['opticalBand'] +'_band_'+ str(param['nSubaperture'])+'x'+ str(param['nSubaperture'])  
    
    # location of the calibration data
    param['pathInput'            ] = 'data_calibration/' 
    
    # location of the output data
    param['pathOutput'            ] = 'data_cl/'
    

    print('Reading/Writting calibration data from ' + param['pathInput'])
    print('Writting output data in ' + param['pathOutput'])

    createFolder(param['pathOutput'])
    
    return param
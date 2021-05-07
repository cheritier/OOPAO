# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 08:18:32 2020

@author: cheritie
"""
import numpy as np
from os import path
from AO_modules.tools.tools  import createFolder


def initializeParameterFile():
    # initialize the dictionaries
    param = dict()
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['r0'                   ] = 0.13                                           # value of r0 in the visibile in [m]
    param['L0'                   ] = 30                                             # value of L0 in the visibile in [m]
    param['fractionnalR0'        ] = [0.25, 0.75]                                            # Cn2 profile
    param['windSpeed'            ] = [10  , 20  ]                                           # wind speed of the different layers in [m.s-1]
    param['windDirection'        ] = [0   , 60  ]                                            # wind direction of the different layers in [degrees]
    param['altitude'             ] = [100 , 1000]                                         # altitude of the different layers in [m]
    
                              
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['diameter'             ] = 40                                             # diameter in [m] -- hard-coded
    param['nSubaperture'         ] = 96                                             # number of PWFS subaperture along the telescope diameter
    param['nPixelPerSubap'       ] = 6                                              # sampling of the PWFS subapertures
    param['resolution'           ] = param['nSubaperture']*param['nPixelPerSubap']  # resolution of the telescope driven by the PWFS
    param['sizeSubaperture'      ] = param['diameter']/param['nSubaperture']        # size of a sub-aperture projected in the M1 space
    param['samplingTime'         ] = 1/1000                                         # loop sampling time of the AO loop in [s]
    param['centralObstruction'   ] = 0                                              # central obstruction in percentage of the diameter
    param['nMissingSegments'     ] = 0                                              # number of missing segments on the M1 pupil
    param['m1_reflectivityy'     ] = np.ones(798)                                   # reflectivity of the 798 segments
    param['isPetalFree'          ] = False                                          # remove the petal contribution from the telescope OPD
#    param['m1_reflectivityy'     ] -= 0 * np.random.randn(798)/20                  # random non uniform reflectivity of the 798 segments
#    param['m1_reflectivityy'][(np.random.rand(param['nMissingSegments'])*797).astype(int)] = 0.# reflectivity of the missing segment set to 0
    
       
    offset    = -60                                                                         # rotation offset to align M1 and M4
    rotdegree = 0                                                                           # rotation of the M1
    param['m1_rotationAngle'     ] = offset + rotdegree                                     # effective rotation of the M1 model
    param['pixelSize'            ] = param['diameter']/param['resolution']                  # number of missing segments on the M1 pupil
    param['m1_shiftX'            ] = -1                                                     # shift X of the M1 pupil in pixel size
    param['m1_shiftY'            ] = -1                                                     # shift Y of the M1 pupil in pixel size
    param['m1_center'            ] = [param['resolution']/2+0.5,param['resolution']/2+0.5]  # center of the M1 pupil
    param['spiderDiameter'       ] = 0.51                                                   # spiders diameter in [m]
    param['gapSize'              ] = 0.0                                                    # gap between segments
          
          
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['magnitude'            ] = 8                                              # magnitude of the guide star
    param['opticalBand'          ] = 'I'                                            # optical band of the guide star see Source.py for the definition of the optical bands
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M4 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    path_local =  'C:/Users/cheritie/myHome/SCAO/cropped_M4IF.fits'
    path_remote = '/Disk3/cheritier/psim/data_calibration/cropped_M4IF.fits'
    
    if path.exists(path_local):
        path_m4 = path_local
    else:
        if path.exists(path_remote):
            path_m4 = path_remote
        else:
            print('ERROR NO FILE FOUND FOR THE M4 INFLUENCE FUNCTIONS')
        
    param['m4_filename'          ] = path_m4
    param['validActCriteria'     ] = 0
    param['nActuator'            ] = 6*892                                          # number of actuators to consider for M4 
    param['isM4'                 ] = True                                           # tag for the deformable mirror class
    param['dm_coordinates'       ] = None                                           # tag for the eformable mirror class
    param['mechanicalCoupling']    = 0                                              # mechanical coupling (non-applicable for M4 case or user-defined influence functions )
    
    # mis-registrations                                                             
    param['shiftX'               ] = 0                                              # shift X of the DM in pixel size units ( tel.D/tel.resolution ) 
    param['shiftY'               ] = 0                                              # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
    param['rotationAngle'        ] = rotdegree                                      # rotation angle of the DM in [degrees]
    param['anamorphosisAngle'    ] = 0                                              # anamorphosis angle of the DM in [degrees]
    param['radialScaling'        ] = 0                                              # radial scaling in percentage of diameter
    param['tangentialScaling'    ] = 0                                              # tangential scaling in percentage of diameter
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['modulation'            ] = 3                                             # modulation radius in ratio of wavelength over telescope diameter
    param['pupilSeparationRatio'  ] = 1                                             # separation ratio between the PWFS pupils
    param['psfCentering'          ] = False                                         # centering of the FFT and of the PWFS mask on the 4 central pixels
    param['calibrationModulation' ] = 50                                            # modulation radius used to select the valid pixels
    param['lightThreshold'        ] = 0.1                                           # light threshold to select the valid pixels
    param['edgePixel'             ] = 0                                             # number of pixel on the external edge of the PWFS pupils
    param['extraModulationFactor' ] = 0                                             # factor to add/remove 4 modulation points (one for each PWFS face)
    param['postProcessing'        ] = 'fullFrame'                                   # post-processing of the PWFS signals WARNING
#    param['postProcessing'        ] = 'slopesMaps'                                   # post-processing of the PWFS signals 
    param['unitCalibration'       ] = False                                         # calibration of the PWFS units using a ramp of Tip/Tilt    
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOOP PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    param['nLoop'                 ] = 50                                           # number of iteration                             
    param['photonNoise'           ] = True                                         # Photon Noise enable  
    param['readoutNoise'          ] = 0                                            # Readout Noise value
    param['gainCL'                ] = 0.5                                          # integrator gain
    param['nModes'                ] = 3000                                         # number of KL modes controlled 
    param['nPhotonPerSubaperture' ] = 1000                                         # number of photons per subaperture (update of ngs.magnitude)
    param['getProjector'          ] = True                                         # modal projector too get modal coefficients of the turbulence and residual phase

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # name of the system
    param['name'] = 'ELT_' +  param['opticalBand'] +'_band_'+ str(param['nSubaperture'])+'x'+ str(param['nSubaperture'])  
    
    # location of the calibration data
    param['pathInput'            ] = '/Disk3/cheritier/psim/data_calibration/' 
    
    # location of the output data
    param['pathOutput'            ] = '/diskb/cheritier/psim/data_cl'
    

    print('Reading/Writting calibration data from ' + param['pathInput'])
    print('Writting output data in ' + param['pathOutput'])

    createFolder(param['pathOutput'])
    
    return param
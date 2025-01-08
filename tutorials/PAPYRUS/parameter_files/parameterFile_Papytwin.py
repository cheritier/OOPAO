# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:12:28 2024

@author: cheritier
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 10:40:42 2023

@author: cheritie - astriffl
"""

from OOPAO.tools.tools import createFolder

def initializeParameterFile():
    # initialize the dictionaries
    param = dict()
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['r0'                   ] = 0.1                                           # value of r0 in the visibile in [m]
    param['L0'                   ] = 30                                             # value of L0 in the visibile in [m]
    param['fractionnalR0'        ] = [0.45,0.1,0.1,0.25,0.1]                        # Cn2 profile
    param['windSpeed'            ] = [5,4,8,10,2]                               # wind speed of the different layers in [m.s-1]
    param['windDirection'        ] = [0,72,144,216,288]                             # wind direction of the different layers in [degrees]
    param['altitude'             ] = [0, 1000,5000,10000,12000 ]                    # altitude of the different layers in [m]
               
                              
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['diameter'             ] = 1.52                                          # diameter in [m]
    param['nSubaperture'         ] = 80                                            # number of PWFS subaperture along the telescope diameter
    param['ratio'                ] = 80//param['nSubaperture'         ]
    param['nPixelPerSubap'       ] = 2                                             # sampling of the PWFS subapertures
    param['resolution'           ] = param['nSubaperture']*param['nPixelPerSubap'] # resolution of the telescope driven by the PWFS
    param['sizeSubaperture'      ] = param['diameter']/param['nSubaperture']       # size of a sub-aperture projected in the M1 space
    param['samplingTime'         ] = 1/500                                         # loop sampling time in [s]
    param['centralObstruction'   ] = 0                                             # central obstruction in percentage of the diameter
    param['nMissingSegments'     ] = 0                                             # number of missing segments on the M1 pupil
    param['m1_reflectivity'      ] = 0.01                                             # reflectivity of the 798 segments
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['magnitude'            ] = -0.04                                              # magnitude of the guide star
    param['opticalBand'          ] = 'K'                                            # optical band of the guide star
    param['opticalBandCalib'     ] = 'R'                                            # optical band of calibration laser
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nActuator'            ] = 17                                             # number of actuators 
    param['mechanicalCoupling'   ] = 0.36
    param['isM4'                 ] = False                                          # tag for the deformable mirror class
    param['dm_coordinates'       ] = None                                           # tag for the deformable mirror class
    
    # mis-registrations                                                             
    # param['shiftX'               ] = 0.006                                              # shift X of the DM in pixel size units ( tel.D/tel.resolution ) 
    # param['shiftY'               ] = -0.003                                              # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
    # param['rotationAngle'        ] = 0.69   -90                                           # rotation angle of the DM in [degrees]
    # param['anamorphosisAngle'    ] = 0                                              # anamorphosis angle of the DM in [degrees]
    # param['radialScaling'        ] = -0.049  -0.02                                           # radial scaling in percentage of diameter
    # param['tangentialScaling'    ] = -0.041  -0.02                                          # tangential scaling in percentage of diameter
    
    # new values
    
    param['shiftX'               ] = 0.002                                              # shift X of the DM in pixel size units ( tel.D/tel.resolution ) 
    param['shiftY'               ] = 0.003                                              # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
    param['rotationAngle'        ] = -89.555                                           # rotation angle of the DM in [degrees]
    param['anamorphosisAngle'    ] = 0                                              # anamorphosis angle of the DM in [degrees]
    param['radialScaling'        ] = -0.135                                           # radial scaling in percentage of diameter
    param['tangentialScaling'    ] = -0.132                                          # tangential scaling in percentage of diameter
    

        ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    param['modulation'            ] = 5                                             # modulation radius in ratio of wavelength over telescope diameter
    param['n_pix_separation'      ] = 0                                            # separation ratio between the PWFS pupils
    param['psfCentering'          ] = False                                         # centering of the FFT and of the PWFS mask on the 4 central pixels
    param['lightThreshold'        ] = 0.3                                           # light threshold to select the valid pixels
    param['postProcessing'        ] = 'fullFrame'                                   # post-processing of the PWFS signals 'slopesMaps' ou 'fullFrame'


    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOOP PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    param['nLoop'                 ] = 5000                                         # number of iteration                             
    param['photonNoise'           ] = True                                         # Photon Noise enable  
    param['readoutNoise'          ] = 0                                            # Readout Noise value
    param['gainCL'                ] = 0.5                                          # integrator gain
    param['nModes'                ] = 600                                          # number of KL modes controlled 
    param['nPhotonPerSubaperture' ] = 1000                                         # number of photons per subaperture (update of ngs.magnitude)
    param['getProjector'          ] = True                                         # modal projector to get modal coefficients of the turbulence and residual phase

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # name of the system
    param['name'] = 'PAPYRUS_' +  param['opticalBand'] +'_band_'+ str(param['nSubaperture'])+'x'+ str(param['nSubaperture'])  
    
    # location of the calibration data
    param['pathInput'            ] = 'data_calibration/' 
    
    # location of the output data
    param['pathOutput'            ] = 'data_cl/'
    

    print('Reading/Writting calibration data from ' + param['pathInput'])
    print('Writting output data in ' + param['pathOutput'])

    createFolder(param['pathOutput'])
    
    return param


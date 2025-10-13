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
    
    param['r0'                   ] = 0.15                                           # value of r0 in the visibile in [m]
    param['L0'                   ] = 30                                             # value of L0 in the visibile in [m]
    param['fractionnalR0'        ] = [1]                        # Cn2 profile
    param['windSpeed'            ] = [2]                               # wind speed of the different layers in [m.s-1]
    param['windDirection'        ] = [0]                             # wind direction of the different layers in [degrees]
    param['altitude'             ] = [0]                    # altitude of the different layers in [m]
               
                              
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['nSubaperture'         ] = 76                                                                           # number of PWFS subaperture along the telescope diameter
    param['nExtraSubaperture'    ] = 4                                                                            # extra subaperture on the edges
    param['diameter'             ] = 1.52 *(param['nSubaperture'] + param['nExtraSubaperture'])/param['nSubaperture'] # diameter in [m]
    param['ratio'                ] = 80//(param['nSubaperture'] + param['nExtraSubaperture'])                     # ratio factor for binned case
    param['nPixelPerSubap'       ] = 1                                                                            # sampling of the PWFS subapertures in pix
    param['resolution'           ] = (param['nSubaperture'] + param['nExtraSubaperture'])*param['nPixelPerSubap'] # resolution of the telescope driven by the PWFS
    param['sizeSubaperture'      ] = param['diameter']/(param['nSubaperture'] + param['nExtraSubaperture'])       # size of a sub-aperture projected in the M1 space
    param['samplingTime'         ] = 1/500                                                                        # loop sampling time in [s]
    param['m1_reflectivity'      ] = 0.01                                                                         # reflectivity of the pupil
    param['fieldOfView'          ] = 0 #field of view in arcsec => 0 is for the minimal value (speeding up the atmosphere computation)

    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['magnitude'            ] = -0.04                                          # magnitude of the guide star
    param['opticalBand'          ] = 'J2'                                            # optical band of the guide star
    param['opticalBandCalib'     ] = 'R'                                            # optical band of calibration laser
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nActuator'            ] = 17                                             # number of actuators 
    param['mechanicalCoupling'   ] = 0.36
    param['isM4'                 ] = False                                          # tag for the deformable mirror class
    param['dm_coordinates'       ] = None                                           # tag for the deformable mirror class
    
    # mis-registrations                                                             
    # latest value 16062025
    param['rotationAngle'        ] = 0.5                                           # rotation angle of the DM in [degrees]    
    param['shiftX'               ] = -0.004                                              # shift X of the DM in pixel size units ( tel.D/tel.resolution ) 
    param['shiftY'               ] = 0.005                                              # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
    param['anamorphosisAngle'    ] = 0                                              # anamorphosis angle of the DM in [degrees]
    param['tangentialScaling'    ] = -0.025                                          # tangential scaling in percentage of diameter
    param['radialScaling'        ] = -0.031                                           # radial scaling in percentage of diameter
    param['dm_inf_funct_factor'  ] = 30*7591.024876
 

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['modulation'            ] = 5                                             # modulation radius in ratio of wavelength over telescope diameter
    param['n_pix_separation'      ] = 0                                            # separation ratio between the PWFS pupils
    param['psfCentering'          ] = False                                         # centering of the FFT and of the PWFS mask on the 4 central pixels
    param['lightThreshold'        ] = 0.3                                           # light threshold to select the valid pixels
    param['postProcessing'        ] = 'fullFrame'                                   # post-processing of the PWFS signals 'slopesMaps' ou 'fullFrame'
    param['pwfs_pupils_shift_x'   ] = [0]*4
    param['pwfs_pupils_shift_y'   ] = [0]*4
    param['rooftop']                = 0 
    param['n_pix_separation'      ] = 40                                            # separation ratio between the PWFS pupils
    param['n_pix_edge'            ] = 20                                            # separation ratio between the PWFS pupils
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


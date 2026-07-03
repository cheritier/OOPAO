# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:12:28 2024

@author: cheritier
"""
from OOPAO.tools.tools import createFolder

def initializeParameterFile():
    # initialize the dictionaries
    param = dict()
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['r0'                   ] = 0.1                                            # value of r0 in the visibile in [m]
    param['L0'                   ] = 30                                             # value of L0 in the visibile in [m]
    param['fractionnalR0'        ] = [0.45,0.1,0.1,0.25,0.1]                        # Cn2 profile
    param['windSpeed'            ] = [5,4,8,10,2]                                   # wind speed of the different layers in [m.s-1]
    param['windDirection'        ] = [0,72,144,216,288]                             # wind direction of the different layers in [degrees]
    param['altitude'             ] = [0, 1000,5000,10000,12000 ]                    # altitude of the different layers in [m]
                              
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['nSubaperture'         ] = 36                                                                                 # number of PWFS subaperture along the telescope diameter
    param['nExtraSubaperture'    ] = 0                                                                                  # extra subaperture on the edges
    param['diameter'             ] = 0.6                                                                               # diameter in [m]
    param['ratio'                ] = 1                                                                                  # ratio factor for binned case
    param['nPixelPerSubap'       ] = 1                                                                                  # sampling of the PWFS subapertures in pix
    param['resolution'           ] = param['nSubaperture']*param['nPixelPerSubap']                                    # resolution of the telescope driven by the PWFS
    param['sizeSubaperture'      ] = param['diameter']/param['nSubaperture']                                          # size of a sub-aperture projected in the M1 space
    param['samplingTime'         ] = 1/1000                                                                             # loop sampling time in [s]
    param['m1_reflectivity'      ] = 1                                                                                  # reflectivity of the pupil
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['magnitude'            ] = -0.04                                       # magnitude of the guide star
    param['opticalBand'          ] = 'J2'                                        # optical band of the guide star J2 = 1550 nm
    param['opticalBandCalib'     ] = 'R'                                         # optical band of calibration laser R = 640 nm
    
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nActuator'            ] = 11                                          # number of actuators 
    param['mechanicalCoupling'   ] = 0.36                                        # Mechanical coupling of the DM influence functions
    param['dm_coordinates'       ] = None                                        # tag for the deformable mirror class
    param['dm_inf_funct_factor'  ] = -1                                           # factor to account for the influence functions deformation in DM units
    param['dm_inf_funct_location'] =  'C:/Users/cheritier/Documents/RAMA/IF_97.npy' # files available here: https://drive.google.com/drive/folders/1hbWyCq_AX1r32JB4jyVfZ-Ss3bVuRInE
    param['dm_pitch'             ] = 1.5e-3 *param['diameter']/13.5e-3                                         # factor to account for the influence functions deformation in DM units
    param['dm_flip_lr'           ] = False
    param['dm_flip_ud'           ] = True
    # mis-registrations                                                             
    param['rotationAngle'        ] = 0                                           # rotation angle of the DM in [degrees]    
    param['shiftX'               ] = 0                                           # shift X of the DM in pixel size units ( tel.D/tel.resolution ) 
    param['shiftY'               ] = - param['dm_pitch']/3  # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
    param['anamorphosisAngle'    ] = 0                                           # anamorphosis angle of the DM in [degrees]
    param['tangentialScaling'    ] = 0.12                                        # tangential scaling in percentage of diameter
    param['radialScaling'        ] = 0.12                                        # radial scaling in percentage of diameter

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['size_quadrant_ramatwin'] = 60
    param['modulation'            ] = 0                                             # modulation radius in ratio of wavelength over telescope diameter
    param['n_pix_separation'      ] = 16                                             # separation between the PWFS pupils in pix
    param['n_pix_edge'            ] = param['size_quadrant_ramatwin']-param['n_pix_separation']//2 -   param['nSubaperture'] - param['nExtraSubaperture']                                            # separation between the PWFS pupils in pix
    param['psfCentering'          ] = False                                         # centering of the FFT and of the PWFS mask on the 4 central pixels
    param['lightThreshold'        ] = 0.1                                           # light threshold to select the valid pixels
    param['postProcessing'        ] = 'fullFrame_sum_flux'                          # post-processing of the PWFS signals 'slopesMaps' ou 'fullFrame'
    param['pwfs_pupils_shift_x'   ] = [0]*4                                         # shift X of the PWFS pupils on the detector
    param['pwfs_pupils_shift_y'   ] = [0]*4                                         # shift Y of the PWFS pupils on the detector
    param['pwfs_rooftop'          ] = 1                                             # size of the PWFS "rooftop" in lambda/D

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # name of the system
    param['name'] = 'RAMA_' +  param['opticalBand'] +'_band_'+ str(param['nSubaperture'])+'x'+ str(param['nSubaperture'])  
    
    # location of the calibration data
    param['path_data'] = 'C:/Users/cheritier/Documents/RAMA/' 

    print('Reading/Writting calibration data from ' + param['path_data'])
    
    return param


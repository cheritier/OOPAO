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
    param['location_data'] = 'C:/Users/cheritier/Documents/oopao_private/ghost/ghost_simulation/'
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['atmosphere_r0'                      ] = 0.15                                           # value of r0 in the visibile in [m]
    param['atmosphere_L0'                      ] = 30                                             # value of L0 in the visibile in [m]
    param['atmosphere_fractionnalR0'           ] = [0.15,0.2,0.05,0.05,0.1,0.1,0.05,0.15,0.1,0.05]                                            # Cn2 profile
    N = 10
    # create a random state
    randomState    = RandomState(42)
    # SLOW
    param['atmosphere_windSpeed'               ] = list(randomState.randint(10,15,N))                                            # wind speed of the different layers in [m.s-1]
    param['atmosphere_windDirection'           ] = list(randomState.randint(0,360,N))                                             # wind direction of the different layers in [degrees]
    param['atmosphere_altitude'                ] = list(randomState.randint(1,10000,N))  
                              
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['telescope_diameter'                ] = 1.82                                             # diameter in [mm]
    param['wfs_n_subaperture'                 ] = 42                                             # number of PWFS subaperture along the telescope diameter
    param['wfs_n_pixel_per_subaperture'       ] = 1                                            # sampling of the PWFS subapertures
    param['wfs_n_extra_subaperture'           ] = 8                                            # extra subaperture to pad the pupils (even number required)   
    param['telescope_sampling_time'           ] = 1/1000                                         # loop sampling time in [s]
    param['telescope_central_obstruction'     ] = 0                                              # central obstruction in percentage of the diameter
    param['telescope_m1_reflectivity'         ] = 1                                   # reflectivity of the 798 segments
          
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['source_magnitude'               ] = 6.16                                              # magnitude of the guide star
    param['source_optical_band'            ] = 'R'                                            # optical band of the guide star
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['dm_n_actuator'              ] = 24                                         # number of actuators 
    param['dm_mechanical_coupling'     ] = 0.35
    param['isM4'                       ] = False                                           # tag for the deformable mirror class
    param['isLBT'                      ] = False

    # ---------------------------
    # mis-registrations                                                             
    param['shiftX'                  ] = 0                                              # shift X of the DM in pixel size units ( tel.D/tel.resolution ) 
    param['shiftY'                  ] = 0                                              # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
    param['rotationAngle'           ] = 0                                              # rotation angle of the DM in [degrees]
    param['anamorphosisAngle'       ] = 0                                              # anamorphosis angle of the DM in [degrees]
    param['radialScaling'           ] = 0                                            # radial scaling in percentage of diameter
    param['tangentialScaling'       ] = 0                                            # tangential scaling in percentage of diameter
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['pyramid_modulation'              ] = 3                                             # modulation radius in ratio of wavelength over telescope diameter
    param['pyramid_n_pix_separation'        ] = 4                                             # separation ratio between the PWFS pupils
    param['pyramid_n_pix_edge'              ] = 2
    param['pyramid_psf_centering'           ] = True                                         # centering of the FFT and of the PWFS mask on the 4 central pixels
    param['pyramid_light_threshold'         ] = 0.1                                           # light threshold to select the valid pixels
    param['pyramid_post_processing'         ] = 'fullFrames_incident_flux'                                   # post-processing of the PWFS signals 
    param['pyramid_sx'              ] = [0,0,0,0]                                   # X-shift for each PWFS pupil 
    param['pyramid_sy'              ] = [0,0,0,0]                                   # Y-shift for each PWFS pupil 
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # name of the system
    param['name'] = 'ekarus' 
    
    # location of the calibration data
    param['pathInput'            ] = 'data_calibration/' 
    
    # location of the output data
    param['pathOutput'            ] = '/diskb/cheritier/psim/data_cl'
    
    
    print('Reading/Writting calibration data from ' + param['pathInput'])
    print('Writting output data in ' + param['pathOutput'])
    createFolder(param['pathInput'])
    createFolder(param['pathOutput'])
        
    return param




def get_imat_ghost(path,wfs,M2C=None,normalization_factor = -0.35*1e6 ):
    
    # load zonal interaction matrix
    imat                                = np.load(path)
    #project on the desired M2C
    if M2C is not None:
        imat                            = imat@M2C
    
    # re-order the imat signals to match OOPAO ordering
    imat_ordered                                = imat.copy()
    imat_ordered[:wfs.nSignal//2,:]             = imat[wfs.nSignal//2:wfs.nSignal,:]
    imat_ordered[wfs.nSignal//2:wfs.nSignal,:]  = imat[:wfs.nSignal//2,:]
    
    return imat_ordered*normalization_factor




def compute_imat_4_ghost(imat, wfs, normalization_factor = -0.35*1e6):
    
    # fill an imat according to the ghost ordering
    imat_4_ghost_ordered                                = np.zeros(imat.shape)
    imat_4_ghost_ordered[:wfs.nSignal//2,:]             = imat[wfs.nSignal//2:wfs.nSignal,:]
    imat_4_ghost_ordered[wfs.nSignal//2:wfs.nSignal,:]  = imat[:wfs.nSignal//2,:]
    
    return imat_4_ghost_ordered/normalization_factor

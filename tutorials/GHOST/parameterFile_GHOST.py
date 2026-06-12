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
    
    param['telescope_diameter'                ] = 6.9                                             # diameter in [mm]
    param['wfs_n_subaperture'                 ] = 36                                             # number of PWFS subaperture along the telescope diameter
    param['wfs_n_pixel_per_subaperture'       ] = 1                                            # sampling of the PWFS subapertures
    param['wfs_n_extra_subaperture'           ] = 6                                            # extra subaperture to pad the pupils (even number required)   
    param['telescope_sampling_time'           ] = 1/1000                                         # loop sampling time in [s]
    param['telescope_central_obstruction'     ] = 0                                              # central obstruction in percentage of the diameter
    param['telescope_m1_reflectivity'         ] = 1                                   # reflectivity of the 798 segments
          
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['source_magnitude'               ] = 6.16                                              # magnitude of the guide star
    param['source_optical_band'            ] = 'I'                                            # optical band of the guide star
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['dm_n_actuator'              ] = 24                                         # number of actuators 
    param['dm_mechanical_coupling'     ] = 0.15
    param['isM4'                       ] = False                                           # tag for the deformable mirror class
    param['isLBT'                      ] = False
    # --------------------------- Specific to GHOST bench
    import scipy.io
    dm_coord            = scipy.io.loadmat(param['location_data']+'dm_coord.mat')
    nAct_ghost          = param['dm_n_actuator']
    diameter_dm_ghost   = 6.9
    coordinates         = np.zeros([492,2])
    coordinates[:,0]    = (dm_coord['x']-(nAct_ghost-1)/2)*param['telescope_diameter']/2/((nAct_ghost-1)/2)
    coordinates[:,1]    = (dm_coord['y']-(nAct_ghost-1)/2)*param['telescope_diameter']/2/((nAct_ghost-1)/2)
    
    param['dm_coordinates'             ] = coordinates                                           # tag for the eformable mirror class
    param['pitch'                   ] = diameter_dm_ghost/nAct_ghost
    # ---------------------------
    # mis-registrations                                                             
    param['shiftX'                  ] = 0                                              # shift X of the DM in pixel size units ( tel.D/tel.resolution ) 
    param['shiftY'                  ] = 0                                              # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
    param['rotationAngle'           ] = 5                                              # rotation angle of the DM in [degrees]
    param['anamorphosisAngle'       ] = 0                                              # anamorphosis angle of the DM in [degrees]
    param['radialScaling'           ] = -0.02                                            # radial scaling in percentage of diameter
    param['tangentialScaling'       ] = -0.02                                            # tangential scaling in percentage of diameter
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['pyramid_modulation'              ] = 3                                             # modulation radius in ratio of wavelength over telescope diameter
    param['pyramid_n_pix_separation'        ] = 6                                             # separation ratio between the PWFS pupils
    param['pyramid_n_pix_edge'              ] = 2
    param['pyramid_psf_centering'           ] = True                                         # centering of the FFT and of the PWFS mask on the 4 central pixels
    param['pyramid_light_threshold'         ] = 0.1                                           # light threshold to select the valid pixels
    param['pyramid_post_processing'         ] = 'slopesMaps'                                   # post-processing of the PWFS signals 
    param['pyramid_sx'              ] = [0,0,0,0]                                   # X-shift for each PWFS pupil 
    param['pyramid_sy'              ] = [0,0,0,0]                                   # Y-shift for each PWFS pupil 

    # --------------------------- Specific to GHOST bench
    param['pyramid_user_valid_signal'] = np.pad(np.load(param['location_data']+'pupil_mask_PYR.npy'),param['wfs_n_extra_subaperture']//2).astype(bool)
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOOP PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['nLoop'                 ] = 5000                                           # number of iteration                             
    param['photonNoise'           ] = True                                         # Photon Noise enable  
    param['readoutNoise'          ] = 0                                            # Readout Noise value
    param['gainCL'                ] = 0.5                                          # integrator gain
    param['nPhotonPerSubaperture' ] = 1000                                         # number of photons per subaperture (update of ngs.magnitude)
    param['getProjector'          ] = True                                         # modal projector too get modal coefficients of the turbulence and residual phase
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # name of the system
    param['name'] = 'ghost' 
    
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

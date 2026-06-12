# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:17:42 2022

@author: cheritie
"""
import numpy as np


def compute_ao_model_pyramid(param, loc='C:/diskb/cheritier/GHOST/'):
    """
    This functions generate a full model of the ghost AO systems

    Parameters
    ----------
    param : Parameter File for the ghost model
    loc : Location of the input data()

    Returns
    -------
    tel :   Telescope Object to be used in OOPAO
    ngs :   Source Object to be used in OOPAO
    dm :    Deformable Mirror Object to be used in OOPAO
    wfs :   Wave-Front Sensor Object to be used in OOPAO
    atm :   Atmosphere Object to be used in OOPAO

    """
    #%% -----------------------     TELESCOPE   ----------------------------------

    from OOPAO.Telescope import Telescope
    # create the Telescope object
    tel = Telescope(resolution          = param['wfs_n_subaperture']*param['wfs_n_pixel_per_subaperture'],
                    diameter            = param['telescope_diameter'],
                    samplingTime        = param['telescope_sampling_time'],
                    centralObstruction  = param['telescope_central_obstruction'])
    
    if param['wfs_n_extra_subaperture'] != 0:
        tel.pad(param['wfs_n_extra_subaperture']*param['wfs_n_pixel_per_subaperture']//2)

    #%% -----------------------     NGS   ----------------------------------
    from OOPAO.Source import Source
    # create the Source object
    ngs=Source(optBand   = param['source_optical_band'],\
               magnitude = param['source_magnitude'])
    
    # combine the NGS to the telescope using '*' operator:
    ngs*tel
    
    #%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
    from OOPAO.DeformableMirror import DeformableMirror

    # if no coordonates specified, create a cartesian dm
    dm = DeformableMirror(telescope     = tel,\
                    nSubap              = param['dm_n_actuator']-1,\
                    mechCoupling        = param['dm_mechanical_coupling'],\
                    coordinates         = param['dm_coordinates'],\
                    misReg              = None,\
                    pitch               = param['pitch'])
    
    param['dm_coordinates'] = dm.coordinates
    param['dm_pitch']          = dm.pitch
    
    from OOPAO.MisRegistration import MisRegistration
    misRegistration_tmp = MisRegistration(param)
    from OOPAO.mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration
    dm = applyMisRegistration(tel = tel, misRegistration_tmp = misRegistration_tmp, dm_input=dm)
    
    #%% -----------------------     PYRAMID WFS   ----------------------------------
    from OOPAO.Pyramid import Pyramid
    
    # create the Pyramid Object
    wfs = Pyramid(nSubap            = param['wfs_n_subaperture'] + param['wfs_n_extra_subaperture'],\
              telescope             = tel,\
              modulation            = param['pyramid_modulation'],\
              lightRatio            = param['pyramid_light_threshold'],\
              n_pix_separation      = param['pyramid_n_pix_separation'],\
              n_pix_edge            = param['pyramid_n_pix_edge'],\
              psfCentering          = param['pyramid_psf_centering'],\
              postProcessing        = param['pyramid_post_processing'],\
              userValidSignal       = param['pyramid_user_valid_signal'])
    # shift the pyramid pupils
    wfs.apply_shift_wfs(sx = param['pyramid_sx'],
                        sy = param['pyramid_sy'])
        
    #%% -----------------------     ATMOSPHERE   ----------------------------------
    from OOPAO.Atmosphere import Atmosphere
    
    atm = Atmosphere(telescope      = tel,\
                     r0             = param['atmosphere_r0'],\
                     L0             = param['atmosphere_L0'],\
                     windSpeed      = param['atmosphere_windSpeed'],\
                     fractionalR0   = param['atmosphere_fractionnalR0'],\
                     windDirection  = param['atmosphere_windDirection'],\
                     altitude       = param['atmosphere_altitude'])
    
    return tel,ngs,dm,wfs,atm
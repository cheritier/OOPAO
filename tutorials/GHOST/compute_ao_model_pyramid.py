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
    tel = Telescope(resolution          = param['n_subaperture']*param['n_pixel_per_subaperture'],
                    diameter            = param['diameter'],
                    samplingTime        = param['sampling_time'],
                    centralObstruction  = param['central_obstruction'])
    
    if param['n_extra_subaperture'] != 0:
        tel.pad(param['n_extra_subaperture']*param['n_pixel_per_subaperture']//2)

    #%% -----------------------     NGS   ----------------------------------
    from OOPAO.Source import Source
    # create the Source object
    ngs=Source(optBand   = param['optical_band'],\
               magnitude = param['magnitude'])
    
    # combine the NGS to the telescope using '*' operator:
    ngs*tel
    
    #%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
    from OOPAO.DeformableMirror import DeformableMirror

    # if no coordonates specified, create a cartesian dm
    dm = DeformableMirror(telescope     = tel,\
                    nSubap              = param['n_actuator']-1,\
                    mechCoupling        = param['mechanical_coupling'],\
                    coordinates         = param['coordinates'],\
                    misReg              = None,\
                    pitch               = param['pitch'])
    
    param['dm_coordinates'] = dm.coordinates
    param['pitch']          = dm.pitch
    
    from OOPAO.MisRegistration import MisRegistration
    misRegistration_tmp = MisRegistration(param)
    from OOPAO.mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration
    dm = applyMisRegistration(tel = tel, misRegistration_tmp = misRegistration_tmp, param = param)
    
    #%% -----------------------     PYRAMID WFS   ----------------------------------
    from OOPAO.Pyramid import Pyramid
    
    # create the Pyramid Object
    wfs = Pyramid(nSubap            = param['n_subaperture'] + param['n_extra_subaperture'],\
              telescope             = tel,\
              modulation            = param['modulation'],\
              lightRatio            = param['light_threshold'],\
              n_pix_separation      = param['n_pix_separation'],\
              n_pix_edge            = param['n_pix_edge'],\
              psfCentering          = param['psf_centering'],\
              postProcessing        = param['post_processing'],\
              userValidSignal       = param['user_valid_signal'])
        
    #%% -----------------------     ATMOSPHERE   ----------------------------------
    from OOPAO.Atmosphere import Atmosphere
    
    atm = Atmosphere(telescope      = tel,\
                     r0             = param['r0'],\
                     L0             = param['L0'],\
                     windSpeed      = param['windSpeed'],\
                     fractionalR0   = param['fractionnalR0'],\
                     windDirection  = param['windDirection'],\
                     altitude       = param['altitude'])
    
    return tel,ngs,dm,wfs,atm
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 10:40:42 2023

@author: cheritie - astriffl
"""
import numpy as np
from PIL import Image
from astropy.io import fits
import copy
import matplotlib.pyplot as plt

def compute_ekarus_model(param,loc,source,IFreal=False):
    """
    This functions generate a full model of the papyrus AO systems

    Parameters
    ----------
    param :  Parameter File for the papyrus model
    loc :    Location of the input data()
    source : True -> T152 pupil / False -> calibration pupil
    IFreal : DEPRECATED True -> ALPAO DM real IF / False -> OOPAO gaussian IF 
        
    Returns
    -------
    tel :    Telescope Object to be used in OOPAO
    ngs :    Source Object to be used in OOPAO
    src :    Source Object to be used in OOPAO
    dm :     Deformable Mirror Object to be used in OOPAO
    wfs :    Wave-Front Sensor Object to be used in OOPAO
    atm :    Atmosphere Object to be used in OOPAO

    """
    #%% -----------------------     TELESCOPE   ----------------------------------
    from OOPAO.Telescope import Telescope
    #inherit from PAPYRUS where the Pupil on-sky size is different from the calibration pupil   
    n_extra_pix = param['nExtraSubaperture'] * param['nPixelPerSubap']
    print(n_extra_pix)
    
    # Telescope Object for the calibration pupil (no spider/no central obstr)
    tel = Telescope(resolution          = param['resolution'],
                    diameter            = param['diameter'],
                    samplingTime        = param['samplingTime'],
                    centralObstruction  = 0,
                    fov                 = 0)
    
    tel.pupil_calib = tel.pupil.copy()
    if n_extra_pix>0:
        tel.pad(n_extra_pix//2)

    # Telescope Object for the sky pupil with central obstruction without spider
    tel_sky = Telescope(resolution      = param['resolution'],
                    diameter            = param['diameter'],
                    samplingTime        = param['samplingTime'],
                    centralObstruction  = 0.34,
                    fov                 = 0)
    
    tel.pupil_sky = tel_sky.pupil.copy()
   
    # Apply the spider
    offset = 20 # offset in degrees for the 4 spiders
    tel_sky.apply_spiders([offset,offset+90,offset+180,offset+270], thickness_spider = 0.015)
    tel.pupil_sky_spiders = tel_sky.pupil.copy()    
    tel.n_extra_pix = n_extra_pix
    #%% -----------------------     NGS   ----------------------------------
    from OOPAO.Source import Source
    # create the Source object
    ngs=Source(optBand   = param['opticalBandCalib'],\
               magnitude = param['magnitude'])

    src=Source(optBand   = param['opticalBand'],\
                   magnitude = param['magnitude'])
            
    # combine the NGS to the telescope using '*' operator:
    ngs*tel

    #%% -----------------------     ATMOSPHERE   ----------------------------------
    from OOPAO.Atmosphere import Atmosphere
    
    atm=Atmosphere(telescope     = tel,\
                   r0            = param['r0'],\
                   L0            = param['L0'],\
                   windSpeed     = param['windSpeed'],\
                   fractionalR0  = param['fractionnalR0'],\
                   windDirection = param['windDirection'],\
                   altitude      = param['altitude'])
        
    #%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
    
    from OOPAO.DeformableMirror import DeformableMirror, MisRegistration
    from OOPAO.InfluenceFunctions import InfluenceFunctions
    diameter = tel.initial_D #to be fine tuned
    resolution = tel.resolution
    name_system = 'EKARUS_DM468'
    
    IF = InfluenceFunctions(name_system=name_system,
                            diameter=diameter,
                            resolution=resolution,
                            specific_parameters = None,
                            loc = param['dm_inf_funct_location'],
                            mis_registration=MisRegistration(param),
                            flip_lr=param['flip_lr'],
                            flip_ud=param['flip_ud'],
                            sign = param['dm_inf_funct_factor']
                            )

    dm=DeformableMirror(telescope    = tel,\
                        nSubap       = param['nActuator']-1,\
                        mechCoupling = param['mechanicalCoupling'],\
                        misReg       = MisRegistration(param), \
                        pitch        = param['dm_pitch'],\
                        modes        = IF)    
    # synthetic DM in case
    coordinates = np.load( param['dm_inf_funct_location'] + 'coord_dm468_theoretical.npy')
    coordinates /=coordinates.max()
    coordinates *=tel.initial_D/2 *1.15
    dm_synthetic = DeformableMirror(telescope = tel,
                           nSubap=param['nActuator']-1,
                           misReg=MisRegistration(param),
                           coordinates=coordinates,
                           mechCoupling=param['mechanicalCoupling'] ,
                           flip_lr=True,
                           flip=param['flip_ud'],
                           sign =  param['dm_inf_funct_factor'])
        
    #%% -----------------------     Tip/Tilt MIRROR   ----------------------------------
    from OOPAO.Zernike import Zernike
    
    Z_TT = Zernike(tel,2)
    Z_TT.computeZernike(tel)
    
    tt = DeformableMirror(telescope    = tel,\
                        nSubap       = 2,\
                        mechCoupling = param['mechanicalCoupling'],\
                        modes        = Z_TT.modesFullRes.reshape(tel.resolution**2,2))    
    #%% -----------------------     PYRAMID WFS   ----------------------------------
    from OOPAO.Pyramid import Pyramid
    
    wfs = Pyramid(nSubap            = (param['nSubaperture']+param['nExtraSubaperture'])//param['ratio'],
              telescope             = tel,
              modulation            = param['modulation'],
              lightRatio            = 0,
              n_pix_separation      = param['n_pix_separation'],
              n_pix_edge            = param['n_pix_edge'],
              psfCentering          = True,
              postProcessing        = param['postProcessing'],
              userValidSignal       = None,
              user_modulation_path  = None)

    # latest values read from parameter file
    wfs.apply_shift_wfs(sx= param['pwfs_pupils_shift_x'],sy= param['pwfs_pupils_shift_y'],units='pixels')
    
    from parameter_files.OCAM2K  import OCAM_param
    from OOPAO.Detector import Detector
    
    # perfect OCAM (No Noise)
    perfect_OCAM = Detector(nRes            = wfs.cam.resolution,
                    integrationTime = tel.samplingTime,
                    bits            = None,
                    FWC             = None,
                    gain            = 1,
                    sensor          = OCAM_param['sensor'],
                    QE              = 1,
                    binning         = 1,
                    psf_sampling    = wfs.zeroPaddingFactor,
                    darkCurrent     = 0,
                    readoutNoise    = 0,
                    photonNoise     = False)
    
    OCAM = Detector(nRes            = wfs.cam.resolution,
                    integrationTime = tel.samplingTime,
                    bits            = OCAM_param['quantization'],
                    FWC             = OCAM_param['FWC'],
                    gain            = 1,
                    sensor          = OCAM_param['sensor'],
                    QE              = OCAM_param['QE'],
                    binning         = 1,
                    psf_sampling    = wfs.zeroPaddingFactor,
                    darkCurrent     = OCAM_param['darkCurrent'],
                    readoutNoise    = OCAM_param['readoutNoise'],
                    photonNoise     = OCAM_param['photonNoise'])
    # OCAM.output_precision = np.uint16
    # wfs.cam = perfect_OCAM
    return tel,ngs,src,dm,dm_synthetic,wfs,atm,tt, perfect_OCAM,OCAM


def centroid( image, threshold=0.01):
    if np.ndim(image) <= 2:
        im = np.reshape(image.copy(), (1, np.shape(image)[0], np.shape(image)[1]))
    else:
        im = np.atleast_3d(image.copy())
    im[im < (threshold*im.max())] = 0
    centroid_out = np.zeros([im.shape[0], 2])
    X_map, Y_map = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[2]))
    X_coord_map = np.atleast_3d(X_map).T
    Y_coord_map = np.atleast_3d(Y_map).T
    norma = np.sum(np.sum(im, axis=1), axis=1)
    centroid_out[:, 0] = np.sum(np.sum(im*X_coord_map, axis=1), axis=1)/norma
    centroid_out[:, 1] = np.sum(np.sum(im*Y_coord_map, axis=1), axis=1)/norma
    return centroid_out
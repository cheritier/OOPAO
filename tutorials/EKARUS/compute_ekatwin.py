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
    from OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube
    
    # mis-registrations object
    misReg = MisRegistration(param)
    D_T152 = 1.82    
    pitch  = 1.82/param['nActuator'] # projected on sky
    
    x = np.linspace(-D_T152/2, D_T152/2, param['nActuator'])
    [X,Y] = np.meshgrid(x,x)
    
    DM_coordinates = np.asarray([X.reshape(param['nActuator']**2),Y.reshape(param['nActuator']**2)]).T
    dist           = np.sqrt(DM_coordinates[:,0]**2 + DM_coordinates[:,1]**2)
    DM_coordinates = DM_coordinates[dist <= D_T152/2 + 1.5*pitch/2, :]
    
    param['dm_coordinates'] = DM_coordinates
    param['pitch']          = pitch
    
    dm=DeformableMirror(telescope    = tel,\
                        nSubap       = param['nActuator']-1,\
                        mechCoupling = param['mechanicalCoupling'],\
                        misReg       = misReg, \
                        coordinates  = DM_coordinates,\
                        pitch        = pitch,\
                        modes        = None,
                        flip_lr      = True,
                        sign         = -1/param['dm_inf_funct_factor'  ])
        
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
              n_pix_separation      = 4//param['ratio'],
              n_pix_edge            = 2//param['ratio'],
              psfCentering          = True,
              postProcessing        = 'fullFrame_sum_flux',
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
    OCAM.output_precision = np.uint16
    wfs.cam = perfect_OCAM
    return tel,ngs,src,dm,wfs,atm,tt, perfect_OCAM,OCAM
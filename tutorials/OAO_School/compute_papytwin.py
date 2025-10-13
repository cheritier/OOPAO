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

def compute_papyrus_model(param,loc,source,IFreal=False):
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
    dm :     Deformable Mirror Object to be used in OOPAO
    wfs :    Wave-Front Sensor Object to be used in OOPAO
    atm :    Atmosphere Object to be used in OOPAO

    """
    #%% -----------------------     TELESCOPE   ----------------------------------
            
    from OOPAO.Telescope import Telescope
    T152onDM_size       = 35.5 # mm
    PapyrusOnDM_size    = 37.5 # mm 
    ratio_sky_calib = T152onDM_size/PapyrusOnDM_size

    
    # create a temporary Telescope object
    tel_calib = Telescope(resolution    = int(np.round(param['nSubaperture']*param['nPixelPerSubap'])),
                    diameter            = param['diameter']/ratio_sky_calib,
                    samplingTime        = param['samplingTime'],
                    centralObstruction  = 0,
                    fov                 = 0)
    
    n_extra_pix = (param['resolution']-tel_calib.resolution)//2
    pupil_calib = np.pad(tel_calib.pupil,[n_extra_pix,n_extra_pix])

    # create a temporary Telescope object
    tel_sky = Telescope(resolution      = int(np.round(param['nSubaperture']*param['nPixelPerSubap']*ratio_sky_calib)),
                    diameter            = param['diameter'],
                    samplingTime        = param['samplingTime'],
                    centralObstruction  = 0.3,
                    fov                 = 0)
    
    n_extra_pix = (param['resolution']-tel_sky.resolution)//2
    pupil_sky = tel_sky.pupil   
    
    
    # redefine the pupil padding to accomodate for the calibration or sky mode
    tel = Telescope(resolution          = param['resolution'],
                    diameter            = param['diameter'] * param['resolution']/tel_calib.resolution,
                    samplingTime        = param['samplingTime'],
                    centralObstruction  = 0,
                    fov                 = param['fieldOfView'])    
    
    if source:
        tel.pupil = pupil_calib
    else:
        tel.pupil = pupil_sky
        
    tel.pupil_sky = pupil_sky
    tel.pupil_calib = pupil_calib
    
    #%% -----------------------     NGS   ----------------------------------
    from OOPAO.Source import Source
    # create the Source object

    ngs=Source(optBand   = param['opticalBandCalib'],\
               magnitude = param['magnitude'])
    
    # combine the NGS to the telescope using '*' operator:
    ngs*tel

    src=Source(optBand   = param['opticalBand'],\
               magnitude = param['magnitude'])
    
    # combine the NGS to the telescope using '*' operator:

    #%% -----------------------     ATMOSPHERE   ----------------------------------
    from OOPAO.Atmosphere import Atmosphere
    
    atm=Atmosphere(telescope     = tel,\
                   r0            = param['r0'],\
                   L0            = param['L0'],\
                   windSpeed     = param['windSpeed'],\
                   fractionalR0  = param['fractionnalR0'],\
                   windDirection = param['windDirection'],\
                   altitude      = param['altitude'])
    atm.initializeAtmosphere(tel)
    #%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
    
    from OOPAO.DeformableMirror import DeformableMirror, MisRegistration
    from OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube

    
    # mis-registrations object
    misReg          = MisRegistration(param)
    pitch           = 2.5 #mm
    DM_diag_size    = param['nActuator'] * pitch #mm
    scale_T152DM = DM_diag_size / T152onDM_size
    D_T152 = 1.52
    
    x = np.linspace(-scale_T152DM * D_T152/2, scale_T152DM * D_T152/2, param['nActuator'])
    [X,Y] = np.meshgrid(x,x)
    
    DM_coordinates = np.asarray([X.reshape(17**2),Y.reshape(17**2)]).T
    dist           = np.sqrt(DM_coordinates[:,0]**2 + DM_coordinates[:,1]**2)
    DM_coordinates = DM_coordinates[dist <= D_T152/2 + 2.2 *pitch * D_T152 / T152onDM_size, :]
    DM_pitch       = pitch * D_T152 / T152onDM_size
    
    
    param['dm_coordinates'] = DM_coordinates
    param['pitch']          = pitch
    
    dm=DeformableMirror(telescope    = tel,\
                        nSubap       = param['nActuator']-1,\
                        mechCoupling = param['mechanicalCoupling'],\
                        misReg       = misReg, \
                        coordinates  = DM_coordinates,\
                        pitch        = DM_pitch,\
                        modes        = None,
                        flip_lr      = True,
                        sign         = 1/param['dm_inf_funct_factor'  ])
        
    #%% -----------------------     Tip/Tilt MIRROR   ----------------------------------
    from OOPAO.Zernike import Zernike
    
    Z_TT = Zernike(tel,2)
    Z_TT.computeZernike(tel)
    
    slow_tt = DeformableMirror(telescope    = tel,\
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
              postProcessing        = 'fullFrame_sum_flux',
              userValidSignal       = None,
              user_modulation_path  = None,
              rooftop               = param['rooftop'])

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

    compute_kl_basis = True

    if compute_kl_basis:
        from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
        M2C = compute_KL_basis(tel = tel,
                               atm = atm,
                               dm  = dm,
                               lim = 1e-3)
    return tel,ngs,src,dm,wfs,atm,slow_tt, perfect_OCAM,OCAM,M2C
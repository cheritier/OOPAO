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
from OOPAO.tools.tools import OopaoError

def compute_rama_model(param,loc,source,IFreal=False):
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
    if n_extra_pix !=0:
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
    
    def get_influence_functions_dm97(loc,
                                     diameter,
                                     resolution_out,
                                     pixel_size_out,
                                     mis_registration = None,
                                     flip_lr = False,
                                     flip_ud = False):
        try:
            IF = np.moveaxis(np.load(loc), 2,0)
        except:
            raise OopaoError('Could not find the RAMA data. Make sure you downloaded it from https://nuage.osupytheas.fr/s/YRbHrHSQA9ZSiQP and indicated the correct path in the parameter file')

        # convert to numpy array
        IF = np.asarray(IF)
        n_if,n_px1,n_px2 = np.shape(IF)
        
        if flip_lr:
            IF = np.flip(IF,axis=2)
        if flip_ud:
            IF = np.flip(IF,axis=1)            
        # compute pixel size projected on sky
        pixel_size_input = (diameter / (n_px1)) # 22 extra pixels on the input data?
        # reshape in 2D
        IF = IF.reshape(IF.shape[0],IF.shape[1]*IF.shape[2])

        
        #reshape in 3D for interpolation
        IF = IF.reshape(n_if,n_px1,n_px2)
        if resolution_out != n_px1:
            IF = np.asarray(interpolate_cube(cube_in=IF,
                         pixel_size_in= pixel_size_input,
                         pixel_size_out=pixel_size_out,
                         resolution_out = resolution_out,
                         mis_registration=mis_registration))
            
        coord  =  centroid(IF)
        IF = IF.reshape(n_if,resolution_out*resolution_out).T

        return IF,coord
        
    if_dm97, coord_dm97 = get_influence_functions_dm97(loc = param['dm_inf_funct_location'],
                                                          diameter=tel.initial_D,#to be fine tuned
                                                          resolution_out=tel.resolution,
                                                          pixel_size_out=tel.pixelSize,
                                                          mis_registration=MisRegistration(param),
                                                          flip_lr = param['dm_flip_lr'],
                                                          flip_ud = param['dm_flip_ud'])
    plt.figure(),plt.imshow(np.var(if_dm97,axis=1).reshape(tel.pupil.shape))
    dm=DeformableMirror(telescope    = tel,\
                        nSubap       = param['nActuator']-1,\
                        mechCoupling = param['mechanicalCoupling'],\
                        misReg       = None, \
                        coordinates  = coord_dm97,\
                        pitch        = param['dm_pitch'],\
                        modes        = if_dm97,
                        sign         = param['dm_inf_funct_factor'  ])
          
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
        
    return tel,ngs,src,dm,wfs,atm


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
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
from OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube
from OOPAO.MisRegistration import MisRegistration

def from_im_fit_ellipse(loc, threshold, n):
    """
    This functions generate the modulation coordinates from 
    a snapshot of the modulated diffraction limit PSF
    The type file should .npy, if needed use function
    im2npy
    No need to use this function if modulation is perfectly circular

    Parameters
    ----------
    loc : Location of the input data()
    threshold : threshold to keep only the pixels of interest for 
                fitting 
    n : number of modulation points wanted
    
    Returns
    -------
    x_t : x coordinates of the n modulation points
    y_t : y coordinates of the n modulation points
    a : minor radius of the fitted ellipse 
    b : major radius of the fitted ellipse
    e : eccentricity of the fitted ellipse

    """
    #Import modulation frame and applying treshold
    Im_mod = np.load(loc+'modulation_frame.npy')
    Im_mod = Im_mod/np.max(Im_mod)
    Im_mod[Im_mod <= threshold] = 0
    Im_mod[Im_mod > threshold] = 1
    
    # Ellipse coordinates points
    ellipse = np.array(np.where(Im_mod==1)).T
    
    X = ellipse[:,0:1] 
    Y = ellipse[:,1:]
    
    # Least squares problem ||Ax - b ||^2
    Afit = np.hstack([X**2, 2 * X * Y, Y**2, 2 * X, 2 * Y])
    Bfit = np.ones_like(X)
    A, B, C, D, E = np.linalg.lstsq(Afit, Bfit, rcond=None)[0].squeeze()
    F = -1
    
    num = 2 * (A*E**2 + C*D**2 + F*B**2 - 2*B*D*E - A*C*F)
    den1 = B**2 - A*C
    den2a = np.sqrt((A-C)**2 + 4*B**2) - (A+C)
    den2b = -np.sqrt((A-C)**2 + 4*B**2) - (A+C)
    
    a = np.sqrt(num / (den1 * den2a))
    b = np.sqrt(num / (den1 * den2b))
    phi =  np.arctan(2*B / (A-C))/2 + np.pi / 4
    e = np.sqrt(1 - a**2/b**2)
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    x_t = a * np.cos(phi) * np.cos(t) + b * np.sin(phi) * np.sin(t)
    y_t = a * np.sin(phi) * np.cos(t) - b * np.cos(phi) * np.sin(t)
    
    return x_t, y_t, a, b, e


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
    T152onDM_size = 35.5 #mm
    PapyrusOnDM_size = 37.5 #mm 
    
    if source:
        ratio_sky_calib = 1
        param['centralObstruction'] = 0
        param['diameter'] = param['diameter'] * PapyrusOnDM_size/T152onDM_size
    else:
        ratio_sky_calib = T152onDM_size/PapyrusOnDM_size
        param['centralObstruction'] = 0.3
        
    # create a temporary Telescope object
    tel_ = Telescope(resolution         = int(np.round(param['nSubaperture']*param['nPixelPerSubap']*ratio_sky_calib)),
                    diameter            = param['diameter'],
                    samplingTime        = param['samplingTime'],
                    centralObstruction  = param['centralObstruction'],
                    fov                 = 0)
        
    # redefine the pupil padding to accomodate for the calibration or sky mode
    tel = Telescope(resolution          = param['resolution'],
                    diameter            = param['diameter'] * param['resolution']/tel_.resolution,
                    samplingTime        = param['samplingTime'],
                    centralObstruction  = param['centralObstruction'],
                    fov                 = 0)
    n_extra_pix = (tel.resolution-tel_.resolution)//2
    tel.pupil = np.pad(tel_.pupil,[n_extra_pix,n_extra_pix])
        
    #%% -----------------------     NGS   ----------------------------------
    from OOPAO.Source import Source
    # create the Source object

    ngs=Source(optBand   = param['opticalBandCalib'],\
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
    
    # hardcoded for now
    alpao_unit     = 30*7591.024876
    
    param['dm_coordinates'] = DM_coordinates
    param['pitch']          = DM_pitch
    
    dm=DeformableMirror(telescope    = tel,\
                        nSubap       = 16,\
                        mechCoupling = param['mechanicalCoupling'],\
                        misReg       = misReg, \
                        coordinates  = DM_coordinates,\
                        pitch        = DM_pitch,\
                        modes        = None,
                        flip_lr      = True,
                        sign         = -1/alpao_unit)
    
    #%% -----------------------     PYRAMID WFS   ----------------------------------
    from OOPAO.Pyramid import Pyramid
    
    wfs = Pyramid(nSubap            = (param['nSubaperture']+param['nExtraSubaperture'])//param['ratio'],
              telescope             = tel,
              modulation            = param['modulation'],
              lightRatio            = 0,
              n_pix_separation      = 40//param['ratio'],
              n_pix_edge            = 20//param['ratio'],
              psfCentering          = True,
              postProcessing        = 'fullFrame_sum_flux',
              userValidSignal       = None,
              user_modulation_path  = None)

    # latest values read from parameter file
    wfs.apply_shift_wfs(sx= param['pwfs_pupils_shift_x'],sy= param['pwfs_pupils_shift_y'],units='pixels')
    
    from parameter_files.OCAM2K  import OCAM_param
    from OOPAO.Detector import Detector
    # perfect OCAM (No Noise)
    OCAM = Detector(nRes            = wfs.cam.resolution,
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
    wfs.cam = OCAM
   
    return tel,ngs,dm,wfs,atm




def check_pwfs_pupils(wfs,valid_pixel_map,n_it=3, correct = False):
    
    wfs.modulation = 20
    
    from OOPAO.tools.tools import centroid
    
    plt.close('all')
    xs = wfs.sx
    ys = wfs.sy
    if correct is False:
        for i in range(4):
            I = wfs.grabFullQuadrant(i+1,valid_pixel_map)
            xc = I.shape[0]//2
            
            [x,y] = np.asarray(centroid(I,threshold=0.3))
            
            I_ = np.abs(wfs.grabFullQuadrant(i+1))
            I_ /= I_.max()
            
            [x_,y_] = np.asarray(centroid(I_,threshold=0.3))
    
            
            plt.figure(1)
            plt.subplot(2,2,i+1)
            plt.imshow(I-I_)
            plt.plot(x,y,'+',markersize = 20)
            plt.plot(x_,y_,'+',markersize = 20)
            plt.axis('off')
            xs[i] += ((x-x_))
            ys[i] += ((y_-y))
            plt.title('PAPYRUS/PAPYTWIN Q'+str(i))
            plt.draw()

    else:
        for i_it in range(n_it):
            wfs.apply_shift_wfs(sx =xs,sy = ys)
        
            for i in range(4):
                I = wfs.grabFullQuadrant(i+1,valid_pixel_map)
                xc = I.shape[0]//2
                
                [x,y] = np.asarray(centroid(I,threshold=0.3))
                
                I_ = np.abs(wfs.grabFullQuadrant(i+1))
                I_ /= I_.max()
                
                [x_,y_] = np.asarray(centroid(I_,threshold=0.3))
        
                
                plt.figure(1)
                plt.subplot(n_it,4,4*i_it + i+1)
                plt.imshow(I-I_)
                plt.plot(x,y,'+',markersize = 20)
                plt.plot(x_,y_,'+',markersize = 20)
                plt.title('Step '+str(i_it)+' -- ['+str(np.round(x-x_,1))+','+str(np.round(y-y_,1))+']')
                plt.axis('off')
                xs[i] += ((x-x_))
                ys[i] += ((y_-y))
                plt.draw()
                plt.pause(0.2)
            
            
def bin_bench_data(valid_pixel,full_int_mat, ratio):
    if ratio !=1:
        mis_reg = MisRegistration()
        pixel_size_in = 1
        pixel_size_out = ratio
        resolution_out = 240//ratio
        cube_in = np.float32(np.atleast_3d(valid_pixel)).T
        valid_pixel = np.squeeze(interpolate_cube(cube_in, pixel_size_in, pixel_size_out, resolution_out,mis_registration=mis_reg)).astype(int).T
        cube_in = (((full_int_mat.reshape(240,240,full_int_mat.shape[1]))).T).astype(float)
        full_int_mat_ = (interpolate_cube(cube_in, pixel_size_in, pixel_size_out, resolution_out,mis_registration=mis_reg).T).reshape(resolution_out*resolution_out,full_int_mat.shape[1])
        full_int_mat_ *= ratio*ratio
        
    else:
        valid_pixel   = valid_pixel.astype(bool)
        full_int_mat_ = (full_int_mat).astype(float)

    
    return valid_pixel, full_int_mat_



def calibrate_mis_registration(ngs,tel,atm,dm,wfs,param,M2C,input_im, index_modes = np.arange(10,150,10)):
    
    from OOPAO.SPRINT import SPRINT
    from OOPAO.tools.tools import emptyClass
    
    # modal basis considered
    
    basis =  emptyClass()
    basis.modes         = M2C[:,index_modes]
    basis.extra         = 'PAP_full_KL'              # EXTRA NAME TO DISTINGUISH DIFFERENT SENSITIVITY MATRICES, BE CAREFUL WITH THIS!     
    
    obj =  emptyClass()
    obj.ngs     = ngs
    obj.tel     = tel
    obj.atm     = atm
    obj.wfs     = wfs
    obj.dm      = dm
    obj.param   = param
        
    Sprint = SPRINT(obj, basis,dm_input=obj.dm,n_mis_reg=5,recompute_sensitivity=True )
    Sprint.estimate(obj, on_sky_slopes = input_im[:,index_modes],dm_input=dm ,n_iteration=2,n_update_zero_point=2,tolerance=100)
    
    
    from OOPAO.mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration
    dm = applyMisRegistration(tel,
                              misRegistration_tmp = Sprint.mis_registration_out,
                              param=param,
                              dm_input=dm)

    return
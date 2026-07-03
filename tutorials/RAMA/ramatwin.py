# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 10:40:42 2023

Accurate version of RAMA AO System used for reproducing the real system in details.
- 03/07/2026: First version of the numerical twin

@author: cheritier
"""
import time
import matplotlib.pyplot as plt
import numpy as np
from Rama import Rama
from OOPAO.tools.displayTools import displayMap, cl_plot
#%% Compute the OOPAO Objects

Ramatwin = Rama()
# telescope object
tel     = Ramatwin.tel
# source object VIS
ngs     = Ramatwin.ngs
# source object IR
src     = Ramatwin.src
# deformable mirror object
dm      = Ramatwin.dm
# Pyramid WFS object
wfs     = Ramatwin.wfs
# atmosphere object
atm     = Ramatwin.atm
# parameter file
param   = Ramatwin.param

dm.display_dm()



#%% load data from the bench

# get access to the data 
from astropy.io import fits
from OOPAO.tools.tools import OopaoError

# data available : https://nuage.osupytheas.fr/s/YRbHrHSQA9ZSiQP

# interaction matrix from the bench
IM_rama_full = fits.getdata(param['path_data']+'IMFull.fits')
# valid pixels used for the reconstruction
valid_pix_full = np.load(param['path_data']+'valid_pixel.npy').astype(bool)
# valid pixels used by DAO (larger)
valid_pix_dao_full = np.load(param['path_data']+'valid_pixel_dao.npy').astype(bool)


sub_valid_pixel = valid_pix_dao_full[valid_pix_full]
# mode 2 command matrix
M2C = np.load(param['path_data']+'M2C.npy')


#%% reduce the size of the valid pixel
from rama_tools import compress_rama_data, compute_rama_frame

valid_pix = compress_rama_data(valid_pix_full, n_pix= param['size_quadrant_ramatwin'])
valid_pix_dao = compress_rama_data(valid_pix_dao_full, n_pix= param['size_quadrant_ramatwin'])

plt.figure()
plt.subplot(1,3,1)
plt.imshow(valid_pix_dao)
plt.subplot(1,3,2)
plt.imshow(valid_pix)
plt.subplot(1,3,3)
plt.imshow(valid_pix_dao_full.astype(float) - valid_pix_full.astype(float))


#%% optimize PWFS pupil position based on the valid pixels provided by DAO
    
Ramatwin.check_pwfs_pupils(valid_pixel_map = valid_pix,correct=True)



#%% better alternative, do it based on the variance map of the iMat

# compute the equivalent rama frame
imat_variance_map = compute_rama_frame(signal=np.var(IM_rama_full,axis=1),valid_signal = valid_pix_dao)

# threshold it
imat_variance_map = imat_variance_map>0.005*imat_variance_map.max()

plt.figure()
plt.imshow(imat_variance_map)

#% optimize PWFS pupil position based on the valid pixels provided by DAO

Ramatwin.check_pwfs_pupils(valid_pixel_map = imat_variance_map,correct=True)


#%% illustrate a few modes

plt.close('all')
ngs**tel*wfs
wfs.modulation = 0   
delta = 10
for ind in range(3):
    
    dm.coefs = M2C[:,delta+ind]*1e-9
    ngs**tel*dm*wfs
    
    
    plt.figure(1)
    plt.subplot(3,3,1+ind*3)
    
    plt.imshow(dm.OPD)
    
    plt.subplot(3,3,2+ind*3)
    plt.imshow(wfs.signal_2D*valid_pix_dao)
    
    plt.subplot(3,3,3+ind*3)

    plt.imshow(compute_rama_frame(signal=IM_rama_full[:,delta+ind],valid_signal = valid_pix_dao)*valid_pix_dao)




#%% PAPYRUS KL Basis Computation (only for the bench)
compute_kl_basis = True

if compute_kl_basis:
    from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
    M2C = compute_KL_basis(tel = tel,
                           atm = atm,
                           dm  = dm,
                           lim = 1e-4)


# apply a few modes for display:  
dm.coefs = M2C[:,:10] * 1e-9
ngs*tel*dm
displayMap(ngs.phase)

#%% Ramatwin Full Interaction Matrix Computation
from OOPAO.calibration.InteractionMatrix import InteractionMatrix

# change modulation for calibration?
ngs**tel*wfs
wfs.modulation = 1
stroke = 1e-9

calib_dm_97 = InteractionMatrix(ngs       = ngs,
                            atm            = atm,
                            tel            = tel,
                            dm             = dm,
                            wfs            = wfs,
                            M2C            = M2C,
                            stroke         = stroke,
                            phaseOffset    = 0,\
                            nMeasurements  = 4,\
                            noise          = 'off',
                            print_time     = False,
                            display        = True)
    
    
wfs.modulation = 0

    
#%%  -----------------------     Close loop  ----------------------------------
from OOPAO.calibration.CalibrationVault import CalibrationVault

# truncate to a given number of modes before the inversion
end_mode    = 83 

# closed loop data
M2C_CL      = M2C[:,:end_mode]  # modes-to-command matrix used in closed loop
calib_CL    = CalibrationVault(calib_dm_97.D[:,:end_mode])  # calibration object with reconstructor (calib_CL.M) for the number of modes selected

# compute the reconstructor (wfs signal - to - dm zonal commands )
reconstructor = M2C_CL@calib_CL.M

plt.figure()
plt.plot(np.std(calib_CL.D,axis=0),label='Ramatwin')
plt.legend()
plt.xlabel('KL Mode Index')
plt.ylabel('Int. Mat STD')
plt.title('Interaction Matrix Conditionning Number:'+str(calib_CL.cond))
#%% Define instrument and WFS path detectors and reference PSFs for the SR computation
from OOPAO.Detector import Detector
from OOPAO.tools.tools import strehlMeter
# select desired pupil(Calib)

Ramatwin.set_pupil(calibration=True,spiders=False)

# compute KL modal projector truncating the modes by the pupil used
KL_basis_dm = (dm.modes@M2C)*np.tile(tel.pupil.flatten()[:,None],M2C.shape[1])
projector_kl = np.linalg.pinv(KL_basis_dm) 


#%%
# select desired pupil(Sky/Calib)
Ramatwin.set_pupil(calibration=True,spiders=False)

# instrument path
src_cam = Detector(tel.resolution*2)
src_cam.psf_sampling = 4  # sampling of the PSF
src_cam.integrationTime = tel.samplingTime # exposure time for the PSF

# WFS path
ngs_cam = Detector(tel.resolution*2)
ngs_cam.psf_sampling = 4
ngs_cam.integrationTime = tel.samplingTime

# initialize the Strehl Ratio computation
tel.resetOPD()

ngs*tel*ngs_cam
ngs_psf_ref = ngs_cam.frame.copy()

src*tel*src_cam

src_psf_ref = src_cam.frame.copy()

#%%


atm.initializeAtmosphere(tel)


# initialize Telescope DM commands
tel.resetOPD()
dm.coefs=0
ngs*tel*dm*wfs
wfs*wfs.focal_plane_camera

plt.close('all')
atm.r0 = 0.05
# To make sure to always replay the same turbulence, generate a new phase screen for the atmosphere and combine it with the Telescope
atm.generateNewPhaseScreen(seed=10)

# combine telescope with atmosphere
tel+atm

# propagate both sources
atm*ngs*tel*ngs_cam
atm*src*tel*src_cam

# loop parameters
nLoop = 500  # number of iterations
gainCL = 0.4  # integrator gain
wfs.cam.photonNoise = True  # enable photon noise on the WFS camera
display = False  # enable the display
frame_delay = 2  # number of frame delay

# variables used to to save closed-loop data data
SR_ngs = np.zeros(nLoop)
SR_src = np.zeros(nLoop)

wfe_atmosphere = np.zeros(nLoop)
wfe_residual_SRC = np.zeros(nLoop)
wfe_residual_NGS = np.zeros(nLoop)
wfsSignal = np.arange(0, wfs.nSignal)*0  # buffer to simulate the loop delay

dm_commands = []
wfs_signals = []
modes_in = []
modes_out = []

# configure the display pannel
plot_obj = cl_plot(list_fig=[atm.OPD,  # list of data for the different subplots
                             tel.OPD,
                             tel.OPD,
                             wfs.cam.frame,
                             wfs.focal_plane_camera.frame,
                             [[0, 0], [0, 0], [0, 0]],
                             np.log10(ngs_cam.frame),
                             np.log10(src_cam.frame)],
                   type_fig=['imshow',  # type of figure for the different subplots
                             'imshow',
                             'imshow',
                             'imshow',
                             'imshow',
                             'plot',
                             'imshow',
                             'imshow'],
                   list_title=['Turbulence [nm]',  # list of title for the different subplots
                               'NGS residual [m]',
                               'SRC residual [m]',
                               'WFS Detector',
                               'WFS Foxal Plane Camera',
                               None,
                               None,
                               None],
                   list_legend=[None,  # list of legend labels for the subplots
                                None,
                                None,
                                None,
                                None,
                                ['SRC@'+str(src.coordinates[0])+'"', 'NGS@'+str(ngs.coordinates[0])+'"'],
                                None,
                                None],
                   list_label=[None,  # list of axis labels for the subplots
                               None,
                               None,
                               None,
                               None,
                               ['Time', 'WFE [nm]'],
                               ['NGS PSF@' + str(ngs.coordinates[0]) + '" -- FOV: ' + str(np.round(ngs_cam.fov_arcsec, 2)) + '"', ''],
                               ['SRC PSF@' + str(src.coordinates[0]) + '" -- FOV: ' + str(np.round(src_cam.fov_arcsec, 2)) + '"', '']],
                   n_subplot=[4, 2],
                   list_display_axis=[None,  # list of the subplot for which axis are displayed
                                      None,
                                      None,
                                      None,
                                      None,
                                      True,
                                      None,
                                      None],
                   list_ratio=[[0.95, 0.95, 0.1],
                               [1, 1, 1, 1]],
                   s=20)  # size of the scatter markers
cam_ = 0
for i in range(nLoop):
    a = time.time()    
        
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    # save the wave-front error of the incoming turbulence within the pupil
    wfe_atmosphere[i] = np.std(tel.OPD[np.where(tel.pupil > 0)])*1e9
    # compute input modes from atmosphere
    modes_in.append(projector_kl@(atm.OPD*tel.pupil).flatten())
    # propagate light from the ngs through the atmosphere, telescope, DM to the WFS and ngs camera if display is enable
    ngs**atm*tel*dm*wfs
    cam_+=wfs.cam.frame
    if display:
        tel*ngs_cam
        NGS_PSF = np.log10(np.abs(ngs_cam.frame))
        # compute Strehl ratio from the PSF image
        SR_ngs[i] = strehlMeter(PSF=ngs_cam.frame, tel=tel, PSF_ref=ngs_psf_ref, display=False)
    else:
        # compute Strehl ratio from phase variance (Maréchal)
        SR_ngs[i] = 100*np.exp(-np.var(ngs.phase[np.where(tel.pupil > 0)]))
    # propagate to the focal plane camera
    wfs*wfs.focal_plane_camera
    # save residuals corresponding to the ngs
    wfe_residual_NGS[i] = np.std(tel.OPD[np.where(tel.pupil > 0)])*1e9

    # save the OPD seen by the ngs
    OPD_NGS = tel.mean_removed_OPD.copy()

    modes_out.append(projector_kl@tel.OPD.flatten())
    # propagate light from the src through the atmosphere, telescope, DM to the src camera if display is enable
    src**atm*tel*dm
    if display:
        tel*src_cam
        SRC_PSF = np.log10(np.abs(src_cam.frame))
        # compute Strehl ratio from the PSF image
        SR_src[i] = strehlMeter(PSF=src_cam.frame, tel=tel, PSF_ref=ngs_psf_ref, display=False)
    else:
        # compute Strehl ratio from phase variance (Maréchal)
        SR_src[i] = 100*np.exp(-np.var(src.phase[np.where(tel.pupil > 0)]))
    # save residuals corresponding to the SRC
    wfe_residual_SRC[i] = np.std(tel.OPD[np.where(tel.pupil > 0)])*1e9
    # save the OPD seen by the src
    OPD_SRC = tel.mean_removed_OPD.copy()
    # save the dm commands
    dm_commands.append(dm.coefs)
    wfs_signals.append(wfs.signal)

    # store the slopes after propagating to the WFS <=> 1 frames delay
    if frame_delay == 1:
        wfsSignal = wfs.signal

    # apply the commands on the DM
    dm.coefs = dm.coefs - gainCL*np.matmul(reconstructor, wfsSignal)

    # store the slopes after computing the commands <=> 2 frames delay
    if frame_delay == 2:
        wfsSignal = wfs.signal
    # print('Elapsed time: ' + str(time.time()-a) + ' s')

    # update displays if required
    if display and i > 1:
        SRC_PSF = np.log10(np.abs(src_cam.frame))
        # update range for PSF images
        plot_obj.list_lim = [None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             [NGS_PSF.max()-4, NGS_PSF.max()],
                             [SRC_PSF.max()-4, SRC_PSF.max()]]
        # update title
        plot_obj.list_title = ['Turbulence '+str(np.round(wfe_atmosphere[i]))+'[nm]',
                               'NGS residual '+str(np.round(wfe_residual_NGS[i]))+'[nm]',
                               'SRC residual '+str(np.round(wfe_residual_SRC[i]))+'[nm]',
                               'WFS Detector',
                               'WFS Focal Plance Camera',
                               None,
                               None,
                               None]

        cl_plot(list_fig=[1e9*atm.OPD,
                          1e9*OPD_NGS,
                          1e9*OPD_SRC,
                          wfs.cam.frame,
                          cam_,
                          [np.arange(i+1), wfe_residual_SRC[:i+1], wfe_residual_NGS[:i+1]],
                          NGS_PSF,
                          SRC_PSF],
                plt_obj=plot_obj)
        plt.pause(0.01)
        if plot_obj.keep_going is False:
            break
    print('-----------------------------------')
    print('Loop'+str(i) + '/' + str(nLoop))
    print('NGS: Strehl ratio [%] : ', np.round(SR_ngs[i],1), ' // WFE [nm] : ', np.round(wfe_residual_NGS[i],2))
    print('SRC: Strehl ratio [%] : ', np.round(SR_src[i],1), ' // WFE [nm] : ', np.round(wfe_residual_SRC[i],2))


    
#%% Closed Loop data analysis

plt.figure()
plt.plot(np.arange(nLoop)*tel.samplingTime, wfe_atmosphere, label='Turbulence')
plt.plot(np.arange(nLoop)*tel.samplingTime, wfe_residual_NGS, label='NGS')
plt.plot(np.arange(nLoop)*tel.samplingTime, wfe_residual_SRC, label='SRC')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('WFE [nm]')

plt.figure()
plt.plot(np.arange(nLoop)*tel.samplingTime, SR_ngs, label='NGS@' + str(np.round(1e9*ngs.wavelength,0)) + ' nm')
plt.plot(np.arange(nLoop)*tel.samplingTime, SR_src, label='SRC@' + str(np.round(1e9*src.wavelength,0)) + ' nm')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('SR [%]')

#% Modal Plot 

# extracted from CL Telemtry
modes_in_dm = np.linalg.pinv(M2C_CL) @ np.asarray(dm_commands).T
modes_out_wfs = calib_dm_97.M @ np.asarray(wfs_signals).T

plt.figure()
plt.loglog(np.std(np.asarray(modes_in).T[:,50:],axis=1),label ='Modal Projection')
plt.loglog(np.std(np.asarray(modes_in_dm)[:,50:],axis=1),'--',label ='From DM Commands')
plt.loglog(np.std(np.asarray(modes_out).T[:,50:],axis=1),label ='Modal Projection')
plt.loglog(np.std(np.asarray(modes_out_wfs)[:,50:],axis=1),'--',label ='From WFS signals')
plt.legend()
plt.xlabel('KL Mode Index')
plt.ylabel('WFE [nm]')
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 10:40:42 2023

Accurate version of PAPYRUS AO System used for reproducing the real system in details.
- 17/06/2025: Update after change of the WFS camera.
- 23/06/2025: Update to prepare the integration with DAO
- 17/09/2026: Update of the PAPYRUS file to match EKARUS config
@author: cheritie
"""
#%%
import time
import matplotlib.pyplot as plt
import numpy as np
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap
from Ekarus import Ekarus
from parameter_files.parameterFile_ekatwin import initializeParameterFile
import os
os.environ["OOPAO_PRECISION"] = "32"
os.environ["OOPAO_BACKEND"] = "cpu"
#%% Compute the OOPAO Objects

param = initializeParameterFile()

Ekatwin = Ekarus(param)
# telescope object
tel     = Ekatwin.tel
# source object VIS
ngs     = Ekatwin.ngs
# source object IR
src     = Ekatwin.src
# deformable mirror object
dm      = Ekatwin.dm

# deformable mirror object
dm_synth      = Ekatwin.dm_synthetic

# Pyramid WFS object
wfs     = Ekatwin.wfs
# atmosphere object
atm     = Ekatwin.atm
# Tip/Tilt mirror object
tt      = Ekatwin.tt
# parameter file
param   = Ekatwin.param

dm.display_dm()


#%%
from Ekarus import compress_ekarus_data
n_pix = 114
n_crop = n_pix - wfs.cam.resolution//2

val_pix = np.load('C:/Users/cheritier/Downloads/valid_pix_map.npy')
imat_bench_full = np.load('C:/Users/cheritier/Downloads/IM_rMod_5.0_nModes400_maxAmp0.010_modal_scaling.npy').reshape(240,240,400)
imat_bench_2D = compress_ekarus_data(imat_bench_full,n_pix,offset_X=15,offset_Y=-4,n_pix_crop=n_crop)
M2C_bench = np.load('C:/Users/cheritier/Downloads/M2C_KL_OOPAO_synthetic.npy')

val_pix = np.var(imat_bench_2D,axis=2)
val_pix/=val_pix.max()
val_pix = val_pix>0.05

plt.figure()
plt.imshow(val_pix)

Ekatwin.check_pwfs_pupils(valid_pixel_map = val_pix,correct=False)

# rehape imat in 2D

imat_bench = imat_bench_2D.reshape(wfs.cam.resolution**2,imat_bench_2D.shape[-1])

#%%


imat_bench_zonal = imat_bench @ np.linalg.pinv(M2C_bench[:,:400])

plt.close('all')


dx = [0,0,0,0]
dy = [0,0,0,0]

wfs.apply_shift_wfs(sx=np.asarray(param['pwfs_pupils_shift_x'])+np.asarray(dx),
                    sy=np.asarray(param['pwfs_pupils_shift_y'])+np.asarray(dy))


Z = np.eye(468)
vect = np.zeros(468)
ind =[10,100,150,300,400]


for i in range(len(ind)):
    vect[ind[i]]=(1) *0.1
dm.coefs = Z @vect
ngs**tel*dm*wfs

from OOPAO.tools.displayTools import interactive_show

im_ = np.zeros(wfs.cam.frame.shape)
for i in range(len(ind)):
    im_ += imat_bench_zonal[:,ind[i]].reshape(wfs.cam.frame.shape)

interactive_show(im_,wfs.signal_2D)

#%%


# index_modes = [10,20,30,40,50,60,70,80,100,110,120,130,140,150,160,170,180,200,220,240,260,280,300,320,340,360,380]

# amp = 1e-9
# dm.coefs = M2C_bench[:,index_modes] *amp


# ngs**tel*dm*wfs


# from OOPAO.SPRINT import SPRINT
# from OOPAO.tools.tools import emptyClass

# # modal basis considered
# basis =  emptyClass()
# basis.modes         = M2C_bench[:,index_modes]
# basis.extra         = 'PAP_full_KL'              # EXTRA NAME TO DISTINGUISH DIFFERENT SENSITIVITY MATRICES, BE CAREFUL WITH THIS!     

 
# obj = emptyClass()

# obj.atm = atm
# obj.tel = tel
# obj.wfs = wfs
# obj.ngs = ngs
# obj.dm = dm
    
# Sprint = SPRINT(obj, basis,n_mis_reg=5,recompute_sensitivity=True , mis_registration_zero_point=dm.misReg,ind_mis_reg=[0,1,2,4,5])

# plt.close('all')
# Sprint.estimate(obj, on_sky_slopes = imat_bench[:,index_modes]*2e5,n_iteration=5,n_update_zero_point=0,tolerance=100)

# dm_sprint = obj.dm.apply_mis_registration(Sprint.mis_registration_out)


#%%


n=100
dm.coefs = M2C_bench[:,n]*0.1
ngs**tel*dm*wfs

from OOPAO.tools.displayTools import interactive_show

interactive_show(imat_bench_2D[:,:,n]*val_pix,wfs.signal_2D*val_pix)

#%% Function to swith to on-sky pupil (possibility to add an offset for the position of the pupil)
Ekatwin.set_pupil(calibration=True,
                   sky_offset=[0,0])

ngs*tel*wfs

plt.figure()
plt.imshow(tel.pupil)
plt.figure()
plt.imshow(wfs.cam.frame)

#%% PAPYRUS KL Basis Computation (only for the bench)
compute_kl_basis = True

if compute_kl_basis:
    from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
    M2C_synth = compute_KL_basis(tel = tel,
                           atm = atm,
                           dm  = dm_synth,
                           lim = 1e-4)
    M2C_exp = compute_KL_basis(tel = tel,
                           atm = atm,
                           dm  = dm,
                           lim = 1e-4)


# apply a few modes for display:  
dm.coefs = M2C_synth[:,:10] * 1e-9
ngs*tel*dm
displayMap(ngs.phase)

#%% Ekatwin Full Interaction Matrix Computation

wfs.modulation = 5
stroke = 0.00001

M2C_bench_centr = np.load('C:/Users/cheritier/OOPAO/tutorials/EKARUS/M2C_KL_OOPAO_central_obstruction.npy')


calib_dm_468 = InteractionMatrix(ngs       = ngs,
                            atm            = atm,
                            tel            = tel,
                            dm             = dm,
                            wfs            = wfs,
                            M2C            = M2C_bench_centr,
                            stroke         = stroke,
                            phaseOffset    = 0,\
                            nMeasurements  = 4,\
                            noise          = 'off',
                            print_time     = False,
                            display        = True)
    
#%%
plt.close('all')
plt.figure()
plt.plot(np.std(calib_dm_468.D,axis=0))
plt.plot(np.std(imat_bench/2,axis=0))

ratio = np.std(imat_bench[:,:300]/2,axis=0) /np.std(calib_dm_468.D[:,:300],axis=0)
plt.figure()
plt.plot(ratio)


#%%

# from Ekarus import decompress_ekarus_data

# imat_4_bench = decompress_ekarus_data(2*calib_dm_468.D[:,:400].reshape(wfs.cam.resolution,wfs.cam.resolution,400), shape = (240,240,400), n_pix=114, offset_X=15, offset_Y=-4,n_pix_crop=n_crop)
# plt.figure()
# plt.imshow(imat_4_bench[:,:,0])
# plt.figure()
# plt.imshow(imat_bench_full[:,:,0])

# imat_4_bench = np.reshape(imat_4_bench,(240**2,400))


# np.save('C:/Users/cheritier/Downloads/IM_rMod_5.0_nModes400_synthetic_v3.npy',imat_4_bench.astype(np.float32))

#%%  -----------------------     Close loop  ----------------------------------


# truncate to a given number of modes before the inversion
end_mode    = 250 

# closed loop data
M2C_CL      = M2C_bench_centr[:,:end_mode]  # modes-to-command matrix used in closed loop
calib_CL    = CalibrationVault(calib_dm_468.D[:,:end_mode])  # calibration object with reconstructor (calib_CL.M) for the number of modes selected

# compute the reconstructor (wfs signal - to - dm zonal commands )
reconstructor = M2C_CL@calib_CL.M

plt.figure()
plt.plot(np.std(calib_CL.D,axis=0),label='Ekatwin')
plt.legend()
plt.xlabel('KL Mode Index')
plt.ylabel('Int. Mat STD')
plt.title('Interaction Matrix Conditionning Number:'+str(calib_CL.cond))

#%%

n = 3
im_b = imat_bench[:,n].reshape(wfs.cam.frame.shape)
im_s = calib_dm_468.D[:,n].reshape(wfs.cam.frame.shape)

interactive_show(im_array=im_b, im_array_ref=im_s)

plt.figure()
plt.plot(imat_bench[:,n]/2)
plt.plot(calib_dm_468.D[:,n])
#%%

dm.coefs = M2C_CL[:,100]*0.1

ngs**tel*dm*wfs

plt.figure()
plt.imshow(ngs.OPD)
plt.figure()
plt.plot(calib_CL.M@wfs.signal)

plt.figure()
plt.imshow(wfs.signal_2D)
#%% Define instrument and WFS path detectors and reference PSFs for the SR computation
from OOPAO.Detector import Detector
from OOPAO.tools.tools import strehlMeter
# select desired pupil(Calib)

Ekatwin.set_pupil(calibration=True,spiders=True)

# compute KL modal projector truncating the modes by the pupil used
KL_basis_dm = (dm.modes@M2C_bench)*np.tile(tel.pupil.flatten()[:,None],M2C_bench.shape[1])
projector_kl = np.linalg.pinv(KL_basis_dm) 


#%%
# select desired pupil(Sky/Calib)
Ekatwin.set_pupil(calibration=False,spiders=False)

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

ngs**tel*ngs_cam
ngs_psf_ref = ngs_cam.frame.copy()

src**tel*src_cam

src_psf_ref = src_cam.frame.copy()

#%%


atm.initializeAtmosphere(tel)


# initialize Telescope DM commands
tel.resetOPD()
dm.coefs=0
ngs*tel*dm*wfs
wfs*wfs.focal_plane_camera

plt.close('all')
atm.r0 = 0.08
# To make sure to always replay the same turbulence, generate a new phase screen for the atmosphere and combine it with the Telescope
atm.generateNewPhaseScreen(seed=10)

# combine telescope with atmosphere
tel+atm


from OOPAO.FieldTransformer import FieldTransformer


pupil_wobble = FieldTransformer(ngs,
                                shift_x=[0],
                                shift_y=[0])

# propagate both sources
atm*ngs*tel*ngs_cam
atm*src*tel*src_cam

# loop parameters
nLoop = 2010  # number of iterations
gainCL = 0.3  # integrator gain
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
    ngs**atm*tel*pupil_wobble*dm*wfs
    cam_+=wfs.cam.frame
    if i>5:
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
    src**atm*tel*pupil_wobble*dm
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
    print('Elapsed time: ' + str(time.time()-a) + ' s')

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
                             [NGS_PSF.max()-3.75, NGS_PSF.max()],
                             [SRC_PSF.max()-3, SRC_PSF.max()]]
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

    
#%%
    
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
modes_out_wfs = calib_dm_468.M @ np.asarray(wfs_signals).T

plt.figure()
plt.loglog(np.std(np.asarray(modes_in).T[:,50:],axis=1),label ='Modal Projection')
plt.loglog(np.std(np.asarray(modes_in_dm)[:,50:],axis=1),'--',label ='From DM Commands')
plt.loglog(np.std(np.asarray(modes_out).T[:,50:],axis=1),label ='Modal Projection')
plt.loglog(np.std(np.asarray(modes_out_wfs)[:,50:],axis=1),'--',label ='From WFS signals')
plt.legend()
plt.xlabel('KL Mode Index')
plt.ylabel('WFE [nm]')
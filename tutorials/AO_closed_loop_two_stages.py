# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:01:52 2024

@author: cheritier


Tutorial Description — Two stages AO System in OOPAO

"""

import time

import matplotlib.pyplot as plt
import numpy as np
from OOPAO.tools.displayTools import cl_plot, displayMap

# %% two loops main parameter

# loop frequency
stage_1_frequency = 500
# number of subaperture for the WFS
stage_1_n_subaperture = 20

# loop frequency
stage_2_frequency = 1500
# number of subaperture for the WFS
stage_2_n_subaperture = 10

ratio_temporal = int(stage_2_frequency/stage_1_frequency)

#%% -----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope

# create the Telescope object
tel = Telescope(resolution           = 6*stage_1_n_subaperture,                          # resolution of the telescope in [pix]
                diameter             = 8,                                        # diameter in [m]        
                samplingTime         = 1/stage_2_frequency,                                   # Sampling time in [s] of the AO loop
                centralObstruction   = 0.,                                      # Central obstruction in [%] of a diameter 
                display_optical_path = False,                                    # Flag to display optical path
                fov                  = 0 )                                     # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of the phase screens but is uncompatible with off-axis targets

#%% -----------------------     NGS   ----------------------------------
from OOPAO.Source import Source

# create the Natural Guide Star object
stage_1_ngs = Source(optBand     = 'I',           # Optical band (see photometry.py)
             magnitude   = 8,             # Source Magnitude
             coordinates = [0,0])         # Source coordinated [arcsec,deg]

# combine the NGS to the telescope using '*'
stage_1_ngs*tel


# create the Natural Guide Star object
stage_2_ngs = Source(optBand     = 'I',           # Optical band (see photometry.py)
             magnitude   = 8,             # Source Magnitude
             coordinates = [0,0])         # Source coordinated [arcsec,deg]


# create the Natural Guide Star object
src = Source(optBand     = 'K',           # Optical band (see photometry.py)
             magnitude   = 8,             # Source Magnitude
             coordinates = [0,0])         # Source coordinated [arcsec,deg]
#%% -----------------------     ATMOSPHERE   ----------------------------------
from OOPAO.Atmosphere import Atmosphere
           
# create the Atmosphere object
atm = Atmosphere(telescope     = tel,                               # Telescope                              
                 r0            = 0.15,                              # Fried Parameter [m]
                 L0            = 25,                                # Outer Scale [m]
                 fractionalR0  = [0.45 ,0.1  ,0.1  ,0.25  ,0.1   ], # Cn2 Profile
                 windSpeed     = [10   ,12   ,11   ,15    ,20    ], # Wind Speed in [m]
                 windDirection = [0    ,72   ,144  ,216   ,288   ], # Wind Direction in [degrees]
                 altitude      = [0    ,1000 ,5000 ,10000 ,12000 ]) # Altitude Layers in [m]


# initialize atmosphere with current Telescope
atm.initializeAtmosphere(tel)


#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
from OOPAO.DeformableMirror import DeformableMirror
   
dm = DeformableMirror(telescope  = tel,                        # Telescope
                    nSubap       = 20,                     # number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)
                    mechCoupling = 0.35)                       # Mechanical Coupling for the influence functions
                        # Mechanical Coupling for the influence functions
    

#%% -----------------------     SHACK-HARTMANN WFS   ----------------------------------
from OOPAO.ShackHartmann import ShackHartmann

# make sure that the stage_1_ngs is propagated to the wfs
stage_1_ngs**tel

stage_1_wfs = ShackHartmann(nSubap = stage_1_n_subaperture,
                    telescope = tel,
                    lightRatio = 0.5,
                    shannon_sampling = False,
                    is_geometric=False)


from OOPAO.Pyramid import Pyramid

# make sure that the ngs_stage_2 is propagated to the wfs
stage_2_ngs**tel

stage_2_wfs = Pyramid(nSubap = stage_2_n_subaperture,
                      telescope = tel, 
                      modulation = 1,
                      lightRatio = 0.1)

#%% -----------------------     Modal Basis - KL Basis  ----------------------------------
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
# use of a single modal basis to prevent modal coupling between DM of stage 1 and stage 2
full_M2C = compute_KL_basis(tel,
                          atm,
                          dm,
                          lim = 0) # inversion stability criterion

# compute KL modal projector truncating the modes by the pupil used
KL_basis_dm = (dm.modes@full_M2C)*np.tile(tel.pupil.flatten()[:,None],full_M2C.shape[1])
projector_kl = np.linalg.pinv(KL_basis_dm) 

# modal DM for stage 1
stage_1_dm = DeformableMirror(telescope  = tel,                        # Telescope
                    nSubap       = stage_1_n_subaperture,                     # number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)
                    mechCoupling = 0.35,
                    modes=KL_basis_dm[:,:300])                       # Mechanical Coupling for the influence functions
    
# modal DM for stage 1
stage_2_dm = DeformableMirror(telescope  = tel,                        # Telescope
                    nSubap       = stage_2_n_subaperture,                     # number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)
                    mechCoupling = 0.35,
                    modes=KL_basis_dm[:,:80])                       # Mechanical Coupling for the influence functions
    

#%% -----------------------     Calibration: Interaction Matrix  ----------------------------------
from OOPAO. calibration.InteractionMatrix import InteractionMatrix
# amplitude of the modes in m
stroke=1e-9

# swap to geometric WFS for the calibration
stage_1_wfs.is_geometric = True
# zonal interaction matrix
stage_1_calibration = InteractionMatrix(ngs            = stage_1_ngs,
                                atm            = atm,
                                tel            = tel,
                                dm             = stage_1_dm,
                                wfs            = stage_1_wfs,   
                                M2C            = np.eye(300), # M2C matrix used 
                                stroke         = stroke,    # stroke for the push/pull in M2C units
                                nMeasurements  = 1,        # number of simultaneous measurements
                                noise          = 'off',     # disable wfs.cam noise 
                                display        = True,      # display the time using tqdm
                                single_pass    = True)      # only push to compute the interaction matrix instead of push-pull


# switch back to diffractive WFS
stage_1_wfs.is_geometric = False


# zonal interaction matrix
stage_2_calibration = InteractionMatrix(ngs            = stage_2_ngs,
                                atm            = atm,
                                tel            = tel,
                                dm             = stage_2_dm,
                                wfs            = stage_2_wfs,   
                                M2C            = np.eye(80), # M2C matrix used 
                                stroke         = stroke,    # stroke for the push/pull in M2C units
                                nMeasurements  = 1,        # number of simultaneous measurements
                                noise          = 'off',     # disable wfs.cam noise 
                                display        = True,      # display the time using tqdm
                                single_pass    = True)      # only push to compute the interaction matrix instead of push-pull


#%% Define instrument and WFS path detectors
from OOPAO.Detector import Detector
# instrument path
src_cam = Detector(tel.resolution*2,log_scale=False)
src_cam.psf_sampling = 4  # sampling of the PSF
src_cam.integrationTime = tel.samplingTime # exposure time for the PSF
src**tel*src_cam
src_psf_ref = src_cam.frame.copy()

# WFS path
stage_1_cam = Detector(tel.resolution*2,log_scale=False)
stage_1_cam.psf_sampling = 4
stage_1_cam.integrationTime = tel.samplingTime
stage_1_ngs**tel*stage_1_cam
stage_1_psf_ref = stage_1_cam.frame.copy()

# WFS path
stage_2_cam = Detector(tel.resolution*2,log_scale=False)
stage_2_cam.psf_sampling = 4
stage_2_cam.integrationTime = tel.samplingTime
stage_2_ngs**tel*stage_2_cam
stage_2_psf_ref = stage_2_cam.frame.copy()



#%%  Closed loop simulation
from OOPAO.tools.tools import strehlMeter


plt.close('all')

stage_1_reconstructor = stage_1_calibration.M 
stage_2_reconstructor = stage_2_calibration.M 


# initialize DM commands
stage_1_dm.coefs=0
stage_2_dm.coefs=0

# To make sure to always replay the same turbulence, generate a new phase screen for the atmosphere and combine it with the Telescope
atm.generateNewPhaseScreen(seed=10)

# propagate both sources
stage_1_ngs**atm*tel*stage_1_cam
stage_2_ngs**atm*tel*stage_2_cam

src**atm*tel*src_cam

# loop parameters
nLoop = 1000  # number of iterations


stage_1_gain = 0.3 # integrator gain
stage_1_frame_delay = 2  # number of frame delay
stage_1_buffer_wfs = stage_1_wfs.signal*0
stage_1_buffer_wfs_fast_rate = []#np.zeros((int(ratio_temporal),stage_1_wfs.raw_data.shape[0],stage_1_wfs.raw_data.shape[1]))


stage_2_gain = 0.4 # integrator gain
stage_2_frame_delay = 2  # number of frame delay
stage_2_buffer_wfs = stage_2_wfs.signal*0

display = False  # enable the display

# variables used to to save closed-loop data data
SR_ngs_stage_1 = np.zeros(nLoop)
SR_ngs_stage_2 = np.zeros(nLoop)
SR_src = np.zeros(nLoop)

wfe_atmosphere = np.zeros(nLoop)
wfe_residual_SRC = np.zeros(nLoop)
wfe_residual_NGS_stage_1 = np.zeros(nLoop)
wfe_residual_NGS_stage_2 = np.zeros(nLoop)

# configure the display pannel
plot_obj = cl_plot(list_fig=[atm.OPD,  # list of data for the different subplots
                             stage_1_ngs.OPD,
                             stage_2_ngs.OPD,
                             stage_1_wfs.cam.frame,
                             stage_2_wfs.cam.frame,
                             [[0, 0], [0, 0], [0, 0],[0, 0]],
                             (stage_1_cam.frame),
                             (stage_2_cam.frame)],
                   type_fig=['imshow',  # type of figure for the different subplots
                             'imshow',
                             'imshow',
                             'imshow',
                             'imshow',
                             'plot',
                             'imshow',
                             'imshow'],
                   list_title=['Turbulence [nm]',  # list of title for the different subplots
                               'stage 1 residual [m]',
                               'stage 2 residual [m]',
                               'WFS 1 Detector',
                               'WFS 2 Detector',
                               None,
                               None,
                               None],
                   list_legend=[None,  # list of legend labels for the subplots
                                None,
                                None,
                                None,
                                None,
                                ['Stage 1','Stage 2','Science'],
                                None,
                                None],
                   list_label=[None,  # list of axis labels for the subplots
                               None,
                               None,
                               None,
                               None,
                               ['Time', 'WFE [nm]'],
                               ['NGS PSF@' + str(stage_1_ngs.coordinates[0]) + '" -- FOV: ' + str(np.round(stage_1_cam.fov_arcsec, 2)) + '"', ''],
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
modes_in = []
modes_out_stage_1 = []
modes_out_stage_2 = []
#%
for i in range(nLoop):
    a = time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    src**atm*tel
    modes_in.append(projector_kl@src.OPD.flatten())
    
    # save the wave-front error of the incoming turbulence within the pupil
    wfe_atmosphere[i] = np.std(src.OPD[np.where(tel.pupil > 0)])*1e9
    # propagate light from the stage_1_ngs through the atmosphere, telescope, DM to the WFS and stage_1_ngs camera
    stage_1_ngs ** atm * tel * stage_1_dm *stage_1_wfs * stage_1_cam
    
    # # propagate light from the stage_2_ngs through the atmosphere, telescope, DM to the WFS and stage_1_ngs camera
    stage_2_ngs ** atm * tel * stage_1_dm * stage_2_dm * stage_2_wfs * stage_2_cam
    
    # # propagate light from the src through the atmosphere, telescope, DM to the WFS and stage_1_ngs camera
    src**atm*tel*stage_1_dm
    modes_out_stage_1.append(projector_kl@src.OPD.flatten())

    src**atm*tel*stage_1_dm*stage_2_dm*src_cam
    modes_out_stage_2.append(projector_kl@src.OPD.flatten())
    
    stage_1_buffer_wfs_fast_rate. append(stage_1_wfs.raw_data)
    
    # # store the slopes after propagating to the WFS <=> 1 frames delay
    if i%ratio_temporal == 0:

        stage_1_wfs.cam._integrated_time = (ratio_temporal-1)* tel.samplingTime
        stage_1_wfs.raw_data = np.sum(np.asarray(stage_1_buffer_wfs_fast_rate),axis=0)
        stage_1_wfs.wfs_integrate()
        stage_1_buffer_wfs_fast_rate = []
        
        if stage_1_frame_delay == 1:
            stage_1_buffer_wfs = stage_1_wfs.signal
        
        stage_1_dm.coefs = stage_1_dm.coefs - stage_1_gain*np.matmul(stage_1_reconstructor, stage_1_buffer_wfs)
    
        # # apply the commands on the DM
        if stage_1_frame_delay == 2:
            stage_1_buffer_wfs = stage_1_wfs.signal

    # # save residuals corresponding to the stage_1_ngs
    wfe_residual_NGS_stage_1[i] = np.std(stage_1_ngs.OPD[np.where(tel.pupil > 0)])*1e9
    wfe_residual_NGS_stage_2[i] = np.std(stage_2_ngs.OPD[np.where(tel.pupil > 0)])*1e9
    wfe_residual_SRC[i] = np.std(src.OPD[np.where(tel.pupil > 0)])*1e9

    # # # save Strehl ratio from the PSF image
    SR_src[i] = strehlMeter(PSF=src_cam.frame, tel=tel, PSF_ref=src_psf_ref, display=False)
    SR_ngs_stage_1[i] = strehlMeter(PSF=stage_1_cam.frame, tel=tel, PSF_ref=stage_1_psf_ref, display=False)
    SR_ngs_stage_2[i] = strehlMeter(PSF=stage_2_cam.frame, tel=tel, PSF_ref=stage_2_psf_ref, display=False)
       
    if stage_2_frame_delay == 1:
        stage_2_buffer_wfs = stage_2_wfs.signal
    
    if i>20:
        stage_2_dm.coefs = stage_2_dm.coefs - stage_2_gain*np.matmul(stage_2_reconstructor, stage_2_buffer_wfs)
    if stage_2_frame_delay == 2:
        stage_2_buffer_wfs = stage_2_wfs.signal
        
    # update displays if required
    if display and i > 1:
        # update range for PSF images
        plot_obj.list_lim = [None,
                             None,
                             None,
                             None,
                             None,
                             None,
                             [(np.log10(stage_1_cam.frame).max())-4, (np.log10(stage_1_cam.frame)).max()],
                             [(np.log10(stage_2_cam.frame)).max()-4, (np.log10(stage_2_cam.frame)).max()]                             ]
        # update title
        plot_obj.list_title = ['Turbulence '+str(np.round(wfe_atmosphere[i]))+'[nm]',
                               'Stage 1 residual '+str(np.round(wfe_residual_NGS_stage_1[i]))+'[nm]',
                               'Stage 2 residual '+str(np.round(wfe_residual_NGS_stage_2[i]))+'[nm]',
                               'WFS Detector',
                               'DM Commands',
                               None,
                               None,
                               None]

        cl_plot(list_fig=[1e9*atm.OPD,
                          1e9*stage_1_ngs.OPD,
                          1e9*stage_2_ngs.OPD,
                          stage_1_wfs.cam.frame,
                          stage_2_wfs.cam.frame,
                          [np.arange(i+1), wfe_residual_NGS_stage_1[:i+1], wfe_residual_NGS_stage_2[:i+1]],
                          np.log10(stage_1_cam.frame),
                           np.log10(stage_2_cam.frame)],
                plt_obj=plot_obj)
        plt.pause(0.01)
        if plot_obj.keep_going is False:
            break
    print('-----------------------------------')
    print('Loop'+str(i) + '/' + str(nLoop))
    print('Stage 1: Strehl ratio [%] : ', np.round(SR_ngs_stage_1[i],1), ' WFE [nm] : ', np.round(wfe_residual_NGS_stage_1[i],2))
    print('Stage 2: Strehl ratio [%] : ', np.round(SR_ngs_stage_2[i],1), ' WFE [nm] : ', np.round(wfe_residual_NGS_stage_2[i],2))
    print('Science : Strehl ratio [%] : ', np.round(SR_src[i],1), ' WFE [nm] : ', np.round(wfe_residual_SRC[i],2))

    
    
#%% Closed Loop data analysis

plt.figure()
plt.loglog(np.std(np.asarray(modes_in).T[:,50:],axis=1),label ='Atmosphere')
plt.loglog(np.std(np.asarray(modes_out_stage_1).T[:,50:],axis=1),label ='Stage 1')
plt.loglog(np.std(np.asarray(modes_out_stage_2).T[:,50:],axis=1),'--',label ='Stage 2')
plt.legend()
plt.xlabel('KL Mode Index')
plt.ylabel('WFE [nm]')

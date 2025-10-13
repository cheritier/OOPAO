# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 10:40:42 2023

Accurate version of PAPYRUS AO System used for reproducing the real system in details.
- 17/06/2025: Update after change of the WFS camera.
- 23/06/2025: Update to prepare the integration with DAO

@author: cheritie - astriffl
"""
import time
import matplotlib.pyplot as plt
import numpy as np
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap
# make sure that the relative import can work
import sys
xs = sys.path
matching = [s for s in xs if "OOPAO" in s]
sys.path.append(matching[0]+'/tutorials/ORP_AO_School/')
loc = matching[0]+'/tutorials/PAPYRUS/'


from Papyrus import Papyrus


#%% Compute the OOPAO Objects

Papytwin = Papyrus()
# telescope object
tel     = Papytwin.tel
# source WFS object
ngs     = Papytwin.ngs
# source science object
src     = Papytwin.src
# deformable mirror object
dm      = Papytwin.dm
# Pyramid WFS object
wfs     = Papytwin.wfs
# atmosphere object
atm     = Papytwin.atm
# slow Tip/Tilt object
slow_tt = Papytwin.slow_tt
# Modes-to-Command matrix
M2C = Papytwin.M2C

# parameter file
param   = Papytwin.param



#%%
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration

M2C_CL      = Papytwin.M2C
wfs.modulation = 3
stroke = 1e-9
calib = InteractionMatrix(  ngs            = ngs,\
                            atm            = atm,\
                            tel            = tel,\
                            dm             = dm,\
                            wfs            = wfs,\
                            M2C            = M2C_CL,\
                            stroke         = stroke,\
                            phaseOffset    = 0,\
                            nMeasurements  = 4,\
                            print_time=False,
                            display=True)

    
a = displayMap(tel.OPD,norma=True, returnOutput=True)
b = displayMap(calib.D[:,:], norma = True,axis=1, returnOutput=True)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(a)
plt.axis('off')
plt.title("PAPYTWIN DM shape")

plt.subplot(1,2,2)
plt.imshow(b)
plt.axis('off')
plt.title("PAPYTWIN Interaction Matrix")

# plt.subplot(1,3,3)
# plt.imshow(c)
# plt.axis('off')
# plt.title("PAPYRUS Interaction Matrix")





#%%


dm.coefs = M2C[:,10] *1e-9


tel.resetOPD()
ngs*tel*dm*wfs

plt.figure(),plt.plot(calib.M@wfs.signal)



dm.coefs = M2C[:,10] *1e-9 * 200


tel.resetOPD()
ngs*tel*dm*wfs

plt.figure(),plt.plot(calib.M@wfs.signal)


dm.coefs = M2C@calib.M@wfs.signal
plt.figure(),plt.imshow(dm.OPD*tel.pupil)


#%%








#%% PAPYTWIN Full Interaction Matrix Computation
ngs*tel*wfs
wfs.modulation = 5

stroke =  1e-9
calib = InteractionMatrix(  ngs            = ngs,
                            atm            = atm,
                            tel            = tel,
                            dm             = dm,
                            wfs            = wfs,
                            M2C            = M2C,
                            stroke         = stroke,
                            phaseOffset    = 0,\
                            nMeasurements  = 2,\
                            noise          = 'off',
                            print_time     = False,
                            display        = True)
    

b = displayMap(calib.D, norma = True,axis=1,returnOutput=True)
plt.title("Synthetic Interaction Matrix")

a[np.isinf(a)] = 0
b[np.isinf(b)] = 0

from OOPAO.tools.displayTools import interactive_show
interactive_show(a,b)

plt.figure()
plt.plot(np.std(calib.D,axis=0),label='PAPYTWIN')
plt.legend()
plt.xlabel('KL Mode Index')
plt.ylabel('Int. Mat STD')

#%%  -----------------------     Close loop  ----------------------------------
tel.resetOPD()

# These are the calibration data used to close the loop
M2C_CL = M2C[:,:]
calib_CL    = CalibrationVault(calib.D[:,:])


#%% test of the interaction matrix

tel.resetOPD()
dm.coefs = M2C_CL[:,10] *1e-9

ngs*tel*dm*wfs
plt.figure()
plt.plot(calib_CL.M @ wfs.signal)

Papytwin.set_pupil(calibration=False)
tel.resetOPD()
wfs.modulation = 5

dm.coefs = M2C_CL[:,10] *1e-9

ngs*tel*dm*wfs
plt.figure()
plt.plot(calib_CL.M @ wfs.signal)


#%%

from OOPAO.NCPA import NCPA

ncpa = NCPA(tel, dm, atm, M2C = M2C_CL, coefficients=[0,0])
dm.coefs = M2C_CL[:,2]*120*1e-9
ncpa.OPD = dm.OPD.copy()

#%%
tel-atm
dm.coefs = M2C_CL[:,2]*120*1e-9

src*tel*dm*src_cam


P = (src_cam.frame)

P /=P.max()

plt.figure()
plt.imshow(np.log10(P))
plt.clim([-2.5,0])


plt.figure()
plt.imshow(tel.OPD) 


#%%
from OOPAO.tools.tools import strehlMeter




Papytwin.set_pupil(calibration=False)
from OOPAO.Atmosphere import Atmosphere

atm = Atmosphere(telescope      = tel, 
                 r0             = 0.05,
                 L0             = 25, 
                 windSpeed      = [8,4,10,5], 
                 fractionalR0   = [0.5,0.15,0.25,0.1], 
                 windDirection  = [0,30,90,180],
                 altitude       = [0,5000,10000,12000])

atm.initializeAtmosphere(tel)


from OOPAO.Detector import Detector
from OOPAO.Source import Source


#%% Define instrument and WFS path detectors
from OOPAO.Detector import Detector
# instrument path
src_cam = Detector(tel.resolution)
src_cam.psf_sampling = 2.5  # sampling of the PSF
src_cam.integrationTime = tel.samplingTime # exposure time for the PSF

# put the scientific target off-axis to simulate anisoplanetism (set to  [0,0] to remove anisoplanetism)
src.coordinates = [0,0]

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

#%%  Closed loop simulation
from OOPAO.tools.tools import strehlMeter

plt.close('all')

# These are the calibration data used to close the loop

reconstructor = M2C_CL@calib_CL.M 

# initialize Telescope DM commands
tel.resetOPD()
dm.coefs=0
ngs*tel*dm*wfs


# You can update the the atmosphere parameter on the fly
# atm.r0 = 0.15
# atm.windSpeed = list(np.random.randint(5,20,atm.nLayer))
# atm.windDirection = list(np.random.randint(0,360,atm.nLayer))

# To make sure to always replay the same turbulence, generate a new phase screen for the atmosphere and combine it with the Telescope
# atm.windSpeed = [4]
atm.generateNewPhaseScreen(seed=10)

# combine telescope with atmosphere
tel+atm

# propagate both sources
atm*ngs*tel*ngs_cam
atm*src*tel*src_cam

# loop parameters
nLoop = 500  # number of iterations
gainCL = 0.4  # integrator gain
wfs.cam.photonNoise = False  # enable photon noise on the WFS camera
display = True  # enable the display
frame_delay = 2  # number of frame delay

# variables used to to save closed-loop data data
SR_ngs = np.zeros(nLoop)
SR_src = np.zeros(nLoop)

wfe_atmosphere = np.zeros(nLoop)
wfe_residual_SRC = np.zeros(nLoop)
wfe_residual_NGS = np.zeros(nLoop)
wfsSignal = np.arange(0, wfs.nSignal)*0  # buffer to simulate the loop delay

C_buff = []
dC_buff = []

for i_w in [1]:
    dm_commands = []
    wfs_signals = []
    # To make sure to always replay the same turbulence, generate a new phase screen for the atmosphere and combine it with the Telescope
    # atm.windSpeed = [i_w]
    atm.generateNewPhaseScreen(seed=10)
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
    
    
    for i in range(nLoop):
        a = time.time()
        # update phase screens => overwrite tel.OPD and consequently tel.src.phase
        atm.update()
        # save the wave-front error of the incoming turbulence within the pupil
        wfe_atmosphere[i] = np.std(tel.OPD[np.where(tel.pupil > 0)])*1e9
        # propagate light from the ngs through the atmosphere, telescope, DM to the WFS and ngs camera
        atm*ngs*tel*dm*wfs*ngs_cam
        # propagate to the focal plane camera
        wfs*wfs.focal_plane_camera
        wfs_signals.append(wfs.signal)
        dm_commands.append(dm.coefs)
        # save residuals corresponding to the ngs
        wfe_residual_NGS[i] = np.std(tel.OPD[np.where(tel.pupil > 0)])*1e9
        # save Strehl ratio from the PSF image
        SR_ngs[i] = strehlMeter(PSF=ngs_cam.frame, tel=tel, PSF_ref=ngs_psf_ref, display=False)
        # save the OPD seen by the ngs
        OPD_NGS = tel.mean_removed_OPD.copy()
        if display:
            NGS_PSF = np.log10(np.abs(ngs_cam.frame))
    
        # propagate light from the src through the atmosphere, telescope, DM to the src camera
        atm*src*tel*dm*ncpa*src_cam
        # tmp_opd = tel.OPD.copy()
        # src*tel*src_cam
        
        # save residuals corresponding to the SRC
        wfe_residual_SRC[i] = np.std(tel.OPD[np.where(tel.pupil > 0)])*1e9
        # save the OPD seen by the src
        OPD_SRC = tel.mean_removed_OPD.copy()
        # save Strehl ratio from the PSF image
        SR_src[i] = strehlMeter(PSF=src_cam.frame, tel=tel, PSF_ref=src_psf_ref, display=False)
    
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
                                 [NGS_PSF.max()-3, NGS_PSF.max()],
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
                              np.log10(wfs.focal_plane_camera.frame),
                              [np.arange(i+1), wfe_residual_SRC[:i+1], wfe_residual_NGS[:i+1]],
                              NGS_PSF,
                              SRC_PSF],
                    plt_obj=plot_obj)
            plt.pause(0.01)
            if plot_obj.keep_going is False:
                break
        print('-----------------------------------')
        print('Loop'+str(i) + '/' + str(nLoop))
        print('NGS: Strehl ratio [%] : ', np.round(SR_ngs[i],1), ' WFE [nm] : ', np.round(wfe_residual_NGS[i],2))
        print('SRC: Strehl ratio [%] : ', np.round(SR_src[i],1), ' WFE [nm] : ', np.round(wfe_residual_SRC[i],2))
    
        
        
    # #%% Closed Loop data analysis
    
    plt.figure(111)
    plt.plot(np.arange(nLoop)*tel.samplingTime, wfe_atmosphere, label='Turbulence')
    plt.plot(np.arange(nLoop)*tel.samplingTime, wfe_residual_NGS, label='NGS')
    plt.plot(np.arange(nLoop)*tel.samplingTime, wfe_residual_SRC, label='SRC')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('WFE [nm]')
    
    plt.figure(112)
    plt.plot(np.arange(nLoop)*tel.samplingTime, SR_ngs, label='NGS@' + str(np.round(1e9*ngs.wavelength,0)) + ' nm')
    plt.plot(np.arange(nLoop)*tel.samplingTime, SR_src, label='SRC@' + str(np.round(1e9*src.wavelength,0)) + ' nm')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('SR [%]')
    
    S = np.asarray(wfs_signals)
    # U = np.asarray(dm_commands)
    
    S = S[20:,:] 
    
    dS = S[20:,:] - S[19:-1,:]
    
    
    C = S@S.T
    dC = dS@dS.T
    
    C_buff.append(C)
    dC_buff.append(dC)


# %%

# C = S@S.T


#%%

ss = 60

for i_ in range(3):
    m = np.zeros(ss)

    for i in range(ss//2,C_buff[i_].shape[0]-ss):
        m+=C_buff[i_][i-ss//2:i+ss//2,i]
    
    plt.figure(0)
    plt.plot(1/np.arange(-ss/2,ss//2),m/m.max(),'-o')

    m = np.zeros(ss)

    for i in range(ss//2,dC_buff[0].shape[0]-ss):
        m+=dC_buff[i_][i-ss//2:i+ss//2,i]
    
    plt.figure(i)
    plt.plot(1/np.arange(-ss/2,ss//2),m/m.max(),'-o')






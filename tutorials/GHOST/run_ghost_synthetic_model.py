# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:43:10 2022

@author: cheritie
"""

import matplotlib.pyplot as plt
import numpy             as np 
import time

from compute_ao_model_pyramid import compute_ao_model_pyramid
# calibration modules compute_ao_model_pyramid
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
# display modules
from OOPAO.tools.displayTools           import displayMap, display_wfs_signals, makeSquareAxes,interactive_show

#% -----------------------     read parameter file   ----------------------------------
from parameterFile_GHOST import initializeParameterFile, get_imat_ghost, compute_imat_4_ghost
param = initializeParameterFile()

# location of the relevant data (WFS pupil mask, KL basis, measured iMat... )
loc = 'C:/Users/cheritier/Documents/oopao_private/ghost/ghost_simulation/'

# create the ghost AO objects
tel,ngs,dm,wfs,atm = compute_ao_model_pyramid(param = param, loc = loc)


#%% Modal basis

M2C = np.load(loc+'KL_PTT_ifun_BMC492_psim.npy')

dm.coefs = M2C[:,10:20]*1e-9

ngs**tel*dm*wfs
# show the KL modes projected on the synthetic DM
displayMap(tel.OPD, norma = True)

# show the corresponding synthetic WFS signals
sim_imat = display_wfs_signals(wfs, wfs.signal, norma=True,returnOutput=True)
sim_imat[np.isinf(sim_imat)] = 0

#%% Load experimental interaction matrix

#% read and display the input imat
# consider only 400 modes in this example
input_imat = get_imat_ghost(loc+'hadamard_IM_slopes_15_no_residual.npy', wfs, M2C=M2C[:,:400])

# display the modes 10 to 20
exp_imat = display_wfs_signals(wfs, input_imat[:,10:20], norma = True,returnOutput=True)
exp_imat[np.isinf(exp_imat)] = 0



#%% SPRINT estimation

# index of the KL modes used to estimate the parameters (for the first estimation pick a few middle-order within the first 100 KL modes -- avoid high order modes)
index_modes = np.arange(80,100)

# modal basis considered
from OOPAO.tools.tools import emptyClass
basis               = emptyClass()
basis.modes         = M2C[:,index_modes]
basis.extra         = 'ghost_smat'

# compact the AO objects into a single object obj
obj         =  emptyClass()
obj.ngs     = ngs
obj.tel     = tel
obj.wfs     = wfs
obj.dm      = dm
obj.atm     = atm
obj.param   = param


from OOPAO.SPRINT import SPRINT
tel.resetOPD()
n_mis_reg             = 5           # consider the 5 mis-reg shift X & Y, rotation, magnification X & Y
recompute_sensitivity = True        # forces to recompute the sensitivity matrices. If False, the pre-eisting ones are loaded if they exist.
mis_registration_zero_point = None  # mis-registration starting point. if None, zero-mis-registration is considered

# create the SPRINT object
Sprint = SPRINT(obj                         = obj,\
                basis                       = basis,\
                n_mis_reg                   = n_mis_reg,\
                mis_registration_zero_point = None,\
                recompute_sensitivity       = True)

    
# estimate the parameters
Sprint.estimate(obj, input_imat[:,index_modes],n_iteration=6,n_update_zero_point=1)   
#%%

# print the identified value: 
print('The validity flag for the estimation of the parameters is '+str(Sprint.validity_flag))
# the value can be printed using
Sprint.mis_registration_out.print_()


#%%

from OOPAO.mis_registration_identification_algorithm.applyMisRegistration import applyMisRegistration

dm_ghost = applyMisRegistration(tel,Sprint.mis_registration_out,param)

input_imat = get_imat_ghost(loc+'hadamard_IM_slopes_15_no_residual.npy', wfs, M2C=M2C[:,:400])

ind = [80,81,82,100]
# show the comparison for a few modes
dm.coefs = M2C[:,ind]*1e-9

ngs**tel*dm*wfs

sim_imat = display_wfs_signals(wfs, wfs.signal, norma=True,returnOutput=True)
sim_imat[np.isinf(sim_imat)] = 0

# display the modes 10 to 20
exp_imat = display_wfs_signals(wfs, input_imat[:,ind], norma = True,returnOutput=True)
exp_imat[np.isinf(exp_imat)] = 0

interactive_show(sim_imat,exp_imat)

display_wfs_signals(wfs, np.var(input_imat,axis=1), norma = True)



#%% Zonal interaction matrix

wfs.modulation = 3

from OOPAO.calibration.InteractionMatrix import InteractionMatrix


stroke = 1e-9

calib = InteractionMatrix(  ngs            = ngs,\
                            atm            = atm,\
                            tel            = tel,\
                            dm             = dm_ghost,\
                            wfs            = wfs,\
                            M2C            = np.eye(492),\
                            stroke         = stroke,\
                            phaseOffset    = 0,\
                            nMeasurements  = 10,\
                            noise          = 'off')
    
    
#%%
imat_from_ghost = get_imat_ghost(loc+'hadamard_IM_slopes_15_no_residual.npy', wfs,normalization_factor=1)

imat_4_ghost = 0.75*compute_imat_4_ghost(calib.D, wfs)

plt.figure()
plt.plot(np.std(imat_4_ghost,axis=0))

plt.plot(np.std(imat_from_ghost,axis=0))



np.save(loc+'imat_4_ghost_mod_3',imat_4_ghost)




#%% Define instrument and WFS path detectors
from OOPAO.Detector import Detector
from OOPAO.Source import Source
# instrument path
src_cam = Detector(tel.resolution*2)
src_cam.psf_sampling = 4  # sampling of the PSF
src_cam.integrationTime = tel.samplingTime # exposure time for the PSF
# create the Scientific Target object located at 10 arcsec from the  ngs
src = Source(optBand     = 'K',           # Optical band (see photometry.py)
             magnitude   = 8,              # Source Magnitude
             coordinates = [0,0])        # Source coordinated [arcsec,deg]

# WFS path
ngs_cam = Detector(tel.resolution*2)
ngs_cam.psf_sampling = 4
ngs_cam.integrationTime = tel.samplingTime

ngs**tel*ngs_cam

ngs_psf_ref = ngs_cam.frame.copy()
src_psf_ref = src_cam.frame.copy()

#%%  Closed loop simulation
from OOPAO.tools.tools import strehlMeter
from OOPAO.calibration.CalibrationVault import CalibrationVault
plt.close('all')

# These are the calibration data used to close the loop
n_modes = 400
calib_CL = CalibrationVault(input_imat[:,:n_modes])
M2C_CL = M2C[:,:n_modes]
reconstructor = M2C_CL@calib_CL.M 

ngs**tel
atm.initializeAtmosphere(tel)

atm.update()
#%%
from OOPAO.tools.displayTools import cl_plot
# initialize Telescope DM commands
dm.coefs=0



# combine telescope with atmosphere
tel+atm

# propagate both sources
ngs**atm*tel*ngs_cam

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
    ngs**atm*tel*dm*wfs*ngs_cam
    # propagate to the focal plane camera
    wfs*wfs.focal_plane_camera
    # save residuals corresponding to the ngs
    wfe_residual_NGS[i] = np.std(tel.OPD[np.where(tel.pupil > 0)])*1e9
    # save Strehl ratio from the PSF image
    SR_ngs[i] = strehlMeter(PSF=ngs_cam.frame, tel=tel, PSF_ref=ngs_psf_ref, display=False)
    # save the OPD seen by the ngs
    OPD_NGS = ngs.OPD.copy()
    if display:
        NGS_PSF = np.log10(np.abs(ngs_cam.frame))

    # propagate light from the src through the atmosphere, telescope, DM to the src camera
    src**atm*tel*dm*src_cam
    # save residuals corresponding to the SRC
    wfe_residual_SRC[i] = np.std(tel.OPD[np.where(tel.pupil > 0)])*1e9
    # save the OPD seen by the src
    OPD_SRC = src.OPD.copy()
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
                             [NGS_PSF.max()-6, NGS_PSF.max()],
                             [SRC_PSF.max()-6, SRC_PSF.max()]]
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

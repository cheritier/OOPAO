# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 10:40:42 2023

Accurate version of PAPYRUS AO System used for reproducing the real system in details.

@author: cheritie - astriffl
"""
from pymatreader import read_mat 
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap, display_wfs_signals
from compute_papytwin import compute_papyrus_model,optimize_pwfs_pupils, bin_bench_data

#% -----------------------     read parameter file   ----------------------------------

from parameter_files.parameterFile_papytwin import initializeParameterFile
param = initializeParameterFile()

# ratio of number of subaperture (1 == real scale simulation)
param['ratio'] = 1
# BE SUR TO SET CONSOLE TO WORKING DIRECTORY BEFORE RUNNING
directory = os.getcwd().replace("\\", "/")

# location of the relevant data (WFS pupil mask, KL basis, measured iMat, T152 pupil, ... )
loc = directory + '/papyrus_inputs/'

# tel_calib,_,dm_calib,_,_ = compute_papyrus_model(param = param, loc = loc, source=False, IFreal=IFreal)
tel,ngs,dm,wfs,atm = compute_papyrus_model(param = param, loc = loc, source=True, IFreal=False)

ngs*tel*dm*wfs

atm.initializeAtmosphere(tel)

#%%
from OOPAO.tools.tools import read_fits

M2C = np.load('M2C.npy')

valid_pixel = np.load('valid_pixel.npy')

# only extract of full experimental int-mat -- full matrix avalaible upon request

int_mat_extract = np.load('int_mat_1_5_10_20_30_50_80_100_150.npy')

# index of the KL modes included in the int-mat
ind = [1, 5, 10, 20, 30, 50, 80, 100, 150]

valid_pixel, int_mat_binned = bin_bench_data(valid_pixel = valid_pixel, full_int_mat = int_mat_extract, ratio = param['ratio'])

#%% SET WFS PUPILS

from OOPAO.tools.displayTools import interactive_show
from OOPAO.Pyramid import Pyramid

   
optimize_pwfs_pupils(wfs = wfs ,valid_pixel_map = valid_pixel)
    
#%%
from parameter_files.OCAM2K  import OCAM_param
from OOPAO.Detector import Detector

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
ngs*tel*wfs


#%%
# dm.modes = dm.modes /alpao_unit
M2C_CL      = M2C[:,ind]


wfs.modulation = 5

stroke = 0.0001
calib = InteractionMatrix(  ngs            = ngs,\
                            atm            = atm,\
                            tel            = tel,\
                            dm             = dm,\
                            wfs            = wfs,\
                            M2C            = M2C_CL,\
                            stroke         = stroke,\
                            phaseOffset    = 0,\
                            nMeasurements  = 1,\
                            noise          = 'off',
                            print_time=False,
                            display=True)
    

#%%

displayMap(int_mat_extract, norma = True,axis=1)
plt.title("Experimental Interaction Matrix")
displayMap(calib.D, norma = True,axis=1)
plt.title("Synthetic Interaction Matrix")


#%%  -----------------------     Close loop  ----------------------------------
tel.resetOPD()
M2C_CL      = M2C.copy()

wfs.modulation = 5

stroke = 0.0001
calib = InteractionMatrix(  ngs            = ngs,\
                            atm            = atm,\
                            tel            = tel,\
                            dm             = dm,\
                            wfs            = wfs,\
                            M2C            = M2C_CL,\
                            stroke         = stroke,\
                            phaseOffset    = 0,\
                            nMeasurements  = 1,\
                            noise          = 'off',
                            print_time=False,
                            display=True)
#%%  -----------------------     Close loop  ----------------------------------
tel.resetOPD()

end_mode    = 195 
# These are the calibration data used to close the loop
# use of experimental calibration
# if full int-mat is available only
#calib_CL    = CalibrationVault(int_mat_binned[:,:end_mode])

# use of synthetic calibration

calib_CL    = CalibrationVault(calib.D[:,:end_mode])

#%%

from OOPAO.Atmosphere import Atmosphere


atm = Atmosphere(telescope = tel, 
                 r0 =0.06,
                 L0=25, 
                 windSpeed = [5], 
                 fractionalR0 = [1], 
                 windDirection = [0],
                 altitude = [0])

atm.initializeAtmosphere(tel)



#%%
from OOPAO.Detector import Detector
from OOPAO.Source import Source



# instrument path
src_cam = Detector(nRes=100)
src_cam.psf_sampling = 4
src_cam.integrationTime = tel.samplingTime


#create scientific source
src =   Source('IR1310', 0)
src*tel


# WFS path
ngs_cam = Detector(nRes = 100)
ngs_cam.psf_sampling = 4
ngs_cam.integrationTime = tel.samplingTime

# initialize Telescope DM commands
tel.resetOPD()
dm.coefs=0
ngs*tel*dm*wfs
wfs*wfs.focal_plane_camera
# Update the r0 parameter, generate a new phase screen for the atmosphere and combine it with the Telescope
atm.r0 = 0.1
atm.generateNewPhaseScreen(seed = 10)
tel+atm

tel.computePSF(4)
plt.close('all')
    


# combine telescope with atmosphere
tel+atm

# initialize DM commands
atm*ngs*tel*ngs_cam
atm*src*tel*src_cam

plt.show()

nLoop = 500
# allocate memory to save data
SR_NGS                      = np.zeros(nLoop)
SR_SRC                      = np.zeros(nLoop)
total                       = np.zeros(nLoop)
residual_SRC                = np.zeros(nLoop)
residual_NGS                = np.zeros(nLoop)
dm_commands = np.zeros((nLoop, dm.nValidAct))
wfsSignal               = np.arange(0,wfs.nSignal)*0

plot_obj = cl_plot(list_fig          = [atm.OPD,
                                        tel.mean_removed_OPD,
                                        tel.mean_removed_OPD,
                                        [[0,0],[0,0],[0,0]],
                                        wfs.cam.frame,
                                        wfs.focal_plane_camera.frame,
                                        np.log10(tel.PSF),
                                        np.log10(tel.PSF)],
                    type_fig          = ['imshow',
                                        'imshow',
                                        'imshow',
                                        'plot',
                                        'imshow',
                                        'imshow',
                                        'imshow',
                                        'imshow'],
                    list_title        = ['Turbulence [nm]',
                                        'NGS@'+str(ngs.coordinates[0])+'" WFE [nm]',
                                        'SRC@'+str(src.coordinates[0])+'" WFE [nm]',
                                        None,
                                        'WFS Detector',
                                        'WFS Focal Plane Camera',
                                        None,
                                        None],
                    list_legend       = [None,None,None,['SRC@'+str(src.coordinates[0])+'"','NGS@'+str(ngs.coordinates[0])+'"'],None,None,None,None],
                    list_label        = [None,None,None,['Time','WFE [nm]'],None,None,['NGS PSF@'+str(ngs.coordinates[0])+'" -- FOV: '+str(np.round(ngs_cam.fov_arcsec,2)) +'"',''],['SRC PSF@'+str(src.coordinates[0])+'" -- FOV: '+str(np.round(src_cam.fov_arcsec,2)) +'"','']],
                    n_subplot         = [4,2],
                    list_display_axis = [None,None,None,True,None,None,None,None],
                    list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)

# loop parameters
gainCL                  = 0.3
wfs.cam.photonNoise     = False
display                 = True
frame_delay             = 2
reconstructor = M2C_CL@calib_CL.M

for i in range(nLoop):
    a=time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    # atm.update()
    
    atm.update()
    
    # save phase variance
    total[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    # propagate light from the NGS through the atmosphere, telescope, DM to the WFS and NGS camera with the CL commands applied
    atm*ngs*tel*dm*wfs*ngs_cam
    wfs*wfs.focal_plane_camera
    # save residuals corresponding to the NGS
    residual_NGS[i] = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    OPD_NGS         = tel.mean_removed_OPD.copy()

    if display==True:        
        NGS_PSF = np.log10(np.abs(ngs_cam.frame))
    
    # propagate light from the SRC through the atmosphere, telescope, DM to the Instrument camera
    atm*src*tel*dm*src_cam
    dm_commands[i,:] = dm.coefs.copy()
    # save residuals corresponding to the NGS
    residual_SRC[i] = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    OPD_SRC         = tel.mean_removed_OPD.copy()
    if frame_delay ==1:        
        wfsSignal=wfs.signal
    
    # apply the commands on the DM
    dm.coefs=dm.coefs-gainCL*np.matmul(reconstructor,wfsSignal)
    
    # store the slopes after computing the commands => 2 frames delay
    if frame_delay ==2:        
        wfsSignal=wfs.signal
    
    print('Elapsed time: ' + str(time.time()-a) +' s')
    
    # update displays if required
    if display==True and i>0:        
        
        SRC_PSF = np.log10(np.abs(src_cam.frame))
        # update range for PSF images
        plot_obj.list_lim = [None,None,None,None,None,None,[NGS_PSF.max()-3, NGS_PSF.max()],[SRC_PSF.max()-4, SRC_PSF.max()]]        
        # update title
        plot_obj.list_title = ['Turbulence WFE:'+str(np.round(total[i]))+'[nm]',
                               'NGS@'+str(ngs.coordinates[0])+'" WFE:'+str(np.round(residual_NGS[i]))+'[nm]',
                               'SRC@'+str(src.coordinates[0])+'" WFE:'+str(np.round(residual_SRC[i]))+'[nm]',
                                None,
                                'WFS Detector',
                                'WFS Focal Plane Camera',
                                None,
                                None]

        cl_plot(list_fig   = [1e9*atm.OPD,1e9*OPD_NGS,1e9*OPD_SRC,[np.arange(i+1),residual_SRC[:i+1],residual_NGS[:i+1]],wfs.cam.frame,wfs.focal_plane_camera.frame,NGS_PSF, SRC_PSF],
                               plt_obj = plot_obj)
        plt.pause(0.01)
        if plot_obj.keep_going is False:
            break
    print('Loop'+str(i)+'/'+str(nLoop)+' NGS: '+str(residual_NGS[i])+' -- SRC:' +str(residual_SRC[i])+ '\n')


#%%


delta = (dm_commands[21:,:] - dm_commands[20:-1,:])[:80,:]
delta_20 = (dm_commands_20[21:,:] - dm_commands_20[20:-1,:])[:80,:]
delta_10 = (dm_commands_10[21:,:] - dm_commands_10[20:-1,:])[:80,:]




plt.figure()
plt.subplot(131)
plt.imshow(delta@delta.T)
plt.title('Windspeed 30 m/s')

plt.subplot(132)
plt.imshow(delta_20@delta_20.T)
plt.title('Windspeed 20 m/s')


plt.subplot(133)
plt.imshow(delta_10@delta_10.T)
plt.title('Windspeed 10 m/s')


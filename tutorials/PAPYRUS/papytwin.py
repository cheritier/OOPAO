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
from Papyrus import Papyrus

#%% Compute the OOPAO Objects

Papytwin = Papyrus()
# telescope object
tel     = Papytwin.tel
# source object
ngs     = Papytwin.ngs
# deformable mirror object
dm      = Papytwin.dm
# Pyramid WFS object
wfs     = Papytwin.wfs
# atmosphere object
atm     = Papytwin.atm
# slow Tip/Tilt object
slow_tt = Papytwin.slow_tt
# parameter file
param   = Papytwin.param

#%% Function to swith to on-sky pupil (possibility to add an offset for the position of the pupil)
Papytwin.set_pupil(calibration=False,
                   sky_offset=[2,2])

ngs*tel*wfs

plt.figure()
plt.imshow(tel.pupil)
plt.figure()
plt.imshow(wfs.cam.frame)

# set back to calibration pupil
Papytwin.set_pupil(calibration=True)
ngs*tel*wfs


#%% PAPYRUS input data from the bench
from pymatreader import read_mat

M2C = read_mat('M2C_KL_OOPAO_synthetic_IF.mat')['M2C_KL']

valid_pixel = read_mat('useful_pixels_20250604_0305.mat')['usefulPix']

im = read_mat('intMat_klOOPAO_synthetic_bin=1_F=500_rMod=5_20250604_0307.mat')['matrix_inf']

# index of the KL modes included in the int-mat
ind = [1, 5, 10, 20, 30, 50, 80, 100, 150]

int_mat_extract= im[:,ind]

valid_pixel, int_mat_binned = Papytwin.bin_bench_data(valid_pixel = valid_pixel, full_int_mat = im, ratio = param['ratio'])

#%% PAPYRUS/PAPYTWIN Pyramid Pupils Comparison 

var_im = np.var(im,axis=1).reshape(240,240)
var_im/=var_im.max()
var_im = var_im>0.005

# in case there is a mis-match set the key-word "correct" to True
correct = False
Papytwin.check_pwfs_pupils(valid_pixel_map = var_im, correct=correct)


#%% PAPYRUS/PAPYTWIN Interaction Matrix Comparison 

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

a = displayMap(int_mat_extract, norma = True,axis=1,returnOutput=True)
plt.title("Experimental Interaction Matrix")
b = displayMap(calib.D[:,:], norma = True,axis=1,returnOutput=True)
plt.title("Synthetic Interaction Matrix")

a[np.isinf(a)] = 0
b[np.isinf(b)] = 0

from OOPAO.tools.displayTools import interactive_show
interactive_show(a,b) # use right and left click to switch between PAPYRUS and PAPYTWIN


#%% PAPYRUS KL Basis Computation (only for the bench)
compute_kl_basis = False

if compute_kl_basis:
    from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
    M2C = compute_KL_basis(tel = tel,
                           atm = atm,
                           dm  = dm,
                           lim = 1e-3)

#%% PAPYRUS/PAPYTWIN DM/WFS Mis-registration calibration

# Slow if index_modes is long
index_modes = np.arange(10,150,50)
calibrate_mis_registration = False
if calibrate_mis_registration:
    Papytwin.calibrate_mis_registration(M2C = M2C,
                           input_im = int_mat_binned,
                           index_modes = index_modes)


#%% PAPYTWIN Full Interaction Matrix Computation
tel-atm
wfs.modulation = 0

stroke = 0.0001
calib_0 = InteractionMatrix(  ngs            = ngs,
                            atm            = atm,
                            tel            = tel,
                            dm             = dm,
                            wfs            = wfs,
                            M2C            = M2C,
                            stroke         = stroke,
                            phaseOffset    = 0,\
                            nMeasurements  = 1,\
                            noise          = 'off',
                            print_time     = False,
                            display        = True)
    

a = displayMap(im[:,index_modes], norma = True,axis=1,returnOutput=True)
plt.title("Experimental Interaction Matrix")
b = displayMap(calib.D[:,index_modes], norma = True,axis=1,returnOutput=True)
plt.title("Synthetic Interaction Matrix")

a[np.isinf(a)] = 0
b[np.isinf(b)] = 0

from OOPAO.tools.displayTools import interactive_show
interactive_show(a,b)

plt.figure()
plt.plot(np.std(im,axis=0),label='PAPYRUS')
plt.plot(np.std(calib.D,axis=0),label='PAPYTWIN')
plt.legend()
plt.xlabel('KL Mode Index')
plt.ylabel('Int. Mat STD')

#%%  -----------------------     Close loop  ----------------------------------
tel.resetOPD()

end_mode    = 195 
M2C_CL = M2C[:,:end_mode]
# These are the calibration data used to close the loop
# use of experimental calibration
# if full int-mat is available only
calib_CL    = CalibrationVault(calib.D[:,:end_mode])

#%%
tel.resetOPD()
wfs.modulation=0
# wfs.referenceSignal = 0*wfs.referenceSignal
# wfs.referenceSignal_2D = 0*wfs.referenceSignal_2D

#%%




r0 = [0.04,0.06,0.08,0.1,0.12,0.12]


buffer =[]


for i_c in [2]:
    data = []
    if i_c == 0:
        tel.resetOPD()
        wfs.modulation=5
        calib_CL    = CalibrationVault(calib.D[:,:end_mode])
        gainCL                  = 0.3

    if i_c == 1:
        tel.resetOPD()
        wfs.modulation=0
        calib_CL    = CalibrationVault(calib.D[:,:end_mode])
        gainCL                  = 0.1
        wfs.referenceSignal = 0*wfs.referenceSignal
        wfs.referenceSignal_2D = 0*wfs.referenceSignal_2D
    if i_c == 2:
        tel.resetOPD()
        wfs.modulation=0        
        calib_CL    = CalibrationVault(calib_0.D[:,:end_mode])
        gainCL                  = 0.3

        
    
    for i_r0 in r0:
        from OOPAO.Atmosphere import Atmosphere
        
        atm = Atmosphere(telescope      = tel, 
                         r0             = i_r0,
                         L0             = 25, 
                         windSpeed      = [0.01], 
                         fractionalR0   = [1], 
                         windDirection  = [0],
                         altitude       = [0])
        
        atm.initializeAtmosphere(tel)
        
        
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
        
        nLoop = 1000
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
        

        wfs.cam.photonNoise     = False
        display                 = True
        frame_delay             = 2
        reconstructor = M2C_CL@calib_CL.M
        
        for i in range(nLoop):
            a=time.time()
            
            # update phase screens => overwrite tel.OPD and consequently tel.src.phase  
            atm.update()
            # save phase variance
            total[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
            # propagate light from the NGS through the atmosphere, telescope, DM to the WFS and NGS camera with the CL commands applied
            atm*ngs*tel*dm*slow_tt*wfs*ngs_cam
            wfs*wfs.focal_plane_camera
            # save residuals corresponding to the NGS
            residual_NGS[i] = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
            OPD_NGS         = tel.mean_removed_OPD.copy()
        
            if display==True:        
                NGS_PSF = np.log10(np.abs(ngs_cam.frame))
            
            # propagate light from the SRC through the atmosphere, telescope, DM to the Instrument camera
            atm*src*tel*dm*slow_tt*src_cam
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
        
        data.append(residual_NGS)
    buffer.append(data)
    
    
#%%
plt.close('all')
case = ['Calib mod 5 - CL mod 5 - gain 0.3','Calib mod 5 - CL mod 0 - gain 0.1','Calib mod 0 - CL mod 0 - gain 0.3']
for i in range(2):
    tmp = []
    for j in range(len(r0)):
        plt.figure(i)
        plt.plot(buffer[i][j])
        tmp.append(np.mean(buffer[i][j][20:]))
    plt.figure(10)
    plt.plot(r0,tmp,label=case[i])
plt.legend()
plt.xlabel('r0@500 nm [m]')
plt.ylabel('WFE [nm]')






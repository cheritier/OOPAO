# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:34:26 2023

IDEAL version of PAPYRUS AO System used for exploration:
    - 20x20 subapertures instead of 74x74
    - mis-registrations ignored

@author: cheritie - astriffl
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.Pyramid import Pyramid
from OOPAO.Detector import Detector
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot
from papyrus_tools import actuators_position

#%% -----------------------     read parameter file   ----------------------------------

from parameter_files.parameterFile_Papytwin_ideal import initializeParameterFile
from parameter_files.OCAM2K  import OCAM_param
from parameter_files.CRED2   import CRED2_param
from parameter_files.CS165CU import CS165CU_param

param = initializeParameterFile()

#%%
plt.ion()

def printf(line):
    sys.stdout.write(line)
    sys.stdout.flush()

#%% -----------------------     TELESCOPE   ----------------------------------

# create the Telescope object
tel = Telescope(resolution          = param['resolution'],
                diameter            = param['diameter'],
                samplingTime        = param['samplingTime'],
                centralObstruction  = param['centralObstruction'])


#%% -----------------------     Sources   ----------------------------------

star_magnitude_vis = 5   # magnitude of the star
star_magnitude_IR  = 4.5
R2QE           = 2   # photometric band to OCAM quantum efficiency (because working on all visible spectrum)
BS2PyWFS       = 0.5 # beam-splitter between OCAM and GSC

src_vis_calib = Source(optBand   = 'R',   
                       magnitude = -2.5)  # Magnitude to get 80% of saturation on OCAM in calibration @ 500 fps, gain=1
src_vis_calib.nPhoton *= BS2PyWFS

src_vis_sky = Source(optBand   = 'R',
                     magnitude = star_magnitude_vis)
src_vis_sky.nPhoton *= BS2PyWFS * R2QE * param['m1_reflectivity'] # Flux adjusted with respect to the transmittivity of T152+PAPYRUS

src_IR  = Source(optBand   = 'IR1310',
                 magnitude = star_magnitude_IR)

src_vis_calib * tel

#%% -----------------------     ATMOSPHERE   ----------------------------------

atm=Atmosphere(telescope     = tel,
               r0            = param['r0'],
               L0            = param['L0'],
               windSpeed     = param['windSpeed'],
               fractionalR0  = param['fractionnalR0'],
               windDirection = param['windDirection'],
               altitude      = param['altitude'])
    
atm.initializeAtmosphere(tel)
atm.update()

tel-atm

#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------

misReg = MisRegistration(param)
coordinates, pitch = actuators_position(param)

dm=DeformableMirror(telescope    = tel,
                    nSubap       = param['nActuator'],
                    mechCoupling = param['mechanicalCoupling'],
                    misReg       = misReg,
                    coordinates  = coordinates,
                    pitch        = pitch)

tel*dm

#%% -----------------------     PYRAMID WFS   ----------------------------------

wfs = Pyramid(nSubap                = param['nSubaperture'],
              telescope             = tel,
              modulation            = param['modulation'],
              lightRatio            = param['lightThreshold'],
              n_pix_separation      = param['n_pix_separation'],
              n_pix_edge            = param['n_pix_edge'],
              psfCentering          = True,
              postProcessing        = 'slopesMaps')    

OCAM = Detector(nRes            = wfs.cam.resolution,
                integrationTime = tel.samplingTime,
                bits            = OCAM_param['quantization'],
                FWC             = OCAM_param['FWC'],
                gain            = 1,
                sensor          = OCAM_param['sensor'],
                QE              = OCAM_param['QE'],
                binning         = 1,
                psf_sampling    = 2,
                darkCurrent     = OCAM_param['darkCurrent'],
                readoutNoise    = OCAM_param['readoutNoise'],
                photonNoise     = OCAM_param['photonNoise'])

wfs.cam = OCAM
# wfs.binning = wfs.cam.binning

#%% -----------------------     Modal Basis   ----------------------------------

# data = read_mat('D:/astriffl/Documents/05-PAPYRUS/00-DATA/M2C_KL_OOPAO_synthetic_IF.mat')
# M2C = data['M2C_KL']
# stroke = 0.02/4e6

from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
M2C = compute_KL_basis(tel, atm, dm,lim = 1e-3)
M2C = M2C[:,:194]


#%% -----------------------     Calibration    ----------------------------------
stroke = 20e-9 
src_vis_calib * tel
wfs.cam.gain = 1

calib = InteractionMatrix(ngs            = src_vis_calib,
                          atm            = atm,
                          tel            = tel,
                          dm             = dm,
                          wfs            = wfs,
                          M2C            = M2C,
                          stroke         = stroke,
                          nMeasurements  = 10,
                          noise          = 'off',
                          display        = True,
                          single_pass    = True)

# wfs.cam.darkCurrent     = OCAM_param['darkCurrent']
# wfs.cam.readoutNoise    = OCAM_param['readoutNoise']
# wfs.cam.photonNoise     = OCAM_param['photonNoise']


#%%  -----------------------     Detectors    ----------------------------------

# Science imagery camera (will be CRED3 soon...)
CRED2 = Detector(nRes            = CRED2_param['resolution'],
                 integrationTime = 0.006,
                 bits            = CRED2_param['quantization'],
                 FWC             = CRED2_param['FWC'],
                 gain            = 1,
                 sensor          = CRED2_param['sensor'],
                 QE              = CRED2_param['QE'],
                 binning         = 1,
                 psf_sampling    = 3.7,
                 darkCurrent     = CRED2_param['darkCurrent'],
                 readoutNoise    = CRED2_param['readoutNoise'],
                 photonNoise     = CRED2_param['photonNoise'])

#Gain Sensing Camera
CS165CU = Detector(nRes            = 200,
                   integrationTime = 0.006,
                   bits            = CS165CU_param['quantization'],
                   FWC             = CS165CU_param['FWC'],
                   gain            = 1,
                   sensor          = CS165CU_param['sensor'],
                   QE              = CS165CU_param['QE'],
                   binning         = 1,
                   psf_sampling    = 3.5,
                   darkCurrent     = CS165CU_param['darkCurrent'],
                   readoutNoise    = CS165CU_param['readoutNoise'],
                   photonNoise     = CS165CU_param['photonNoise'])


# wfs.focal_plane_camera = CS165CU
# wfs.focal_plane_camera.is_focal_plane_camera = True

CRED2.psf_sampling = 4
CRED2.resolution = 200

#%%  -----------------------     Close-loop    ----------------------------------

n_iter   = 1000 
gain     = 0.3
M2C_CL   = M2C
calib_CL = calib
display = True

plt.close("all")

wfs.cam.gain = 299
wfs.focal_plane_camera.gain = 5

# Init CL
src_vis_sky*tel*dm*wfs
tel.resetOPD()
dm.coefs = 0
wfsSignal = np.zeros(wfs.nSignal)

# Compute reconstructor
reconstructor = M2C_CL @ calib_CL.M

# Coupling tel and atm
tel+atm
atm.windSpeed = [7,7,7,7,7]
atm.generateNewPhaseScreen(1)

# Init variables
SR         = np.zeros(n_iter)
total      = np.zeros(n_iter)
residual   = np.zeros(n_iter)

# Init plot
plot_obj = cl_plot(list_fig          = [atm.OPD*1e9,
                                        tel.OPD*1e9,
                                        [[0,0],[0,0],[0,0]],
                                        wfs.focal_plane_camera.frame,
                                        CS165CU.frame,
                                        CRED2.frame],
                   
                   type_fig          = ['imshow',
                                        'imshow',
                                        'plot',
                                        'imshow',
                                        'imshow',
                                        'imshow'],
                   
                   list_title        = ['Atmosphere [nm]',
                                        'Telescope OPD [nm]',
                                        'Residual [nm RMS]',
                                        'OCAM frame',
                                        'GSC',
                                        'CRED2 frame'],
                   
                   list_legend       = [None,None,['Turbulence','Residuals'],None,None,None],
                   list_label        = [None,None,['Time', 'WFE [nm]'],None,None,None],
                   n_subplot         = [3,2],
                   list_display_axis = [None,None,True,None,None,None],
                   list_ratio        = [[1.05,1.05,0.1],[1,1,1]], s=20)

for i in range(n_iter):
    atm.update()
    
    atm*src_vis_sky*tel*dm*wfs
    wfs*wfs.focal_plane_camera
    
    total[i]=np.std(atm.OPD[np.where(tel.pupil>0)])*1e9
    OPD_vis = tel.mean_removed_OPD.copy()
    residual[i] = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    
    # apply the commands on the DM
    dm.coefs = dm.coefs - gain * (reconstructor @ wfsSignal)
    
    src_IR*tel
    atm*src_IR*tel*dm
    tel*CRED2
    
    SR[i] = np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))*100
    OPD_IR = tel.mean_removed_OPD.copy()
    # store the slopes after computing the commands => 2 frames delay
    wfsSignal=wfs.signal
        
    if display == True:
        plot_obj.list_lim = [None,None,None,None,None,[CRED2.frame.min(), CRED2.frame.max()]]
        
        plot_obj.list_title = ['Turbulence %.2f [nm]'%total[i],
                               'Residual %.2f [nm]'%residual[i],
                               None,
                               'OCAM %.1f %%'%wfs.cam.saturation,
                               'GSC %.1f %%'%wfs.focal_plane_camera.saturation,
                               'CRED2 %.1f %%'%CRED2.saturation]

        cl_plot(list_fig   = [atm.OPD*1e9,
                              OPD_IR*1e9,
                              [np.arange(i+1),total[:i+1],residual[:i+1]],
                              wfs.cam.frame,
                              wfs.focal_plane_camera.frame,
                              CRED2.frame],
                              plt_obj = plot_obj)

    plt.pause(0.002)
    if plot_obj.keep_going is False:
        break
    printf('\r%3d/%3d | SR = %3.1f%% | Res = %4.1f nm '%(i+1,n_iter,SR[i],residual[i]))
    
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:12:50 2024

@author: cheritier
"""

import matplotlib.pyplot as plt
import numpy as np
from OOPAO.tools.displayTools import displayMap

# %%
plt.ion()
# number of subaperture for the WFS
n_subaperture = 20

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% ------------------------     TELESCOPE   -----------------------------------
from OOPAO.Telescope import Telescope

# create the Telescope object
tel = Telescope(resolution           = 6*n_subaperture,                          # resolution of the telescope in [pix]
                diameter             = 1.52,                                        # diameter in [m]        
                samplingTime         = 1/1000,                                   # Sampling time in [s] of the AO loop
                centralObstruction   = 0.25,                                      # Central obstruction in [%] of a diameter 
                display_optical_path = False,                                    # Flag to display optical path
                fov                  = 10 )                                     # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of the phase screens but is uncompatible with off-axis targets

# Apply spiders to the telescope pupil
thickness_spider    = 0.015                                                # thickness of the spiders in m
angle               = [45, 135, 225, 315]                                        # angle in degrees for each spider
offset_Y            = [-0., -0., 0., 0.]                                     # shift offsets for each spider
offset_X            = None

tel.apply_spiders(angle, thickness_spider, offset_X=offset_X, offset_Y=offset_Y)

# display current pupil
plt.figure()
plt.imshow(tel.pupil)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% ------------------------     SOURCES   -----------------------------------
from OOPAO.Source import Source

# create the Natural Guide Star object
ngs = Source(optBand     = 'I',           # Optical band (see photometry.py)
             magnitude   = 8,             # Source Magnitude
             coordinates = [0,0])         # Source coordinated [arcsec,deg]

# combine the NGS to the telescope using '*':
ngs*tel

# create the Scientific Target object located at 10 arcsec from the  ngs
src = Source(optBand     = 'K',           # Optical band (see photometry.py)
             magnitude   = 8,              # Source Magnitude
             coordinates = [0,0])        # Source coordinated [arcsec,deg]

# combine the SRC to the telescope using '*'
src*tel

# check that the ngs and tel.src objects are the same
tel.src.print_properties()

# The Telescope has an OPD property (Optical Path Difference in [m] that is not wavelength dependant) 
plt.figure()
plt.imshow(tel.OPD)
plt.title('Telescope OPD in [m]')

# The Telescope now has a source attached to it with a phase property (in [rad]) that is wavelength dependant:
plt.figure()
plt.imshow(tel.src.phase)
plt.title('Telescope Source Phase in [rad]')

#%%
# compute a PSF using the computePSF method
 
tel.computePSF(zeroPaddingFactor = 6)

log_PSF = np.log10(np.abs(tel.PSF))

plt.figure()
plt.imshow(log_PSF,extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([log_PSF.max()-5, log_PSF.max()])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()
plt.title('Log Scale PSF @'+str(tel.src.wavelength*1e9)+' nm')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#-------------------------     ATMOSPHERE   -----------------------------------
from OOPAO.Atmosphere import Atmosphere
           
# create the Atmosphere object
atm = Atmosphere(telescope     = tel,                               # Telescope                              
                 r0            = 0.05,                              # Fried Parameter [m]
                 L0            = 25,                                # Outer Scale [m]
                 fractionalR0  = [0.45 ,0.1  ,0.1  ,0.25  ,0.1   ], # Cn2 Profile
                 windSpeed     = [10   ,12   ,11   ,15    ,20    ], # Wind Speed in [m]
                 windDirection = [0    ,72   ,144  ,216   ,288   ], # Wind Direction in [degrees]
                 altitude      = [0    ,1000 ,5000 ,10000 ,12000 ]) # Altitude Layers in [m]

# initialize atmosphere with current Telescope
atm.initializeAtmosphere(tel)

# The phase screen can be updated using atm.update method (Temporal sampling given by tel.samplingTime)
atm.update()

# display the atm.OPD = resulting OPD 
plt.figure()
plt.imshow(atm.OPD*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#--------------- PROPAGATE THE LIGHT THROUGH THE ATMOSPHERE -------------------

# The Telescope and Atmosphere can be combined using the '+' operator (Propagation through the atmosphere): 
tel+atm # This operations makes that the tel.OPD is automatically over-written by the value of atm.OPD when atm.OPD is updated. 

# It is possible to print the optical path: 
tel.print_optical_path()

# display the atm.OPD 
plt.figure()
plt.imshow(atm.OPD*1e9)
plt.title('OPD Atmosphere [nm]')
plt.colorbar()

plt.figure()
plt.imshow(tel.OPD*1e9)
plt.title('OPD Telescope [nm]')
plt.colorbar()

plt.figure()
plt.imshow(tel.src.phase)
plt.title('Telescope Source Phase [rad]')
plt.colorbar()

#%% 
# display the atmosphere layers for the sources specified in list_src: 
atm.display_atm_layers(list_src=[ngs,src])

#%%
# the sources coordinates can be updated on the fly: 
src.coordinates = [4,0]

atm.display_atm_layers(list_src=[ngs,src])

#%%
# be careful not to search for a source outside the field of view of the telescope: 
src.coordinates = [40,0]
atm.display_atm_layers(list_src=[ngs,src])
# this will result in an error: 
atm*src*tel

#%% set it back to nominal value
src.coordinates = [0,0]
atm.display_atm_layers(list_src=[ngs,src])
# this will result in an error: 
atm*src*tel

# compute a PSF using the computePSF method

atm*ngs*tel
tel.computePSF(zeroPaddingFactor = 6)

log_PSF = np.log10(np.abs(tel.PSF))

plt.figure()
plt.imshow(log_PSF,extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([log_PSF.max()-5, log_PSF.max()])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()
plt.title('Log Scale PSF @'+str(tel.src.wavelength*1e9)+' nm')

atm*src*tel
tel.computePSF(zeroPaddingFactor = 6)

log_PSF = np.log10(np.abs(tel.PSF))

plt.figure()
plt.imshow(log_PSF,extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([log_PSF.max()-5, log_PSF.max()])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()
plt.title('Log Scale PSF @'+str(tel.src.wavelength*1e9)+' nm')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#--------------------     Scientific Detector   -------------------------------

from OOPAO.Detector import Detector

# define a detector with its properties (see Detector class for further documentation)
cam = Detector(integrationTime = tel.samplingTime,      # integration time of the detector
               photonNoise     = True,                  # enable photon noise
               readoutNoise    = 0,                     # readout of the detector in [e-/pixel]
               QE              = 1,                     # quantum efficiency
               psf_sampling    = 4,                     # sampling for the PSF computation 2 = Shannon sampling
               binning         = 1)                     # Binning factor of the PSF


cam_binned = Detector( integrationTime = tel.samplingTime,      # integration time of the detector
                       photonNoise     = True,                  # enable photon noise
                       readoutNoise    = 2,                     # readout of the detector in [e-/pixel]
                       QE              = 0.8,                   # quantum efficiency
                       psf_sampling    = 2,                     # sampling for the PSF computation 2 = Shannon sampling
                       binning         = 4)                     # Binning factor of the PSF

#%% Propagate to a detector to get a focal plane image

# 1 --- combine to go through atmosphere
tel+atm

# computation of a PSF on the detector using the '*' operator
ngs*tel*cam*cam_binned

plt.figure()
plt.imshow(cam.frame,extent=[-cam.fov_arcsec/2,cam.fov_arcsec/2,-cam.fov_arcsec/2,cam.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam.pixel_size_arcsec,3))+'"')

plt.figure()
plt.imshow(cam_binned.frame,extent=[-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2,-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam_binned.pixel_size_arcsec,3))+'"')

# computation of a PSF on the detector using the '*' operator
atm*ngs*tel*cam*cam_binned

plt.figure()
plt.imshow(cam.frame,extent=[-cam.fov_arcsec/2,cam.fov_arcsec/2,-cam.fov_arcsec/2,cam.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam.pixel_size_arcsec,3))+'"')

plt.figure()
plt.imshow(cam_binned.frame,extent=[-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2,-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam_binned.pixel_size_arcsec,3))+'"')

#%%# 2 --- Separate to have diffraction limited images

# The Telescope and Atmosphere can be separated using the '-' operator (Free space propagation) 
tel-atm
tel.print_optical_path()

# computation of a PSF on the detector using the '*' operator
ngs*tel*cam*cam_binned

plt.figure()
plt.imshow(cam.frame,extent=[-cam.fov_arcsec/2,cam.fov_arcsec/2,-cam.fov_arcsec/2,cam.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam.pixel_size_arcsec,3))+'"')

plt.figure()
plt.imshow(cam_binned.frame,extent=[-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2,-cam_binned.fov_arcsec/2,cam_binned.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam_binned.pixel_size_arcsec,3))+'"')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% -----------------------     DEFORMABLE MIRROR   ----------------------------------
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration

# mis-registrations object (rotation, shifts..) of the DM
misReg = MisRegistration()
misReg.shiftX = 0           # in [m]
misReg.shiftY = 0           # in [m]
misReg.rotationAngle = 0    # in [deg]


# specifying a given number of actuators along the diameter: 
nAct = 16       # This value is the number of actuator in the pupil! So 1 extra actuator is added automatically

dm = DeformableMirror(telescope  = tel,                        # Telescope
                    nSubap       = nAct,                     # number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)
                    mechCoupling = 0.35,                       # Mechanical Coupling for the influence functions
                    misReg       = misReg,                     # Mis-registration associated 
                    coordinates  = None,                       # coordinates in [m]. Should be input as an array of size [n_actuators, 2] 
                    pitch        = tel.D/nAct)                 # inter actuator distance. Only used to compute the influence function coupling. The default is based on the n_subaperture value. 
    

# plot the dm actuators coordinates with respect to the pupil
plt.figure()
plt.imshow(np.reshape(np.sum(dm.modes**7,axis=1),[tel.resolution,tel.resolution]).T + tel.pupil,extent=[-tel.D/2,tel.D/2,-tel.D/2,tel.D/2])
plt.plot(dm.coordinates[:,0],dm.coordinates[:,1],'rx')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates VS Telescope Pupil')


#%% Apply a command on the DM and propagate light through it

dm.coefs = np.random.randn(dm.nValidAct)*300e-9 # random vector for the valid actuators

#% propagate through the DM
ngs*tel*dm*cam

plt.figure()
plt.imshow(dm.OPD)
plt.title('OPD DM [nm]')
plt.colorbar()


plt.figure()
plt.imshow(tel.OPD*1e9)
plt.title('OPD Telescope [nm]')
plt.colorbar()


plt.figure()
plt.imshow(cam.frame,extent=[-cam.fov_arcsec/2,cam.fov_arcsec/2,-cam.fov_arcsec/2,cam.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('PSF @'+str(tel.src.wavelength*1e9)+' nm')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#------------------------     PYRAMID WFS   -----------------------------------
from OOPAO.Pyramid import Pyramid

# make sure tel and atm are separated to initialize the PWFS
tel.isPaired = False
tel.resetOPD()

wfs = Pyramid(nSubap            = n_subaperture,                # number of subaperture = number of pixel accros the pupil diameter
              telescope         = tel,                          # telescope object
              lightRatio        = 0.5,                          # flux threshold to select valid sub-subaperture
              modulation        = 3,                            # Tip tilt modulation radius
              binning           = 1,                            # binning factor (applied only on the )
              n_pix_separation  = 4,                            # number of pixel separating the different pupils
              n_pix_edge        = 2,                            # number of pixel on the edges of the pupils
              postProcessing    = 'fullFrame_camera_flux')  # slopesMaps,

# propagate the light to the Wave-Front Sensor
tel*wfs

plt.close('all')
plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS Camera Frame')
plt.colorbar()

plt.figure()
plt.imshow(wfs.validSignal)
plt.title('WFS Valid Signal')
plt.colorbar()

#%% Apply Noises

# The photon Noise of the detector can be disabled the same way than for a Detector class
wfs.cam.photonNoise = True

ngs*tel*wfs

plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS Camera Frame - With Noise')

wfs.cam.photonNoise = False
ngs*tel*wfs
plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS Camera Frame - Without Noise')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------- Modal Basis - Zernike ---------------------------------
from OOPAO.Zernike import Zernike

#% ZERNIKE Polynomials
# create Zernike Object
Z = Zernike(tel,20)
# compute polynomials for given telescope
Z.computeZernike(tel)

# mode to command matrix to project Zernike Polynomials on DM
M2C_zernike = np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical,:]))@Z.modes

# show the first 10 zernikes applied on the DM
dm.coefs = M2C_zernike[:,:10]
tel*dm
displayMap(tel.OPD)

#%%------------------------ Modal Basis - KL  ---------------------------------

from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
# use the default definition of the KL modes with forced Tip and Tilt. For more complex KL modes, consider the use of the compute_KL_basis function. 
M2C_KL = compute_KL_basis(tel, atm, dm,lim = 1e-2) # matrix to apply modes on the DM

# apply the 10 first KL modes
dm.coefs = M2C_KL[:,:10]
# propagate through the DM
ngs*tel*dm
# show the first 10 KL modes applied on the DM
displayMap(tel.OPD)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% -----------------------     Calibration: Interaction Matrix  ----------------------------------
from OOPAO.calibration.InteractionMatrix import InteractionMatrix

# amplitude of the modes in m
stroke=ngs.wavelength/16
# zonal Interaction Matrix
M2C_zonal = np.eye(dm.nValidAct)

# modal Interaction Matrix for 300 modes
M2C_modal = M2C_KL[:,:300]

tel-atm
# zonal interaction matrix
calib_modal = InteractionMatrix(ngs            = ngs,
                                atm            = atm,
                                tel            = tel,
                                dm             = dm,
                                wfs            = wfs,   
                                M2C            = M2C_modal, # M2C matrix used 
                                stroke         = stroke,    # stroke for the push/pull in M2C units
                                nMeasurements  = 6,        # number of simultaneous measurements
                                noise          = 'off',     # disable wfs.cam noise 
                                display        = True,      # display the time using tqdm
                                single_pass    = True)      # only push to compute the interaction matrix instead of push-pull


#%% Display  interaction matrix signals
from OOPAO.tools.displayTools import display_wfs_signals
N = 9

display_wfs_signals(wfs, calib_modal.D[:,:N])


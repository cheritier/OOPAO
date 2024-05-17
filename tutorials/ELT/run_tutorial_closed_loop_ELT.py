# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:40:49 2023

@author: cheritier
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap
import matplotlib.gridspec as gridspec


# %%
plt.ion()

n_subaperture = 20


#%% -----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope

# create the Telescope object
tel = Telescope(resolution           = 6*n_subaperture,                          # resolution of the telescope in [pix]
                diameter             = 4,                                        # diameter in [m]        
                samplingTime         = 1/1000,                                   # Sampling time in [s] of the AO loop
                centralObstruction   = 0.2,                                      # Central obstruction in [%] of a diameter 
                display_optical_path = True)                                     # Flag to display optical path

# Apply spiders to the telescope pupil

thickness_spider    = 0.05                                                       # thickness of the spiders in m
angle               = [45, 135, 225, 315]                                        # angle in degrees for each spider
offset_Y            = [-0.2, 0.2, 0.2, -0.2]                                     # shift offsets for each spider
offset_X            = None

tel.apply_spiders(angle, thickness_spider, offset_X=offset_X, offset_Y=offset_Y)

# display current pupil
plt.figure()
plt.imshow(tel.pupil)

#%% -----------------------     NGS   ----------------------------------
from OOPAO.Source import Source

# create the Natural Guide Star object
ngs = Source(optBand   = 'I',           # Optical band (see photometry.py)
             magnitude = 4)             # Source Magnitude

# create the Scientific Target object
src = Source(optBand   = 'K',           # Optical band (see photometry.py)
             magnitude = 4)             # Source Magnitude

# combine the NGS to the telescope using '*'
ngs*tel

# check that the ngs and tel.src objects are the same
tel.src.print_properties()

# compute PSF 
tel.computePSF(zeroPaddingFactor = 6)
plt.figure()
plt.imshow(np.log10(np.abs(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([-1,4])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()

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

# The phase screen can be updated using atm.update method (Temporal sampling given by tel.samplingTime)
atm.update()

# display the atm.OPD = resulting OPD 
plt.figure()
plt.imshow(atm.OPD*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()
# display the atmosphere layers

atm.display_atm_layers()





#%%
# The Telescope and Atmosphere can be combined using the '+' operator (Propagation through the atmosphere): 
tel+atm

# This operations makes that the tel.OPD is automatically over-written by the value of atm.OPD when atm.OPD is updated. 
# It is possible to print the optical path: 
tel.print_optical_path()
tel.computePSF(zeroPaddingFactor=4)
plt.figure()
plt.imshow((np.log10(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([-1,4])

plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()

# The Telescope and Atmosphere can be separated using the '-' operator (Free space propagation) 
tel-atm
tel.print_optical_path()

tel.computePSF(zeroPaddingFactor=4)
plt.figure()
plt.imshow((np.log10(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([-1,4])

plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()


#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration

# mis-registrations object
misReg = MisRegistration()
misReg.rotationAngle = 0


# if no coordinates specified, create a cartesian dm that will be in a Fried geometry
dm = DeformableMirror(telescope  = tel,                        # Telescope
                    nSubap       = n_subaperture,              # number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)
                    mechCoupling = 0.35,                       # Mechanical Coupling for the influence functions
                    misReg       = misReg,                     # Mis-registration associated 
                    coordinates  = None,                       # coordinates in [m]. Should be input as an array of size [n_actuators, 2] 
                    pitch        = None)                        # inter actuator distance. Only used to compute the influence function coupling. The default is based on the n_subaperture value. 
    

# specifying a given number of actuators along the diameter: 
nAct = 13
    
dm = DeformableMirror(telescope  = tel,                        # Telescope
                    nSubap       = nAct-1,                     # number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)
                    mechCoupling = 0.35,                       # Mechanical Coupling for the influence functions
                    misReg       = misReg,                     # Mis-registration associated 
                    coordinates  = None,                       # coordinates in [m]. Should be input as an array of size [n_actuators, 2] 
                    pitch        = tel.D/nAct)                        # inter actuator distance. Only used to compute the influence function coupling. The default is based on the n_subaperture value. 
    

# plot the dm actuators coordinates with respect to the pupil

plt.figure()
plt.imshow(np.reshape(np.sum(dm.modes**5,axis=1),[tel.resolution,tel.resolution]).T + tel.pupil,extent=[-tel.D/2,tel.D/2,-tel.D/2,tel.D/2])
plt.plot(dm.coordinates[:,0],dm.coordinates[:,1],'x')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates')

#%% -----------------------     SH WFS   ----------------------------------
from OOPAO.ShackHartmann import ShackHartmann

# make sure tel and atm are separated to initialize the PWFS
tel.isPaired = False

tel.resetOPD()
wfs = ShackHartmann(nSubap          = n_subaperture,        # number of subaperture
              telescope             = tel,                  # telescope object
              lightRatio            = 0.5,                  # flux threshold to select valid sub-subaperture
              binning_factor        = 1,                    # binning factor
              is_geometric          = False,                # Flag to use a geometric shack-hartmann (direct gradient measurement)
              shannon_sampling      = True)                 # Flag to use a shannon sampling for the shack-hartmann spots

# propagate the light to the Wave-Front Sensor
tel*wfs

plt.close('all')
plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS Camera Frame')

#%% 
# The photon Noise can be disabled or enabled setting the wfs.cam.photonNoise flag:
wfs.cam.photonNoise = False

tel*wfs
plt.close('all')
plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS Camera Frame - Without Noise')

wfs.cam.photonNoise = True
tel*wfs
plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS Camera Frame - With Noise')





#%% -----------------------     Modal Basis - Zernike  ----------------------------------
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

#%% -----------------------     Modal Basis - KL Basis  ----------------------------------


from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
# use the default definition of the KL modes with forced Tip and Tilt. For more complex KL modes, consider the use of the compute_KL_basis function. 
M2C_KL = compute_KL_basis(tel, atm, dm)
dm.coefs = M2C_KL[:,:10]
tel*dm
displayMap(tel.OPD)
#%% to manually measure the interaction matrix

# amplitude of the modes in m
stroke=1e-9
# Modal Interaction Matrix

#%%
wfs.is_geometric = True

M2C_zonal = np.eye(dm.nValidAct)
# zonal interaction matrix
calib_zonal = InteractionMatrix(  ngs            = ngs,\
                            atm            = atm,\
                            tel            = tel,\
                            dm             = dm,\
                            wfs            = wfs,\
                            M2C            = M2C_zonal,\
                            stroke         = stroke,\
                            nMeasurements  = 100,\
                            noise          = 'off')

plt.figure()
plt.plot(np.std(calib_zonal.D,axis=0))
plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')

#%%

# Modal interaction matrix
calib_KL = CalibrationVault(calib_zonal.D@M2C_KL)

plt.figure()
plt.plot(np.std(calib_KL.D,axis=0))
plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')

#%% switch to a diffractive SH-WFS

wfs.is_geometric = False

#%%
tel.resetOPD()
# initialize DM commands
dm.coefs=0
ngs*tel*dm*wfs
tel+atm

# dm.coefs[100] = -1

tel.computePSF(4)
plt.close('all')
    
# These are the calibration data used to close the loop
calib_CL    = calib_KL
M2C_CL      = M2C_KL


# combine telescope with atmosphere
tel+atm

# initialize DM commands
dm.coefs=0
ngs*tel*dm*wfs


plt.show()

nLoop = 200
# allocate memory to save data
SR                      = np.zeros(nLoop)
total                   = np.zeros(nLoop)
residual                = np.zeros(nLoop)
wfsSignal               = np.arange(0,wfs.nSignal)*0
SE_PSF = []
LE_PSF = np.log10(tel.PSF_norma_zoom)

plot_obj = cl_plot(list_fig          = [atm.OPD,tel.mean_removed_OPD,wfs.cam.frame,[dm.coordinates[:,0],np.flip(dm.coordinates[:,1]),dm.coefs],[[0,0],[0,0]],np.log10(tel.PSF_norma_zoom),np.log10(tel.PSF_norma_zoom)],\
                   type_fig          = ['imshow','imshow','imshow','scatter','plot','imshow','imshow'],\
                   list_title        = ['Turbulence OPD','Residual OPD','WFS Detector','DM Commands',None,None,None],\
                   list_lim          = [None,None,None,None,None,[-4,0],[-4,0]],\
                   list_label        = [None,None,None,None,['Time','WFE [nm]'],['Short Exposure PSF',''],['Long Exposure_PSF','']],\
                   n_subplot         = [4,2],\
                   list_display_axis = [None,None,None,None,True,None,None],\
                   list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)
# loop parameters
gainCL                  = 0.4
wfs.cam.photonNoise     = True
display                 = True

reconstructor = M2C_CL@calib_CL.M

for i in range(nLoop):
    a=time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    # save phase variance
    total[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    # save turbulent phase
    turbPhase = tel.src.phase
    # propagate to the WFS with the CL commands applied
    ngs*tel*dm*wfs
    src*tel
    tel.print_optical_path()
    dm.coefs=dm.coefs-gainCL*np.matmul(reconstructor,wfsSignal)
    # store the slopes after computing the commands => 2 frames delay
    wfsSignal=wfs.signal
    b= time.time()
    print('Elapsed time: ' + str(b-a) +' s')
    # update displays if required
    if display==True:        
        tel.computePSF(4)
        if i>15:
            SE_PSF.append(np.log10(tel.PSF_norma_zoom))
            LE_PSF = np.mean(SE_PSF, axis=0)
        
        cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD,wfs.cam.frame,dm.coefs,[np.arange(i+1),residual[:i+1]],np.log10(tel.PSF_norma_zoom), LE_PSF],
                               plt_obj = plot_obj)
        plt.pause(0.1)
        if plot_obj.keep_going is False:
            break
    
    SR[i]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    residual[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    OPD=tel.OPD[np.where(tel.pupil>0)]

    print('Loop'+str(i)+'/'+str(nLoop)+' Turbulence: '+str(total[i])+' -- Residual:' +str(residual[i])+ '\n')

#%%
plt.figure()
plt.plot(total)
plt.plot(residual)
plt.xlabel('Time')
plt.ylabel('WFE [nm]')

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:51:32 2020

@author: cheritie
"""
# commom modules
import matplotlib.pyplot as plt
import numpy             as np 
import time

import __load__psim
__load__psim.load_psim()

from AO_modules.Atmosphere       import Atmosphere
from AO_modules.Pyramid         import Pyramid
from AO_modules.DeformableMirror import DeformableMirror
from AO_modules.MisRegistration  import MisRegistration
from AO_modules.Telescope        import Telescope
from AO_modules.Source           import Source
# calibration modules 
from AO_modules.calibration.compute_KL_modal_basis import compute_M2C
from AO_modules.calibration.ao_calibration import ao_calibration
# display modules
from AO_modules.tools.displayTools           import displayMap

#%% -----------------------     read parameter file   ----------------------------------
from parameterFile_VLT_I_Band import initializeParameterFile
param = initializeParameterFile()

#%% -----------------------     TELESCOPE   ----------------------------------

# create the Telescope object
tel = Telescope(resolution          = param['resolution'],\
                diameter            = param['diameter'],\
                samplingTime        = param['samplingTime'],\
                centralObstruction  = param['centralObstruction'])

# create the Source object
ngs=Source(optBand   =  param['opticalBand'],\
           magnitude = param['magnitude'])

# combine the NGS to the telescope using '*' operator:
ngs*tel

tel.computePSF(zeroPaddingFactor = 2)
plt.figure()
plt.imshow(np.log(np.abs(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.colorbar()


#%% -----------------------     ATMOSPHERE   ----------------------------------

atm=Atmosphere(telescope     = tel,\
               r0            = param['r0'],\
               L0            = param['L0'],\
               windSpeed     = param['windSpeed'],\
               fractionalR0  = param['fractionnalR0'],\
               windDirection = param['windDirection'],\
               altitude      = param['altitude'])
# initialize atmosphere
atm.initializeAtmosphere(tel)
    

plt.figure(10)
ax = plt.subplot(1,1,1)
im = ax.imshow(atm.OPD)
plt.colorbar(im) 

#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
# mis-registrations object
misReg = MisRegistration(param)
# if no coordonates specified, create a cartesian dm
dm=DeformableMirror(telescope    = tel,\
                    nSubap       = param['nSubaperture'],\
                    mechCoupling = param['mechanicalCoupling'],\
                    misReg       = misReg)

plt.figure()
plt.plot(dm.coordinates[:,0],dm.coordinates[:,1],'x')
plt.xlabel('[m]')
plt.ylabel('[m]')

#%% -----------------------     PYRAMID WFS   ----------------------------------

# make sure tel and atm are separated to initialize the PWFS
wfs = Pyramid(nSubap                = param['nSubaperture'],\
              telescope             = tel,\
              modulation            = param['modulation'],\
              lightRatio            = param['lightThreshold'],\
              pupilSeparationRatio  = param['pupilSeparationRatio'],\
              calibModulation       = param['calibrationModulation'],\
              psfCentering          = param['psfCentering'],\
              edgePixel             = param['edgePixel'],\
              extraModulationFactor = param['extraModulationFactor'],\
              postProcessing        = param['postProcessing'])


    

from AO_modules.tools.set_paralleling_setup import set_paralleling_setup
set_paralleling_setup(wfs,ELT=False)

#%%
tel-atm
# light propagation to the WFS
tel*wfs
plt.close('all')
plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS Camera Frame')


#%% -----------------------     Modal Basis   ----------------------------------
# compute the modal basis
foldername_M2C  = None  # name of the folder to save the M2C matrix, if None a default name is used 
filename_M2C    = None  # name of the filename, if None a default name is used 


M2C_KL = compute_M2C(telescope            = tel,\
                                 atmosphere         = atm,\
                                 deformableMirror   = dm,\
                                 param              = param,\
                                 nameFolder         = None,\
                                 nameFile           = None,\
                                 remove_piston      = True,\
                                 HHtName            = None,\
                                 baseName           = None ,\
                                 mem_available      = 8.1e9,\
                                 minimF             = False,\
                                 nmo                = 300,\
                                 ortho_spm          = True,\
                                 SZ                 = np.int(2*tel.OPD.shape[0]),\
                                 nZer               = 3,\
                                 NDIVL              = 1)


#%%
ao_calib =  ao_calibration(param            = param,\
                           ngs              = ngs,\
                           tel              = tel,\
                           atm              = atm,\
                           dm               = dm,\
                           wfs              = wfs,\
                           nameFolderIntMat = None,\
                           nameIntMat       = None,\
                           nameFolderBasis  = None,\
                           nameBasis        = None,\
                           nMeasurements    = 1)

#%%

dm.coefs = ao_calib.M2C[:,:10]*1e-10

tel*dm*wfs
from AO_modules.tools.displayTools import displayPyramidSignals


displayPyramidSignals(wfs, wfs.signal)


#%% to manually measure the interaction matrix
#
## amplitude of the modes in m
#stroke=1e-9
## Modal Interaction Matrix 
#M2C = M2C[:,:param['nModes']]
#from AO_modules.calibration.InteractionMatrix import interactionMatrix
#
#calib = interactionMatrix(ngs            = ngs,\
#                             atm            = atm,\
#                             tel            = tel,\
#                             dm             = dm,\
#                             wfs            = wfs,\
#                             M2C            = M2C,\
#                             stroke         = stroke,\
#                             phaseOffset    = 0,\
#                             nMeasurements  = 100,\
#                             noise          = False)
#
#plt.figure()
#plt.plot(np.std(calib.D,axis=0))
#plt.xlabel('Mode Number')
#plt.ylabel('WFS slopes STD')
#plt.ylabel('Optical Gain')


#%% DISPLAY MODAL BASIS AND OPTICAL GAINS ESTIMATION
#

tel-atm
tel.resetOPD()
# project the mode on the DM
dm.coefs = ao_calib.M2C[:,:param['nModes']]

tel*dm
#
# show the modes projected on the dm, cropped by the pupil and normalized by their maximum value
displayMap(tel.OPD,norma=True)
plt.title('Basis projected on the DM')

KL_dm = np.reshape(tel.OPD,[tel.resolution**2,tel.OPD.shape[2]])

covMat = (KL_dm.T @ KL_dm) / tel.resolution**2

plt.figure()
plt.imshow(covMat)
plt.title('Orthogonality')
plt.show()

plt.figure()
plt.plot(np.round(np.std(np.squeeze(KL_dm[tel.pupilLogical,:]),axis = 0),5))
plt.title('KL mode normalization projected on the DM')
plt.show()

projector = np.linalg.pinv(KL_dm)


# get fitting error
from AO_modules.calibration.getFittingError import getFittingError

opd_fitting_2D,opd_corr_2D,opd_turb_2D = getFittingError(OPD     =  atm.OPD,\
                                                         proj    = projector,\
                                                         basis   = KL_dm,\
                                                         display = True)


phi_fitting_2D = opd_fitting_2D*2*np.pi/ngs.wavelength
# compute interaction matrix with an offset
from AO_modules.calibration.InteractionMatrix import interactionMatrix

calib_offset = interactionMatrix(ngs            = ngs,\
                              atm            = atm,\
                              tel            = tel,\
                              dm             = dm,\
                              wfs            = wfs,\
                              M2C            = ao_calib.M2C,\
                              stroke         = 10e-9,\
                              phaseOffset    = phi_fitting_2D,\
                              nMeasurements  = 50,\
                              noise          = 'off')




OG = np.diag(calib_offset.D.T@ao_calib.calib.D)/np.diag(ao_calib.calib.D.T@ao_calib.calib.D)
plt.figure()
plt.plot(OG)

gOpt = np.diag(1/OG)
#%%
N = 10
a = np.abs(np.random.randn(N))
a/=np.sum(a)

param['fractionnalR0'        ] = list(a)                                            # Cn2 profile
param['windSpeed'            ] = list(np.random.randint(10,50,N))                                           # wind speed of the different layers in [m.s-1]
param['windDirection'        ] = list(np.random.randint(1,360,N))                                            # wind direction of the different layers in [degrees]
param['altitude'             ] = list(np.random.randint(1,20,N))  
# create the Atmosphere object
atm=Atmosphere(telescope     = tel,\
               r0            = param['r0'],\
               L0            = param['L0'],\
               windSpeed     = param['windSpeed'],\
               fractionalR0  = param['fractionnalR0'],\
               windDirection = param['windDirection'],\
               altitude      = param['altitude'])    # initialize atmosphere
atm.initializeAtmosphere(tel)


#%%
# These are the calibration data used to close the loop
calib_CL    = ao_calib.calib
M2C_CL      = ao_calib.M2C

plt.close('all')

# combine telescope with atmosphere
tel+atm

# initialize DM commands
dm.coefs=0
ngs*tel*dm*wfs

plt.ion()
# setup the display
fig         = plt.figure(79)
ax1         = plt.subplot(2,3,1)
im_atm      = ax1.imshow(tel.src.phase)
plt.colorbar(im_atm)
plt.title('Turbulence phase [rad]')

ax2         = plt.subplot(2,3,2)
im_dm       = ax2.imshow(dm.OPD*tel.pupil)
plt.colorbar(im_dm)
plt.title('DM phase [rad]')
tel.computePSF(zeroPaddingFactor=6)

ax4         = plt.subplot(2,3,3)
im_PSF_OL   = ax4.imshow(tel.PSF_trunc)
plt.colorbar(im_PSF_OL)
plt.title('OL PSF')


ax3         = plt.subplot(2,3,5)
im_residual = ax3.imshow(tel.src.phase)
plt.colorbar(im_residual)
plt.title('Residual phase [rad]')

ax5         = plt.subplot(2,3,4)
im_wfs_CL   = ax5.imshow(wfs.cam.frame)
plt.colorbar(im_wfs_CL)
plt.title('Pyramid Frame CL')

ax6         = plt.subplot(2,3,6)
im_PSF      = ax6.imshow(tel.PSF_trunc)
plt.colorbar(im_PSF)
plt.title('CL PSF')

plt.show()

# allocate memory to save data
SR                      = np.zeros(param['nLoop'])
total                   = np.zeros(param['nLoop'])
residual                = np.zeros(param['nLoop'])
wfsSignal               = np.arange(0,wfs.nSignal)*0

# loop parameters
gainCL                  = 0.5
wfs.cam.photonNoise     = False
display                 = True

param['nLoop']          = 500


atm.initializeAtmosphere(tel)
for i in range(param['nLoop']):
    a=time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
     # save phase variance
    total[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
     # save turbulent phase
    turbPhase = tel.src.phase
    if display == True:
           # compute the OL PSF and update the display
       tel.computePSF(zeroPaddingFactor=6)
       im_PSF_OL.set_data(np.log(tel.PSF_trunc/tel.PSF_trunc.max()))
       im_PSF_OL.set_clim(vmin=-3,vmax=0)
       
     # propagate to the WFS with the CL commands applied
    tel*dm*wfs
    
     # save the DM OPD shape
    dmOPD=tel.pupil*dm.OPD*2*np.pi/ngs.wavelength
    
    dm.coefs=dm.coefs-gainCL*M2C_CL@gOpt@calib_CL.M@wfsSignal
     # store the slopes after computing the commands => 2 frames delay
    wfsSignal=wfs.signal
    b= time.time()
    print('Elapsed time: ' + str(b-a) +' s')
    # update displays if required
    if display==True:
        
       # Turbulence
       im_atm.set_data(turbPhase)
       im_atm.set_clim(vmin=turbPhase.min(),vmax=turbPhase.max())
       # WFS frame
       C=wfs.cam.frame
       im_wfs_CL.set_data(C)
       im_wfs_CL.set_clim(vmin=C.min(),vmax=C.max())
       # DM OPD
       im_dm.set_data(dmOPD)
       im_dm.set_clim(vmin=dmOPD.min(),vmax=dmOPD.max())
     
       # residual phase
       D=tel.src.phase
       D=D-np.mean(D[tel.pupil])
       im_residual.set_data(D)
       im_residual.set_clim(vmin=D.min(),vmax=D.max()) 
    
       tel.computePSF(zeroPaddingFactor=6)
       im_PSF.set_data(np.log(tel.PSF_trunc/tel.PSF_trunc.max()))
       im_PSF.set_clim(vmin=-4,vmax=0)
       plt.draw()
       plt.show()
       plt.pause(0.001)
    
    
    SR[i]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    residual[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    OPD=tel.OPD[np.where(tel.pupil>0)]

    print('Loop'+str(i)+'/'+str(param['nLoop'])+' Turbulence: '+str(total[i])+' -- Residual:' +str(residual[i])+ '\n')

#%%
plt.figure()
plt.plot(total)
plt.plot(residual)
plt.xlabel('Time')
plt.ylabel('WFE [nm]')

plt.pause(10)

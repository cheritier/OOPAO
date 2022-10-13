# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:51:32 2020

@author: cheritie
"""
# commom modules
import matplotlib.pyplot as plt
import numpy             as np 
import time
plt.ion()
import __load__psim
__load__psim.load_psim()

from AO_modules.Atmosphere       import Atmosphere
from AO_modules.ShackHartmann          import ShackHartmann
from AO_modules.DeformableMirror import DeformableMirror
from AO_modules.MisRegistration  import MisRegistration
from AO_modules.Telescope        import Telescope
from AO_modules.Source           import Source
# calibration modules 
from AO_modules.calibration.compute_KL_modal_basis import compute_M2C
# display modules
from AO_modules.tools.displayTools           import displayMap

#%% -----------------------     read parameter file   ----------------------------------
from parameter_files.parameterFile_VLT_SPHERE_SH_WFS import initializeParameterFile
param = initializeParameterFile()


#%% -----------------------     TELESCOPE   ----------------------------------

# create the Telescope object
tel = Telescope(resolution          = param['resolution'],\
                diameter            = param['diameter'],\
                samplingTime        = param['samplingTime'],\
                centralObstruction  = param['centralObstruction'])


thickness_spider = 0.05                         # size in m
angle            = [45,135,225,315]             # in degrees
offset_X         = [-0.4,0.4,0.4,-0.4]        # shift offset of the spider
offset_Y         = None

tel.apply_spiders(angle, thickness_spider, offset_X = offset_X, offset_Y= offset_Y)
    


plt.figure()
plt.imshow(tel.pupilReflectivity)
# %% -----------------------     NGS   ----------------------------------
# create the Source object
ngs=Source(optBand   = param['opticalBand'],\
           magnitude = param['magnitude'])

# combine the NGS to the telescope using '*' operator:
ngs*tel

tel.computePSF(zeroPaddingFactor = 6)
PSF_diff = tel.PSF/tel.PSF.max()
N = 500

fov_pix = tel.xPSF_arcsec[1]/tel.PSF.shape[0]
fov = N*fov_pix
plt.figure()
plt.imshow(np.log10(PSF_diff[N:-N,N:-N]), extent=(-fov,fov,-fov,fov))
plt.clim([-4.5,0])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')


#%% -----------------------     ATMOSPHERE   ----------------------------------

# create the Atmosphere object
atm=Atmosphere(telescope     = tel,\
               r0            = param['r0'],\
               L0            = param['L0'],\
               windSpeed     = param['windSpeed'],\
               fractionalR0  = param['fractionnalR0'],\
               windDirection = param['windDirection'],\
               altitude      = param['altitude'])
# initialize atmosphere
atm.initializeAtmosphere(tel)

atm.update()

plt.figure()
plt.imshow(atm.OPD*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()


tel+atm
tel.computePSF(8)
plt.figure()
plt.imshow((np.log10(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([-1,3])

plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()

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
plt.title('DM Actuator Coordinates')

#%% -----------------------     SH WFS   ----------------------------------

# make sure tel and atm are separated to initialize the PWFS
tel-atm

wfs = ShackHartmann(nSubap      = param['nSubaperture'],\
              telescope         = tel,\
              lightRatio        = param['lightThreshold'],\
              is_geometric      = param['is_geometric'])

tel*wfs
plt.close('all')

plt.figure()
plt.imshow(wfs.valid_subapertures)
plt.title('WFS Valid Subapertures')
plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS Camera Frame')

#%% -----------------------     Modal Basis   ----------------------------------
# compute the modal basis
foldername_M2C  = None  # name of the folder to save the M2C matrix, if None a default name is used 
filename_M2C    = None  # name of the filename, if None a default name is used 
# KL Modal basis
M2C = compute_M2C(telescope            = tel,\
                                  atmosphere         = atm,\
                                  deformableMirror   = dm,\
                                  param              = param,\
                                  mem_available      = 8.1e9,\
                                  nmo                = 1000,\
                                  nZer               = 3,\
                                  remove_piston = True,\
                                  recompute_cov      = False) # forces to recompute covariance matrix



tel.resetOPD()
# project the mode on the DM
dm.coefs = M2C[:,:50]

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

#%%
from AO_modules.calibration.InteractionMatrix import interactionMatrix
# wfs.is_geometric = False

stroke = 1e-9
# controlling 1000 modes
param['nModes'] = 1000
M2C_KL = np.asarray(M2C[:,:param['nModes']])
# Modal interaction matrix
calib_KL = interactionMatrix(  ngs            = ngs,\
                            atm            = atm,\
                            tel            = tel,\
                            dm             = dm,\
                            wfs            = wfs,\
                            M2C            = M2C_KL,\
                            stroke         = stroke,\
                            nMeasurements  = 200,\
                            noise          = 'off')
wfs.is_geometric = True

calib_KL_geo = interactionMatrix(  ngs            = ngs,\
                            atm            = atm,\
                            tel            = tel,\
                            dm             = dm,\
                            wfs            = wfs,\
                            M2C            = M2C_KL,\
                            stroke         = stroke,\
                            nMeasurements  = 200,\
                            noise          = 'off')
plt.figure()
plt.plot(np.std(calib_KL.D,axis=0))
plt.plot(np.std(calib_KL_geo.D,axis=0))

plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')


#%%
# These are the calibration data used to close the loop
wfs.is_geometric = False

calib_CL    = calib_KL_geo
M2C_CL      = M2C_KL.copy()



plt.close('all')
tel.resetOPD()
tel*dm
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
im_PSF_OL   = ax4.imshow(tel.PSF)
plt.colorbar(im_PSF_OL)
plt.title('OL PSF')


ax3         = plt.subplot(2,3,5)
im_residual = ax3.imshow(tel.src.phase)
plt.colorbar(im_residual)
plt.title('Residual phase [rad]')

ax5         = plt.subplot(2,3,4)
im_wfs_CL   = ax5.imshow(wfs.cam.frame)
plt.colorbar(im_wfs_CL)
plt.title('SH Frame CL')

ax6         = plt.subplot(2,3,6)
im_PSF      = ax6.imshow(tel.PSF)
plt.colorbar(im_PSF)
plt.title('CL PSF')

plt.show()

param['nLoop'] = 1000
# allocate memory to save data
SR                      = np.zeros(param['nLoop'])
total                   = np.zeros(param['nLoop'])
residual                = np.zeros(param['nLoop'])
wfsSignal               = np.arange(0,wfs.nSignal)*0

# loop parameters
gainCL                  = 0.4
wfs.cam.photonNoise     = True
display                 = True

reconstructor = M2C_CL@calib_CL.M
PSF_LE = []
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
       tel.computePSF(zeroPaddingFactor=2)
       im_PSF_OL.set_data(np.log(tel.PSF/tel.PSF.max()))
       im_PSF_OL.set_clim(vmin=-3,vmax=0)
       
     # propagate to the WFS with the CL commands applied
    tel*dm*wfs
    
     # save the DM OPD shape
    dmOPD=tel.pupil*dm.OPD*2*np.pi/ngs.wavelength
    
    dm.coefs=dm.coefs-gainCL*np.matmul(reconstructor,wfsSignal)
     # store the slopes after computing the commands => 2 frames delay
    wfsSignal=wfs.signal
    b= time.time()
    print('Elapsed time: ' + str(b-a) +' s')
    tel.computePSF(zeroPaddingFactor=2)
    if i>10:
        PSF_LE.append(tel.PSF)

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
    
       im_PSF.set_data(np.log(tel.PSF/tel.PSF.max()))
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
plt.xlabel('Time [ms]')
plt.ylabel('WFE [nm]')

plt.figure()
PSF = np.mean(PSF_LE,axis=0)
PSF/=PSF.max()
N = 200
fov_pix = tel.xPSF_arcsec[1]/PSF.shape[0]
fov = N*fov_pix
plt.imshow(np.log10(PSF[N:-N,N:-N]), extent=(-fov,fov,-fov,fov))
plt.clim([-4.5,0])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')


tel.resetOPD()
tel.computePSF(6)
PSF_diff = tel.PSF/tel.PSF.max()
plt.figure()
plt.imshow(np.log10(PSF_diff[N:-N,N:-N]), extent=(-fov,fov,-fov,fov))
plt.clim([-4.5,0])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')

plt.figure()
plt.plot(residual)
plt.xlabel('Time')
plt.ylabel('WFE [nm]')

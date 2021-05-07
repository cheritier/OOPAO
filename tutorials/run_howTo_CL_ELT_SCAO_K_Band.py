# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:33:01 2020

@author: cheritie
"""

import matplotlib.pyplot as plt
import numpy             as np 
from astropy.io import fits as pfits
import time

import __load__psim
__load__psim.load_psim()

# local modules 
from AO_modules.Telescope         import Telescope
from AO_modules.Source            import Source
from AO_modules.Atmosphere        import Atmosphere
from AO_modules.Pyramid           import Pyramid
from AO_modules.DeformableMirror  import DeformableMirror

from AO_modules.calibration.compute_KL_modal_basis import compute_M2C
from AO_modules.calibration.ao_calibration import ao_calibration

from AO_modules.MisRegistration   import MisRegistration

# ELT modules
from AO_modules.M1_model.make_ELT_pupil             import generateEeltPupilReflectivity
from AO_modules.M4_model.make_M4_influenceFunctions import getPetalModes

#%% -----------------------     read parameter file   ----------------------------------

from parameterFile_ELT_SCAO_K_Band_3000_KL   import initializeParameterFile

param = initializeParameterFile()

# remove interactive plotting (tutorial code)
plt.ioff()


#%% -----------------------     TELESCOPE   ----------------------------------
# create the pupil of the ELT
M1_pupil_reflectivity = generateEeltPupilReflectivity(refl = param['m1_reflectivityy'],\
                                          npt       = param['resolution'],\
                                          dspider   = param['spiderDiameter'],\
                                          i0        = param['m1_center'][0]+param['m1_shiftX'] ,\
                                          j0        = param['m1_center'][1]+param['m1_shiftY'] ,\
                                          pixscale  = param['pixelSize'],\
                                          gap       = param['gapSize'],\
                                          rotdegree = param['m1_rotationAngle'],\
                                          softGap   = True)

# extract the valid pixels
M1_pupil = M1_pupil_reflectivity>0

# create the Telescope object
tel = Telescope(resolution          = param['resolution'],\
                diameter            = param['diameter'],\
                samplingTime        = param['samplingTime'],\
                centralObstruction  = param['centralObstruction'],\
                pupilReflectivity   = M1_pupil_reflectivity,\
                pupil               = M1_pupil)

#%% -----------------------     NGS   ----------------------------------
# create the Source object
ngs=Source(optBand   = param['opticalBand'],\
           magnitude = param['magnitude'])

# combine the NGS to the telescope using '*' operator:
ngs*tel

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

#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------


# mis-registrations object reading the param
misReg = MisRegistration(param)

# M4 is genererated already projected in the M1 space
dm = DeformableMirror(telescope    = tel,\
                    nSubap       = param['nSubaperture'],\
                    misReg       = misReg,\
                    M4_param     = param)
                    
#%% Display M4 model vs pupil

m4_sum_cube = np.reshape(np.sum(dm.modes**3, axis =1),[tel.resolution,tel.resolution]) 

plt.figure()
plt.imshow(m4_sum_cube.T*tel.pupil.T)
plt.show()

#%%
try:
    petals,petals_float = getPetalModes(tel,dm,[1,2,3,4,5,6])
except:
    petals,petals_float = getPetalModes(tel,dm,[1])
tel.index_pixel_petals = petals
tel.isPetalFree =False

#%% -----------------------     PYRAMID WFS   ----------------------------------

# make sure tel and atm are separated to initialize the PWFS
tel-atm

# create the Pyramid Object
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

# propagate the light through the WFS
tel*wfs

plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS frame')

plt.figure()
plt.imshow(wfs.referenceSignal_2D)
plt.colorbar()
plt.title('Reference Slopes-Maps')
plt.show()
#%% -----------------------    MODAL BASIS   ----------------------------------

# compute the initial modal basis
foldername_M2C  = None  # name of the folder to save the M2C matrix, if None a default name is used 
filename_M2C    = None  # name of the filename, if None a default name is used 
#
#
M2C = compute_M2C(  telescope          = tel,\
                  	atmosphere         = atm,\
                    deformableMirror   = dm,\
                    param              = param,\
                    nameFolder         = None,\
                    nameFile           = None,\
                    remove_piston      = True,\
                    HHtName            = 'covariance_matrix_HHt_tutorial',\
                    baseName           = 'KL_piston_tip_tilt_minimized_forces_tutorial' ,\
                    minimF             = True,\
                    nmo                = 4300,\
                    ortho_spm          = True,\
                    nZer               = 3)
#
## compute the modal basis that minimizes the forces of M4


#%% -----------------------    ZONAL INTERACTION MATRIX   ----------------------------------
from AO_modules.calibration.compute_zonal_interactionMatrix_ELT import computeZonalInteractionMatrix

foldername_intMat      = None          # folder where to save the data. If None a default location is used.
filename_intMat        = None          # name of the file containing the intMat. if None a default name is used
extra           = ''                   # extra name to concatenate to the name of the file. ex: '_no_offset'
offset          = 0                    # phase offset in [rad] used to measure the interaction matrix
amplitude       = 1e-9                 # amplitude of the modes in m for the intMat measurement
n               = 100                  # number of modes measured at the same time

# computation of the interaction matrix
#calibZon = computeZonalInteractionMatrix( ngs               = ngs,\
#                                          atm               = atm,\
#                                          tel               = tel,\
#                                          dm                = dm ,\
#                                          wfs               = wfs,\
#                                          param             = param,\
#                                          nameFolder        = foldername_intMat,\
#                                          nameFile          = filename_intMat,\
#                                          nameExtra         = extra,\
#                                          phaseOffset       = 0,\
#                                          amplitude         = amplitude,\
#                                          nMeasurements     = n)

# The interaction matrix is saved as calibZon.D
# The inverse of the interaction matrix is saved as calibZon.M




#%% -----------------------    (General function) INTERACTION MATRIX   ----------------------------------
# this function can also be used but the previous one is recommanded

#from AO_modules.calibration.InteractionMatrix import interactionMatrix
## more or less the same function with the differences:
##   _ it doesn't save the matrices computed
##   _ it computes the interaction matrix for any modal basis, not only zonal ones
## general way to compute an interaction matrix
#calib = interactionMatrix(ngs           = ngs,\
#                          atm           = atm,\
#                          tel           = tel,\
#                          dm            = dm,\
#                          wfs           = wfs,\
#                          M2C           = M2C,\
#                          stroke        = amplitude,\
#                          phaseOffset   = 0,\
#                          nMeasurements = 100)
#%% -----------------------    CALIBRATION OBJECT   ----------------------------------
# create a calibration object: 
# _ locate and load the modal basis (filename and foldername can be user defined)
# _ locate and load the zonal interaction matrix (filename and foldername can be user defined). If the file does not exist, it computes the zonal interaction matrix. 
# _ project the zonal interaction matrix on the modal basis and invert it to get the modal reconstructor.
# _ locate and loa the modal gains corresponding to param['r0']. If the file is not found it keeps the gains at 1. 

# the output is a class containing all the object: 
#   ao_calib.M2C  -> Mode to Command Matrix#
#   ao_calib.calib.D    -> modal interaction matrix
#   ao_calib.calib.M    -> modal reconstructor
#   ao_calib.gOpt       -> modal gains
#   ao_calib.basis      -> modal basis in [m]
#   ao_calib.projector  -> pseudo inverse of the modal basis to get modal coefficients

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
                           nMeasurements    = 100)      # number of modes measured per cycle in the intMat computation



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


#%% Display modal basis information
plt.iof()
from AO_modules.tools.displayTools import displayMap
#
# project the mode on the DM
dm.coefs = ao_calib.M2C[:,:100]

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


from AO_modules.calibration.getFittingError import getFittingError
tel+atm
getFittingError(tel.OPD,proj =ao_calib.projector, basis = ao_calib.basis , display = True)

plt.show()


#%% -----------------------    CLOSED LOOP   ----------------------------------

M2C_CL              = ao_calib.M2C
ModalGainsMatrix    = ao_calib.gOpt
reconstructor       = np.matmul(M2C_CL,ao_calib.calib.M)          # reconstructor matrix wfsSignals to dm commands

# combine with atmosphere
tel+atm

# initialize dm commands
dm.coefs = 0
ngs*tel*dm*wfs
# ----------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------
# set up the closed loop parameters
gainCL    = param['gainCL']
wfsSignal = np.arange(0,wfs.nSignal)*0
nLoop     = 500

SR       = np.zeros(nLoop)
total    = np.zeros(nLoop)
residual = np.zeros(nLoop)

wfs.cam.photonNoise = True
display             = True

plt.show()


for i_loop in range(nLoop):
     a = time.time()
     # update phase screens
     atm.update()
     # save phase variance
     total[i_loop]   = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
     # save turbulent phase
     turbPhase  = tel.src.phase   
     # propagate to the WFS with the CL commands applied
     tel*dm*wfs
    # save the dm phase
     dmOPD      = tel.pupil*dm.OPD*2*np.pi/ngs.wavelength
     # dm open loop commands
     openLoopCommands   = np.matmul(ao_calib.calib.M,wfsSignal)
     # application of the modal gains and projection on dm influence functions 
     closedLoopCommands = np.matmul(M2C_CL,np.matmul(ModalGainsMatrix,openLoopCommands))
     # apply new commands on the dm
     dm.coefs= dm.coefs-gainCL*closedLoopCommands
     # case petal free 
     if tel.isPetalFree:
        dm.OPD = tel.removePetalling(dm.OPD)
     # store the slopes after computing the commands => 2 frames delay
     wfsSignal=wfs.pyramidSignal
    
     # update displays if required
     if display==True:
         # Turbulence
         im_atm.set_data(turbPhase)
         im_atm.set_clim(vmin=turbPhase.min(),vmax=turbPhase.max())
         # WFS frame
         C=wfs.cam.frame
         im_wfs_CL.set_data(C)
         im_wfs_CL.set_clim(vmin=C.min(),vmax=C.max())
         # dm OPD
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
         plt.pause(0.00001)

     SR[i_loop]       = np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))         
     residual[i_loop] = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    
     print('Loop'+str(i_loop)+'/'+str(nLoop)+'Turbulence: '+str(total[i_loop])+' -- Residual:' +str(residual[i_loop])+ ' -- SR ' + str(SR[i_loop])+'\n')
     b = time.time()
     print('Elapsed Time: ' + str(b-a) + ' s ')


plt.figure()
plt.plot(residual)
plt.xlabel('Time')
plt.ylabel('WFE [nm]')
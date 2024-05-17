# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:51:32 2020

@author: cheritie
"""

import time

import __load__oopao
__load__oopao.load_oopao()
import matplotlib.pyplot as plt
import numpy as np

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.BioEdge import BioEdge
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap
# %% -----------------------     read parameter file   ----------------------------------
from parameterFile_VLT_I_Band_PWFS import initializeParameterFile

param = initializeParameterFile()

# %%
plt.ion()

#%% -----------------------     TELESCOPE   ----------------------------------

# create the Telescope object
tel = Telescope(resolution          = param['resolution'],\
                diameter            = param['diameter'],\
                samplingTime        = param['samplingTime'],\
                centralObstruction  = param['centralObstruction'])

#%% -----------------------     NGS   ----------------------------------
# create the Source object
ngs=Source(optBand   = param['opticalBand'],\
           magnitude = param['magnitude'])

# combine the NGS to the telescope using '*' operator:
ngs*tel

tel.computePSF(zeroPaddingFactor = 6)
plt.figure()
plt.imshow(np.log10(np.abs(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([-1,3])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()

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

#%% -----------------------     PYRAMID WFS   ----------------------------------

param['postProcessing'] = 'fullFrame_incidence_flux'
bio = BioEdge(nSubap                = param['nSubaperture'],\
              telescope             = tel,\
              modulation            = 5,\
              grey_width            = 0,\
              lightRatio            = param['lightThreshold'],\
              n_pix_separation      = 0,\
              psfCentering          = True,\
              postProcessing        = param['postProcessing'],calibModulation=50)


#%%
from OOPAO.tools.tools import compute_fourier_mode

bio.modulation = 5

F = compute_fourier_mode(pupil = tel.pupil, spatial_frequency = 12, angle_deg = -45)

tel.OPD = F*tel.pupil*100e-9

tel*bio

plt.figure()
plt.imshow(bio.signal_2D)

FP = np.abs(np.asarray(bio.modulation_camera_em))**2



index = np.arange(0,128,4)

# compute_gif(np.log10(np.asarray(FP))[index,:,:], 'bio_edge',fps = 8,vlim = [5,8])



#%%

PSF =[]
N = 80
for i in range(32):
    tmp = np.abs(np.asarray(FP)[index[i],:,:])**2
    
    tmp/=tmp.max()
    
    tmp = (100*(tmp[N:-N,N:-N]))
    
    
    tmp2 = frame[i].copy()
    
    tmp2[80:,:80] =  frame[i][:80,:80]
    tmp2[:80,:80] =  frame[i][80:,:80]
    tmp2[:80,:80] =  frame[i][80:,80:] 
    tmp2[80:,80:] =  frame[i][:80,80:]
    data =tmp+tmp2
    data[:,80] = np.nan
    # data[80,:] = np.nan
    
    
    

    PSF.append(data)



compute_gif(np.asarray(PSF), 'bio_edge',fps = 8,vlim =[-1,1])




#%%
bio.modulation = 0
frame = []
FP = []

for i in range(32):
    tel.OPD = F*tel.pupil*50e-9 + bio_tt[i,:,:]*ngs.wavelength/2/np.pi
    tel*bio
    frame.append(bio.signal_2D/bio.signal_2D.max())
    
    # FP.append(bio.get_modulation_frame())
    


#%%

from OOPAO.tools.displayTools import compute_gif




compute_gif(np.asarray(frame), 'bio_edge',fps = 8)
    
    


#%% -----------------------     Modal Basis   ----------------------------------
# compute the modal basis
# foldername_M2C  = None  # name of the folder to save the M2C matrix, if None a default name is used 
# filename_M2C    = None  # name of the filename, if None a default name is used 
# # KL Modal basis
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
M2C_KL = compute_M2C(telescope            = tel,\
                                  atmosphere         = atm,\
                                  deformableMirror   = dm,\
                                  param              = param,\
                                  nameFolder         = None,\
                                  nameFile           = None,\
                                  remove_piston      = True,\
                                  HHtName            = None,\
                                  baseName           = None ,\
                                  mem_available      = 6.1e9,\
                                  minimF             = False,\
                                  nmo                = 1100,\
                                  ortho_spm          = True,\
                                  SZ                 = np.int(2*tel.OPD.shape[0]),\
                                  nZer               = 3,\
                                  NDIVL              = 1)

#%% to manually measure the interaction matrix

# amplitude of the modes in m
stroke=1e-9
# Modal Interaction Matrix

#%%
M2C_zonal = np.eye(dm.nValidAct)
# zonal interaction matrix
calib_bio = InteractionMatrix(  ngs            = ngs,\
                            atm            = atm,\
                            tel            = tel,\
                            dm             = dm,\
                            wfs            = bio,\
                            M2C            = M2C_KL,\
                            stroke         = stroke,\
                            nMeasurements  = 50,\
                            noise          = 'off')
    
plt.figure()
plt.plot(np.std(32*calib_bio.D*stroke,axis=0))
plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')
plt.ylim([0,1])


#%%

tel.resetOPD()
# initialize DM commands
dm.coefs=0
ngs*tel*dm*bio
tel+atm

# dm.coefs[100] = -1

tel.computePSF(4)
plt.close('all')
    
# These are the calibration data used to close the loop
calib_CL    = calib_bio
M2C_CL      = M2C_KL


# combine telescope with atmosphere
tel+atm

# initialize DM commands
dm.coefs=0
ngs*tel*dm*bio


plt.show()

param['nLoop'] = 200
# allocate memory to save data
SR                      = np.zeros(param['nLoop'])
total                   = np.zeros(param['nLoop'])
residual                = np.zeros(param['nLoop'])
bioSignal               = np.arange(0,bio.nSignal)*0
SE_PSF = []
LE_PSF = np.log10(tel.PSF_norma_zoom)

plot_obj = cl_plot(list_fig          = [atm.OPD,tel.mean_removed_OPD,bio.cam.frame,[[0,0],[0,0]],[dm.coordinates[:,0],np.flip(dm.coordinates[:,1]),dm.coefs],np.log10(tel.PSF_norma_zoom),np.log10(tel.PSF_norma_zoom)],\
                   type_fig          = ['imshow','imshow','imshow','plot','scatter','imshow','imshow'],\
                   list_title        = ['Turbulence OPD','Residual OPD','bio Detector',None,None,None,None],\
                   list_lim          = [None,None,None,None,None,[-4,0],[-4,0]],\
                   list_label        = [None,None,None,None,['Time','WFE [nm]'],['DM Commands',''],['Short Exposure PSF',''],['Long Exposure_PSF','']],\
                   n_subplot         = [4,2],\
                   list_display_axis = [None,None,None,True,None,None,None],\
                   list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)
# loop parameters
gainCL                  = 0.4
bio.cam.photonNoise     = True
display                 = True

reconstructor = M2C_CL@calib_CL.M

for i in range(param['nLoop']):
    a=time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    # save phase variance
    total[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    # save turbulent phase
    turbPhase = tel.src.phase
    # propagate to the bio with the CL commands applied
    tel*dm*bio
        
    dm.coefs=dm.coefs-gainCL*np.matmul(reconstructor,bioSignal)
    # store the slopes after computing the commands => 2 frames delay
    bioSignal=bio.signal
    b= time.time()
    print('Elapsed time: ' + str(b-a) +' s')
    # update displays if required
    if display==True:        
        tel.computePSF(4)
        if i>15:
            SE_PSF.append(np.log10(tel.PSF_norma_zoom))
            LE_PSF = np.mean(SE_PSF, axis=0)
        
        cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD,bio.cam.frame,[np.arange(i+1),residual[:i+1]],dm.coefs,np.log10(tel.PSF_norma_zoom), LE_PSF],
                               plt_obj = plot_obj)
        plt.pause(0.1)
        if plot_obj.keep_going is False:
            break
    
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


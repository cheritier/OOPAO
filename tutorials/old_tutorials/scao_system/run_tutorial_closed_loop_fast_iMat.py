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

#%% -----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope
from OOPAO.tools.tools import zero_pad_array

n_subaperture = 20
n_pix_subap = 4

# create a temporary Telescope object
tel_tmp = Telescope(resolution       = n_pix_subap*(n_subaperture),                          # resolution of the telescope in [pix]
                diameter             = 10,                                        # diameter in [m]        
                samplingTime         = 1/1000,                                   # Sampling time in [s] of the AO loop
                centralObstruction   = 0.)                                      # Central obstruction in [%] of a diameter 

# create the Telescope object with zeros on the edges 
n_extra_subap = 0

tel = Telescope(resolution           = n_pix_subap*(n_subaperture + n_extra_subap),                          # resolution of the telescope in [pix]
                diameter             = 10,                                        # diameter in [m]        
                samplingTime         = 1/1000,                                   # Sampling time in [s] of the AO loop
                centralObstruction   = 0.,                                      # Central obstruction in [%] of a diameter 
                pupil = zero_pad_array(tel_tmp.pupil, n_extra_subap*n_pix_subap/2))                                     # Flag to display optical path

# display current pupil
plt.figure()
plt.imshow(tel.pupil)



#%%






#%% -----------------------     NGS   ----------------------------------
from OOPAO.Source import Source

# create the Natural Guide Star object
ngs = Source(optBand   = 'I',           # Optical band (see photometry.py)
             magnitude = 4)             # Source Magnitude

# combine the NGS to the telescope using '*'
ngs*tel

# create the Natural Guide Star object
src = Source(optBand   = 'I',           # Optical band (see photometry.py)
             magnitude = 4)             # Source Magnitude

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

#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration

# mis-registrations object
misReg = MisRegistration()
misReg.rotationAngle = 0


# if no coordinates specified, create a cartesian dm that will be in a Fried geometry
dm = DeformableMirror(telescope  = tel,                        # Telescope
                    nSubap       = n_subaperture+n_extra_subap,# number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)
                    mechCoupling = 0.35,                       # Mechanical Coupling for the influence functions
                    misReg       = misReg,                     # Mis-registration associated 
                    coordinates  = None,                       # coordinates in [m]. Should be input as an array of size [n_actuators, 2] 
                    pitch        = None)                        # inter actuator distance. Only used to compute the influence function coupling. The default is based on the n_subaperture value. 
    

# plot the dm actuators coordinates with respect to the pupil

plt.figure()
plt.imshow(np.reshape(np.sum(dm.modes**5,axis=1),[tel.resolution,tel.resolution]).T + tel.pupil,extent=[-tel.D/2,tel.D/2,-tel.D/2,tel.D/2])
plt.plot(dm.coordinates[:,0],dm.coordinates[:,1],'x')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates')

#%% -----------------------     Modal Basis - KL Basis  ----------------------------------

from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
# use the default definition of the KL modes with forced Tip and Tilt. For more complex KL modes, consider the use of the compute_KL_basis function. 
M2C_KL = compute_KL_basis(tel, atm, dm)
dm.coefs = M2C_KL[:,:10]
tel*dm
displayMap(tel.OPD)


#%%
from OOPAO.Pyramid import Pyramid

wfs = Pyramid(nSubap                = n_subaperture+n_extra_subap,\
              telescope             = tel,\
              modulation            = 3,\
              lightRatio            = 0.1,\
              n_pix_separation      = 8,\
              n_pix_edge            = None,
              psfCentering          = False,\
              postProcessing        = 'fullFrame')
#%%
plt.figure()
plt.imshow(wfs.cam.frame)    
#%%



tel-atm
[X,Y] = np.meshgrid(np.linspace(-wfs.nRes//2,wfs.nRes//2 , wfs.nRes ,endpoint =False), np.linspace(-wfs.nRes//2 , wfs.nRes//2,wfs.nRes,endpoint =False))



mod_path = np.asarray(wfs.modulation_path)

mask = -np.pi/2 * (abs(X) + abs(Y))

plt.figure(),plt.imshow(mask%2)

buffer_mask = []
for i_mod in mod_path:
    X0 = i_mod[0] * wfs.zeroPaddingFactor
    Y0 = i_mod[1] * wfs.zeroPaddingFactor
    mask = -np.pi/2 * (abs(X-X0) + abs(Y-Y0))
    buffer_mask.append(np.fft.fftshift(np.exp(1j*mask)))



# displayMap(np.asarray(buffer_mask), axis =0)

support = wfs.supportPadded.copy()
a = time.time()

support[wfs.center-wfs.telescope.resolution//2:wfs.center+wfs.telescope.resolution//2,wfs.center-wfs.telescope.resolution//2:wfs.center+wfs.telescope.resolution//2] = wfs.maskAmplitude * np.exp(1j*tel.src.phase)

FFT_support = np.fft.fft2(support)

buffer_images =[]


for i in range(len(mod_path)):
    tmp = np.fft.fftshift(np.fft.ifft2(FFT_support*buffer_mask[i]))
    
    buffer_images.append(np.abs(tmp)**2)
    
b = time.time()

print(b-a)
    
displayMap(np.asarray(buffer_images), axis =0)
    

plt.figure(),plt.imshow(np.sum(buffer_images,axis=0))


tel*wfs

plt.figure(),plt.imshow(wfs.cam.frame)



#%%
from joblib import Parallel, delayed
import time

def fast_tt(mask_in):
    tmp = np.fft.fftshift(np.fft.ifft2(FFT_support*mask_in))
    return (np.abs(tmp)**2)
    
a = time.time()
support[wfs.center-wfs.telescope.resolution//2:wfs.center+wfs.telescope.resolution//2,wfs.center-wfs.telescope.resolution//2:wfs.center+wfs.telescope.resolution//2] = wfs.maskAmplitude * np.exp(1j*tel.src.phase)

FFT_support = np.fft.fft2(support)

#define the parallel jobs
def job_loop_multiple_tt():
    Q = Parallel(n_jobs=15,prefer='threads')(delayed(fast_tt)(i) for i in buffer_mask)
    return Q 

# apply the pyramid transform in parallel
buffer_images = job_loop_multiple_tt()

output_image = np.sum(buffer_images,axis=0)

b = time.time()

print(b-a)

a = time.time()

tel*wfs
b = time.time()

print(b-a)


plt.figure(), plt.imshow(output_image)

#%% to manually measure the interaction matrix
from OOPAO.tools.interpolateGeometricalTransformation import interpolate_cube_special

# amplitude of the modes in m
stroke=1e-9
# Modal Interaction Matrix


def fast_imat(wfs):
    
    dm_coord            = dm.coordinates

    dm_tmp = DeformableMirror(telescope     = tel,\
                    nSubap                  = wfs.nSubap,\
                    mechCoupling            = dm.mechCoupling,\
                    coordinates             = dm_coord*0,\
                    misReg                  = None,\
                    pitch                   = dm.pitch)
        
    wfs_tmp = Pyramid(nSubap            = wfs.nSubap,\
                  telescope             = wfs.telescope,\
                  modulation            = wfs.modulation,\
                  lightRatio            = 0,\
                  n_pix_separation      = wfs.n_pix_separation,\
                  psfCentering          = wfs.psfCentering,\
                  postProcessing        = 'fullFrame', 
                  userValidSignal       = np.ones((wfs.cam.resolution,wfs.cam.resolution),int))
        
    M2C_zonal       = np.eye(dm.nValidAct)
    tel-atm
    
    dm_tmp.coefs    = M2C_zonal[:,dm_tmp.nValidAct//2]*1e-9
    tel*dm_tmp*wfs_tmp*wfs
    push            = wfs.cam.frame
    ps              = wfs.signal
    
    dm_tmp.coefs    = -M2C_zonal[:,dm_tmp.nValidAct//2]*1e-9
    tel*dm_tmp*wfs_tmp*wfs
    pull            = wfs.cam.frame
    pl              = wfs.signal

    test_frame      = (push-pull)/2/1e-9
    signal_ref      = (ps-pl)/2 / 1e-9
    
    cube_in         = np.tile(test_frame[:,:,None],dm.nValidAct).T
    m               = MisRegistration()
    delta           = (dm.coordinates-dm_tmp.coordinates)
    pixel_size_in   = tel.D/wfs.nSubap
    pixel_size_out  = tel.D/wfs.nSubap
    resolution_out  = wfs.cam.resolution
    cube_out        = interpolate_cube_special(cube_in,delta[:,1],delta[:,0], pixel_size_in, pixel_size_out, resolution_out,mis_registration=m)
    
    imat = []
    
    if wfs.validSignal.shape[0] != wfs.validSignal.shape[1]:
        # case slopesmaps
        validMap = wfs.validI4Q.copy()
    else:
        # case intensity maps
        validMap = 1    
        
    for i_act in range(dm.nValidAct):
            tmp = np.squeeze(cube_out[i_act,:,:])    
            if wfs.validSignal.shape[0] == wfs.validSignal.shape[1]:
                signal = tmp[np.where(wfs.validSignal==1)]/wfs.norma
            else:
                I1  = wfs.grabQuadrant(1,cameraFrame=tmp)*validMap
                I2  = wfs.grabQuadrant(2,cameraFrame=tmp)*validMap
                I3  = wfs.grabQuadrant(3,cameraFrame=tmp)*validMap
                I4  = wfs.grabQuadrant(4,cameraFrame=tmp)*validMap
            
                # slopesMaps computation cropped to the valid pixels
                Sx         = (I1-I2+I4-I3)            
                Sy         = (I1-I4+I2-I3) 

                # 2D slopes maps      
                slopesMaps = (np.concatenate((Sx,Sy)))/wfs.norma
                # slopes vector
                signal     = slopesMaps[np.where(wfs.validSignal==1)]
                
            imat.append(signal)
            
    return np.asarray(imat).T, wfs_tmp.signal_2D, signal_ref

imat , ref_2D , signal_ref = fast_imat(wfs)


plt.figure()
plt.imshow(ref_2D)
plt.colorbar()

from OOPAO.tools.displayTools import display_wfs_signals

display_wfs_signals(wfs,signal_ref)
plt.colorbar()

display_wfs_signals(wfs,imat[:,100])
plt.colorbar()



#%%
# Modal interaction matrix
calib_KL = CalibrationVault(imat@M2C_KL[:,:300])

plt.figure()
plt.plot(np.std(calib_KL.D,axis=0))
plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')





#%%
wfs.cam.photonNoise = 0
tel.resetOPD()

dm.coefs = M2C_KL[:,10] *200e-9

NCPA = -dm.OPD *tel.pupil

# wfs.referenceSignal = wfs.referenceSignal*0
# wfs.referenceSignal_2D = wfs.referenceSignal_2D*0


tel*dm*wfs

plt.figure(), plt.imshow(wfs.cam.frame)


# wfs.referenceSignal = wfs.signal.copy()
# wfs.referenceSignal_2D = wfs.signal_2D.copy()

offset_1D = wfs.signal.copy()
offset_2D = wfs.signal_2D.copy()



tel.resetOPD()


tel*wfs

plt.figure(), plt.imshow(wfs.cam.frame)
plt.figure(), plt.imshow(wfs.signal_2D)


#%%
from convolutive_model import convolutive_model

CM = convolutive_model(tel,dm,wfs,M2C_CL)

CM.compute_OG_constants()


#%%
tel+atm
tel*wfs
plt.figure(), plt.plot(CM.optical_gains(wfs.get_modulation_frame(radius =10000)))


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
M2C_CL      = M2C_KL[:,:300]


# combine telescope with atmosphere
tel+atm

# initialize DM commands
dm.coefs=0
ngs*tel*dm*wfs


src = Source('K', 0)

from OOPAO.OPD_map import OPD_map


static_OPD = OPD_map(tel, OPD =NCPA)

plt.show()

nLoop = 200
# allocate memory to save data
SR                      = np.zeros(nLoop)
total                   = np.zeros(nLoop)
residual                = np.zeros(nLoop)
wfsSignal               = np.arange(0,wfs.nSignal)*0
SE_PSF = []
LE_PSF = np.log10(tel.PSF_norma_zoom)
SE_PSF_K = []

plot_obj = cl_plot(list_fig          = [atm.OPD,tel.mean_removed_OPD,wfs.cam.frame,np.log10(wfs.get_modulation_frame(radius =10)),[[0,0],[0,0]],[dm.coordinates[:,0],np.flip(dm.coordinates[:,1]),dm.coefs],np.log10(tel.PSF_norma_zoom),np.log10(tel.PSF_norma_zoom)],\
                   type_fig          = ['imshow','imshow','imshow','imshow','plot','scatter','imshow','imshow'],\
                   list_title        = ['Turbulence OPD [m]','Residual OPD [m]','WFS Detector Plane','WFS Focal Plane',None,None,None,None],\
                   list_lim          = [None,None,None,[-3,0],None,None,[-4,0],[-5,0]],\
                   list_label        = [None,None,None,None,['Time [ms]','WFE [nm]'],['DM Commands',''],['Short Exposure I Band PSF',''],['Long Exposure K Band PSF','']],\
                   n_subplot         = [4,2],\
                   list_display_axis = [None,None,None,None,True,None,None,None],\
                   list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)
# loop parameters
gainCL                  = 0.6
wfs.cam.photonNoise     = True
display                 = True

reconstructor = calib_CL.M

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
    
    OG = np.diag(1./CM.optical_gains(wfs.get_modulation_frame(radius =40)))
    # OG = np.eye(300)    
    dm.coefs = dm.coefs-gainCL*M2C_CL@OG@np.matmul(calib_CL.M,wfsSignal-0*offset_1D) 
    
    
    
    
    
    # store the slopes after computing the commands => 2 frames delay
    wfsSignal=wfs.signal
    b= time.time()
    
    
    print('Elapsed time: ' + str(b-a) +' s')
    # update displays if required
    if display==True:        
        tel.computePSF(4)
        SE_PSF.append(np.log10(tel.PSF_norma_zoom))

        if i>15:
            src*tel*static_OPD

            tel.computePSF(4)
            SE_PSF_K.append(np.log10(tel.PSF_norma_zoom))

            LE_PSF = np.mean(SE_PSF_K, axis=0)
        
        cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD,wfs.cam.frame,np.log10(wfs.get_modulation_frame(radius=10)),[np.arange(i+1),residual[:i+1]],dm.coefs,(SE_PSF[-1]), LE_PSF],
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

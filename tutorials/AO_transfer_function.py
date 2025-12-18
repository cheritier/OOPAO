# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:58:58 2024

@author: cheritier
"""

# commom modules
import matplotlib.pyplot as plt
import numpy             as np 
import time

from OOPAO.Atmosphere       import Atmosphere
from OOPAO.Pyramid          import Pyramid
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration  import MisRegistration
from OOPAO.Telescope        import Telescope
from OOPAO.Source           import Source
# calibration modules 
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
from OOPAO.calibration.ao_calibration import ao_calibration
# display modules
from OOPAO.tools.displayTools           import displayMap

import time

import matplotlib.pyplot as plt
import numpy as np

from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap
import matplotlib.gridspec as gridspec


# number of subaperture for the WFS
n_subaperture = 20


#%% -----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope

# create the Telescope object
tel = Telescope(resolution           = 8*n_subaperture,                          # resolution of the telescope in [pix]
                diameter             = 4,                                        # diameter in [m]        
                samplingTime         = 1/1000,                                   # Sampling time in [s] of the AO loop
                centralObstruction   = 0.1,                                      # Central obstruction in [%] of a diameter 
                display_optical_path = False,                                    # Flag to display optical path
                fov                  = 0 )                                     # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of the phase screens but is uncompatible with off-axis targets

# display current pupil
plt.figure()
plt.imshow(tel.pupil)

#%% -----------------------     NGS   ----------------------------------
from OOPAO.Source import Source

# create the Natural Guide Star object
ngs = Source(optBand     = 'I',           # Optical band (see photometry.py)
             magnitude   = 8,             # Source Magnitude
             coordinates = [0,0])         # Source coordinated [arcsec,deg]

# create the Scientific Target object located at 10 arcsec from the  ngs
src = Source(optBand     = 'K',           # Optical band (see photometry.py)
             magnitude   = 8,              # Source Magnitude
             coordinates = [0,0])        # Source coordinated [arcsec,deg]

# combine the NGS to the telescope using '*'
src*tel

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



#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration

# if no coordinates specified, create a cartesian dm that will be in a Fried geometry
dm = DeformableMirror(telescope  = tel,                        # Telescope
                    nSubap       = n_subaperture,              # number of subaperture of the system considered (by default the DM has n_subaperture + 1 actuators to be in a Fried Geometry)
                    mechCoupling = 0.35,                       # Mechanical Coupling for the influence functions
                    misReg       = None,                     # Mis-registration associated 
                    coordinates  = None,                       # coordinates in [m]. Should be input as an array of size [n_actuators, 2] 
                    pitch        = None)                        # inter actuator distance. Only used to compute the influence function coupling. The default is based on the n_subaperture value. 
    


# plot the dm actuators coordinates with respect to the pupil

plt.figure()
plt.imshow(np.reshape(np.sum(dm.modes**5,axis=1),[tel.resolution,tel.resolution]).T + tel.pupil,extent=[-tel.D/2,tel.D/2,-tel.D/2,tel.D/2])
plt.plot(dm.coordinates[:,0],dm.coordinates[:,1],'rx')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates')
#%%

from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis

M2C = compute_KL_basis(tel, atm, dm,lim=0)

tel.resetOPD()
dm.coefs = M2C  
ngs**tel*dm
KL_DM = tel.OPD.reshape(tel.resolution**2,M2C.shape[1])
covmat = KL_DM.T@KL_DM
projector = np.diag(1/np.diag(covmat))@KL_DM.T



#%%


# create the Atmosphere object
atm=Atmosphere(telescope     = tel,\
               r0            = 0.15,\
               L0            = 25,\
               windSpeed     = [10],\
               fractionalR0  = [1],\
               windDirection = [0],\
               altitude      = [100],)
               # initialize atmosphere
atm.initializeAtmosphere(tel)





#%% 1-- PERFECT WFS (See comment to do it with simulated WFS)

N = 256*8+20

mod_coeffs_turb = np.zeros([N,M2C.shape[1],3])
mod_coeffs_res  = np.zeros([N,M2C.shape[1],3])

for i_g in range(2):
    atm.generateNewPhaseScreen(10)

    tel+atm
    phase_res           = np.zeros(tel.resolution**2)
    phase_corr          = np.zeros(tel.resolution**2)
    phase_wfs_delayed   = np.zeros(tel.resolution**2)
    
     # try different gain values
    gain_cl = 0.2+0.4*i_g
    
    for i in range(N):
        print(str(i)+'/'+str(N))
        atm.update()
        
        phase_turb = np.reshape(atm.OPD, tel.resolution**2)
                
        mod_coeffs_turb[i,:,i_g] = projector@phase_turb
        
        # tel*dm*wfs
        phase_wfs = np.copy(phase_turb)+phase_corr.copy()
        # phase_wfs = np.reshape(tel.OPD, tel.resolution**2)
    
        phase_corr = np.reshape(phase_corr - gain_cl*phase_wfs_delayed, tel.resolution**2)
        # dm.coefs -= gain_cl*ao_calib.M2C@ao_calib.calib.M@wfs_signal
    
        phase_wfs_delayed = phase_wfs
        # wfs_signal = wfs.signal
    
        mod_coeffs_res[i,:,i_g] = projector@phase_wfs

#%%
start = 20
end = N

plt.figure()
plt.loglog(1e9*np.std(mod_coeffs_turb[start:end,:,0],axis = 0),label = 'Turbulence')
plt.loglog(1e9*np.std(mod_coeffs_res[start:end,:,0],axis = 0),label = 'Loop Gain 0.2')
plt.loglog(1e9*np.std(mod_coeffs_res[start:end,:,1],axis = 0),label = 'Loop Gain 0.6')

plt.xlabel('KL Modes')
plt.ylabel('WF [nm]')
plt.legend()

#%% plot theoretical trsnfer functions

def TF_Her_Hcl_Hol_Hn(fp,loop_gain,Ti,Tau,Tdm):
    dfp = fp[1]-fp[0]
    I=1j
    S = I*2.*np.pi*fp
    #H_WFS = (1.-np.exp(-S*Ti)) / (S*Ti)
    H_WFS = np.exp(-1.*Ti/2*S) #; like in simulation
    H_RTC = np.exp(-1.*Tau*S)
    H_DM = np.exp(-Tdm*S)
    H_DAC = (1.-np.exp(-S*Ti)) / (S*Ti)
    CC=loop_gain / (1-np.exp(-S*Ti))
    H_OL = H_WFS*H_RTC*H_DAC*H_DM*CC
    H_CL = H_OL/(1+H_OL)
    H_ER = 1./(1.+H_OL)
    H_N=H_CL/H_WFS
    return H_ER,H_CL, H_OL,H_N


from scipy import signal 
plt.close('all')

out = np.arange(1,100000)


for i in range(3):
    fs = 300
    samplingTime = 1/fs
    
    f, Poi = signal.csd(out,out,fs,nperseg=4*256)
    
    
    loop_gain   = np.round(0.2 + 0.4*i,1) 
    fp          = (f.copy())[1:]
    Ti          = samplingTime
    Tau         = samplingTime/2
    Tdm         = samplingTime/2
    
    
    Her,Hcl,Hol,Hn = TF_Her_Hcl_Hol_Hn(fp,loop_gain,Ti,Tau,Tdm)
    
        
    Her_dB =20.*np.log10(np.abs(Her))
    plt.figure(10)

    plt.semilogx(f[1:],Her_dB,label='Gain '+str(loop_gain))
plt.semilogx(f[1:],Her_dB*0,'k')
    
# plt.grid('both')

plt.grid(which = 'minor')
plt.grid(which = 'major')

plt.legend()

plt.xlabel('Frequency [Hz]')
plt.ylabel(r'20 log($\frac{input}{output}$) [dB]')

#%%
plt.close('all')

for i_g in range(2):
        
    from scipy import signal 
    out = mod_coeffs_res[:,:,i_g].T
    inp = mod_coeffs_turb[:,:,i_g].T
    fs = 1/tel.samplingTime
    end = N
    start =20
    mod_test = [0,99,299]
    from OOPAO.tools.displayTools import makeSquareAxes
    for i_mode in range(3):
        mode = mod_test[i_mode]    
        f, Poi = signal.csd(out[mode,start:end],inp[mode,start:end],fs,nperseg=8*128)
        f, Pii = signal.csd(inp[mode,start:end],inp[mode,start:end],fs,nperseg=8*128)
        
        loop_gain   = np.round(0.2+0.4*i_g,1)
        fp          = (f.copy())[1:]
        Ti          = 1*tel.samplingTime
        Tau         = 1*tel.samplingTime/2
        Tdm         = 1*tel.samplingTime/2
        
        Her,Hcl,Hol,Hn = TF_Her_Hcl_Hol_Hn(fp,loop_gain,Ti,Tau,Tdm)
        
        Her_dB =20.*np.log10(np.abs(Her))
        FTR = 20.*np.log10(np.abs(Poi/Pii))
        DSP = 20.*np.log10(np.abs(Pii))
        
        plt.figure(10)
        plt.subplot(1,3,i_mode+1)
        plt.semilogx(f[1:],(FTR[1:]),linewidth=2,label='loop gain '+str(loop_gain)+' -- OOPAO')
        plt.semilogx(f[1:],Her_dB,'--',alpha = 0.8,linewidth=2,label='loop gain '+str(loop_gain)+' -- Theory')

        plt.title('Rejection Transfer Function')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel(r'20 log($\frac{\Phi_{res}}{\Phi_{turb}}$) [dB]')
        plt.ylim([-50,50])
        makeSquareAxes(plt.gca())

        makeSquareAxes(plt.gca())
plt.grid(which='both')
plt.legend()
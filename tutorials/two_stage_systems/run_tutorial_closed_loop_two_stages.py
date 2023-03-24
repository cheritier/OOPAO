# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:30:14 2023

@author: cheritier

This is a tutorial for two-stages AO systems. 
Two 'AO objects' are defined with their own telescope, atmosphere, source, wave-front sensor and deformable mirror to simulate the two stages. 
Then the loop is closed for each stage in a cascaded fashion.

"""

import matplotlib.pyplot as plt

# commom modules
import numpy as np 



#%% -----------------------     read parameter file stage 1  ----------------------------------
from parameterFile_stage_1                  import initializeParameterFile
# initialize parameter file
param_stage_1 = initializeParameterFile()

from parameterFile_stage_2                  import initializeParameterFile
# initialize parameter file
param_stage_2 = initializeParameterFile()

from OOPAO.calibration.initialization_AO_PWFS import run_initialization_AO_PWFS
# generate an AO obj_stage_1ect
obj_stage_1 = run_initialization_AO_PWFS(param_stage_1)


#%% -----------------------    ----------------------------------

# speed factor between first and second stage (integer value only)
speed_factor = int(param_stage_1['samplingTime']/param_stage_2['samplingTime'])

from OOPAO.calibration.get_fast_atmosphere import get_fast_atmosphere

# compute a faster atmosphere corresponding to the speed_factor value
obj_stage_1.tel_fast,obj_stage_1.atm_fast = get_fast_atmosphere(obj_stage_1, param=param_stage_1, speed_factor= speed_factor)


#%%


from parameterFile_stage_2                  import initializeParameterFile
# initialize parameter file
param_stage_2 = initializeParameterFile()

# generate an AO obj_stage_1ect
obj_stage_2 = run_initialization_AO_PWFS(param_stage_2)

#%% UPDATE ATMOSPHERE PARAMETERS

# initialize a random state for repeatibility of the random generator
randomState    = np.random.RandomState(42)

# get the number of layers in the atmosphere
n_layer = obj_stage_1.atm_fast.nLayer

# choose the wind speed of the layers between two values (input must be a list)
param_stage_1['windSpeed']      = list(randomState.randint(10,20,n_layer))                                            # wind speed of the different layers in [m.s-1]

# update the atmosphere wind speed
obj_stage_1.atm_fast.windSpeed  = param_stage_1['windSpeed']

# initialize the atmosphere
obj_stage_1.atm_fast.initializeAtmosphere(obj_stage_1.tel_fast)

obj_stage_1.atm_fast.generateNewPhaseScreen(seed=1)

# select the number of frames to compute
param_stage_1['nLoop'         ] = 2000                                           # number of iteration                             

#%% -----------------   RUN THE TWO STAGES CLOSED LOOP -----------------------------
from OOPAO.closed_loop.run_cl_first_stage      import run_cl_first_stage

# disable/enable the display of the phase screens
obj_stage_1.display = False

# name of the output file -- if set to None automatically genmerated from the parameter file
filename = None

# extra name added at the end of the filename
extra_name = ''
# name of the output folder -- if set to None automatically genmerated from the parameter file
OPD_buffer = []
nLoop = 1000
import time

wfs_signals             = np.zeros([nLoop,obj_stage_1.wfs.nSignal])
dm_commands             = np.zeros([nLoop,obj_stage_1.dm.nValidAct])
residuals_first_stage            = np.zeros(nLoop)
residuals_second_stage            = np.zeros(nLoop)

ao_turbulence           = np.zeros(nLoop)
residual_phase_screen   = np.zeros([nLoop,obj_stage_1.tel.resolution,obj_stage_1.tel.resolution],dtype=np.float32)

reconstructor_first_stage = obj_stage_1.M2C@obj_stage_1.calib.M

wfs_signal_first_stage = obj_stage_1.wfs.signal*0
wfs_signal_second_stage  = obj_stage_2.wfs.signal*0
reconstructor_second_stage = obj_stage_2.M2C@obj_stage_2.calib.M



obj_stage_1.tel_fast + obj_stage_1.atm_fast

obj_stage_1.tel.resetOPD()
obj_stage_2.tel.resetOPD()

obj_stage_1.tel.isPaired = True
obj_stage_2.tel.isPaired = True

obj_stage_1.dm.coefs = 0
obj_stage_1.tel.src * obj_stage_1.tel * obj_stage_1.dm * obj_stage_1.wfs

obj_stage_2.dm.coefs = 0
obj_stage_2.tel.src * obj_stage_2.tel * obj_stage_2.dm * obj_stage_2.wfs


get_le_psf = True
PSF_LE_1 = []
PSF_LE_2 = []

display = True

if display:
    from OOPAO.tools.displayTools import cl_plot

    plot_obj = cl_plot(list_fig          = [obj_stage_1.atm_fast.OPD,obj_stage_1.atm_fast.OPD,obj_stage_1.atm_fast.OPD],\
                       type_fig          = ['imshow','imshow','imshow'],\
                       list_title        = ['Input OPD -- First Stage [m]','Residual OPD -- First Stage [m]','Residual OPD -- Second Stage [m]'],\
                       list_lim          = [None,None,None],\
                       list_label        = [None,None,None],\
                       n_subplot         = [3,1],\
                       list_display_axis = [None,None,None],\
                       list_ratio        = [[0.95,0.1],[1,1,1]])

gain_cl_first_stage = 0.7
gain_cl_second_stage = 0.7

from OOPAO.OPD_map import OPD_map
opd = OPD_map(telescope=obj_stage_2.tel)


for i_loop in range(nLoop):
    
    # update atmospheric phase screen
    obj_stage_1.atm_fast.update()
    
    # skipping the first phase screen
    if i_loop>0:
        	OPD_buffer.append(obj_stage_1.atm_fast.OPD_no_pupil)
    
    # save phase variance of atmosphere
    ao_turbulence[i_loop]=np.std(obj_stage_1.atm_fast.OPD[np.where(obj_stage_1.tel.pupil>0)])*1e9
    
    # save turbulent OPD of the first stage
    OPD_first_stage_in = obj_stage_1.tel_fast.OPD.copy()
    
    # Propagate to the First stage at the first stage rate using a bufer of OPD
    if len(OPD_buffer)==speed_factor:    
        
        OPD_first_stage                 = np.mean(OPD_buffer,axis=0)
        obj_stage_1.tel.OPD_no_pupil    = OPD_first_stage.copy()
        obj_stage_1.tel.OPD             = OPD_first_stage.copy()*obj_stage_1.tel.pupil            

        # propagate to the WFS with the CL commands applied
        obj_stage_1.tel*obj_stage_1.dm*obj_stage_1.wfs
        
        OPD_first_stage_res = obj_stage_1.tel.OPD.copy()
        
        # Reinitialize the OPD buffer
        OPD_buffer = []
        
        # Apply the commands of the first stage dm
        obj_stage_1.dm.coefs        = obj_stage_1.dm.coefs-gain_cl_first_stage*np.matmul(reconstructor_first_stage,wfs_signal_first_stage)
        
        # Assign value of the wfs signal to a buffer to simulate two frames delay
        wfs_signal_first_stage      = obj_stage_1.wfs.signal

    # Propagate through the dm to get the residual at the rate of the second stage    
    obj_stage_1.tel_fast*obj_stage_1.dm

    # save residuals of the first stage
    residuals_first_stage[i_loop]=np.std(obj_stage_1.tel_fast.OPD[np.where(obj_stage_1.tel.pupil>0)])*1e9

    # save residuals OPD of the first stage    
    OPD_first_stage_out = obj_stage_1.tel_fast.OPD.copy()
    
    # assign the residuals OPD of the first stage to a static phase screen
    opd.OPD = OPD_first_stage_out

    # reset OPD of the second stage telescope  (necessary to go through the static phase screen)
    obj_stage_2.tel.resetOPD()
    
    # Propagate to the second stage telescope 
    obj_stage_2.tel.src * obj_stage_2.tel * opd
    
    # save input OPD of the second stage    
    OPD_second_stage_in = obj_stage_2.tel.OPD.copy()

    # Propagate to the second stage wfs 
    obj_stage_2.tel*obj_stage_2.dm*obj_stage_2.wfs
    
    # Apply the commands of the second stage dm
    obj_stage_2.dm.coefs=obj_stage_2.dm.coefs-gain_cl_second_stage*np.matmul(reconstructor_second_stage,wfs_signal_second_stage)
    
    # Assign value of the wfs signal to a buffer to simulate two frames delay
    wfs_signal_second_stage      = obj_stage_2.wfs.signal.copy()

    # save residual OPD of the second stage    
    OPD_second_stage_out = obj_stage_2.tel.OPD.copy()

    # save residuals OPD of the second stage    
    residuals_second_stage[i_loop]=np.std(obj_stage_2.tel.OPD[np.where(obj_stage_2.tel.pupil>0)])*1e9


    if get_le_psf:
        if i_loop >100:
            obj_stage_1.tel.computePSF(zeroPaddingFactor = 4)
            PSF_LE_1.append(obj_stage_1.tel.PSF)
            obj_stage_2.tel.computePSF(zeroPaddingFactor = 4)
            PSF_LE_2.append(obj_stage_2.tel.PSF)

    if display==True:
        cl_plot(list_fig = [OPD_first_stage_in,OPD_first_stage_out,OPD_second_stage_out],plt_obj=plot_obj)
        plt.pause(0.01)
        if plot_obj.keep_going is False:
            break
    print('---------------------------------------------- Loop'+str(i_loop)+'/'+str(nLoop)+ '-------------------------------------------------------')
    print('{: ^30s}'.format('Turbulence ')                  + '{: ^18s}'.format(str(np.round(ao_turbulence[i_loop],2)))           +'{: ^18s}'.format('[nm]'))
    print('{: ^30s}'.format('Residual First Stage')         + '{: ^18s}'.format(str(np.round(residuals_first_stage[i_loop],2)))   +'{: ^18s}'.format('[nm]'))
    print('{: ^30s}'.format('Residual Second Stage')        + '{: ^18s}'.format(str(np.round(residuals_second_stage[i_loop],2)))   +'{: ^18s}'.format('[nm]'))



#%% check the long exposure psf if requested

# get the long exposure PSF
le_psf = np.mean(np.asarray(PSF_LE_1),axis =0)

# normalize 
le_psf /= le_psf.max()

plt.figure(),
plt.subplot(1,2,1)
plt.imshow(np.log10(le_psf),extent = [obj_stage_1.tel.xPSF_arcsec[0],obj_stage_1.tel.xPSF_arcsec[1],obj_stage_1.tel.xPSF_arcsec[0],obj_stage_1.tel.xPSF_arcsec[1]])
plt.clim([-4,0])    
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')

# get the long exposure PSF
le_psf = np.mean(np.asarray(PSF_LE_2),axis =0)

# normalize 
le_psf /= le_psf.max()
plt.subplot(1,2,2)

plt.imshow(np.log10(le_psf),extent = [obj_stage_1.tel.xPSF_arcsec[0],obj_stage_1.tel.xPSF_arcsec[1],obj_stage_1.tel.xPSF_arcsec[0],obj_stage_1.tel.xPSF_arcsec[1]])
plt.clim([-4,0])    
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')

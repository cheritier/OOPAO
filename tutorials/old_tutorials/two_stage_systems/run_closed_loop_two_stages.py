# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 12:01:20 2022

@author: cheritie

This is a tutorial for two-stages AO systems using two functions: 
    _ run_cl_first_stage        : function to perform closed-loop simulation temporally sampled at a given rate (defined by the tel_fast object) with an AO correction applied at a lower temporal rate (defined by the tel_fast object) 
    _ run_cl_from_phase_screens : function to perform closed-loop simulation using phase-screens as an input. In this case, the residual phase screens of the first stage ar considered 
    
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
param_stage_1['nLoop'         ] = 400                                           # number of iteration                             

#%% -----------------   RUN THE TWO STAGES CLOSED LOOP -----------------------------
from OOPAO.closed_loop.run_cl_first_stage      import run_cl_first_stage

# disable/enable the display of the phase screens
obj_stage_1.display = True

# name of the output file -- if set to None automatically genmerated from the parameter file
filename = None

# extra name added at the end of the filename
extra_name = ''
# name of the output folder -- if set to None automatically genmerated from the parameter file

output = run_cl_first_stage(param_stage_1, obj_stage_1,\
                           speed_factor             = speed_factor,\
                           filename_phase_screen    = filename,\
                           extra_name               = extra_name,\
                           destination_folder       = None,\
                           get_le_psf               = True)


#%% check the long exposure psf if requested

# get the long exposure PSF
le_psf = output['long_exposure_psf']

# normalize 
le_psf /= le_psf.max()

plt.figure(),
plt.imshow(np.log10(le_psf),extent = [obj_stage_1.tel.xPSF_arcsec[0],obj_stage_1.tel.xPSF_arcsec[1],obj_stage_1.tel.xPSF_arcsec[0],obj_stage_1.tel.xPSF_arcsec[1]])
plt.clim([-4,0])    
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')


#%%
plt.figure(1)
ax = plt.subplot(1,1,1)
im = ax.imshow(output['residual_phase_screen'][0,:,:])
plt.colorbar(im)
# replay the phase screens
for i in range(param_stage_1['nLoop']):
    tmp = output['residual_phase_screen'][i,:,:]
    im.set_data(tmp)
    im.set_clim(vmin=tmp.min(),vmax=tmp.max())     
    plt.draw()
    plt.pause(0.001)
    
    

#%%


from parameterFile_stage_2                  import initializeParameterFile
# initialize parameter file
param_stage_2 = initializeParameterFile()

# generate an AO obj_stage_1ect
obj_stage_2 = run_initialization_AO_PWFS(param_stage_2)

#%%
param_stage_2['nPhotonPerSubaperture'] = 1000

obj_stage_2.M2C_cl = obj_stage_2.M2C
obj_stage_2.display= True
param_stage_2['gainCL'] = 0.3
from OOPAO.closed_loop.run_cl_from_phase_screens import run_cl_from_phase_screens



output_stage_2 = run_cl_from_phase_screens(param         = param_stage_2,\
                                            ao_object    = obj_stage_2,
                                            phase_screens = list(output['residual_phase_screen']))
    
#%% check the long exposure psf if requested

# get the long exposure PSF
le_psf = output_stage_2['long_exposure_psf']

# normalize 
le_psf /= le_psf.max()

plt.figure(),
plt.imshow(np.log10(le_psf),extent = [obj_stage_1.tel.xPSF_arcsec[0],obj_stage_1.tel.xPSF_arcsec[1],obj_stage_1.tel.xPSF_arcsec[0],obj_stage_1.tel.xPSF_arcsec[1]])
plt.clim([-4,0])    
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
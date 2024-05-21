# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:18:57 2024

@author: cheritier
"""
#%%
from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
from OOPAO.Detector import Detector
from OOPAO.Atmosphere import Atmosphere
import matplotlib.pyplot as plt

# create a telescope object samplingTime defines the rate at which the atmosphere is updated
tel = Telescope(80, 10,samplingTime= 0.01)

# create a source object
ngs = Source('K', 8)

# associate the ngs to the Telescope
ngs*tel

# create an atmosphere object
atm = Atmosphere(telescope= tel, r0 = 0.1, L0=20, windSpeed=[10], fractionalR0=[1], windDirection=[10], altitude=[0])
atm.initializeAtmosphere(tel)

# combine telescope with the atmosphere
tel+atm

# create different detector objects

cam = Detector(psf_sampling=1)

cam2 = Detector(psf_sampling=2)

cam3 = Detector(psf_sampling=4)

ngs*tel*cam*cam2*cam3

# set the different parameters
cam.integrationTime = None # if None it will use the AO loop frequency
cam.readoutNoise = 10
cam.photonNoise = True
cam.QE = 0.4


cam2.integrationTime = 20*tel.samplingTime
cam2.readoutNoise = 10
cam2.photonNoise = True
cam2.FWC = 20000 # saturation of the detector in e-



cam3.resolution = 20
cam3.readoutNoise = 10
cam3.photonNoise = True
cam3.integrationTime = tel.samplingTime


# number of pixels to crop to zoom on the core of the PSF 

plt.close('all')

from OOPAO.tools.displayTools import cl_plot


plot_obj = cl_plot(list_fig          = [cam.frame[:],cam2.frame[:],cam3.frame[:]],\
                   type_fig          = ['imshow','imshow','imshow'],\
                   list_title        = ['Integration Time: '+str(cam.integrationTime)+' s','Integration Time: '+str(cam2.integrationTime)+' s','Integration Time: '+str(cam3.integrationTime)+' s'],\
                   list_lim          = [None,None,None],\
                   list_label        = [None,None,None],\
                   n_subplot         = [3,1],\
                   list_display_axis = [None,None,None],\
                   list_ratio        = [[0.95,0.1],[1,1,1]]) 

for i in range(20000):
    atm.generateNewPhaseScreen(i)
    tel-atm
    ngs*tel*cam
    ngs*tel*cam2
    ngs*tel*cam3

    cl_plot(list_fig   = [cam.frame[:],cam2.frame[:],cam3.frame[:]],
                               plt_obj = plot_obj)
    plt.pause(0.001)
    if plot_obj.keep_going is False:
        break


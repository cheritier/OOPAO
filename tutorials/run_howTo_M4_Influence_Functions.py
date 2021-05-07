# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:09:47 2020

@author: cheritie
"""

import __load__psim
__load__psim.load_psim()

import matplotlib.pyplot as plt
import numpy as np 
from AO_modules.M1_model.make_ELT_pupil             import generateEeltPupilReflectivity
from AO_modules.M4_model.make_M4_influenceFunctions import makeM4influenceFunctions
from AO_modules.MisRegistration                     import MisRegistration

#%% ------------------------------------------- COMPUTE M1 PUPIL ----------------------------------------------------

# This is an example with uniform reflectivities of segments
refl = np.ones(798)

dead = 0
refl[(np.random.rand(dead)*797).astype(int)] = 0.
# The generated image will be 752 pixels wide
nSubap        = 96

offset = -60                                                                    # rotation offset to align M1 and M4

nPixPerSubap  = 4 
nRes          = nSubap*nPixPerSubap
i0            = nRes/2+0.5  # the centre will be at the centre of the image
j0            = nRes/2+0.5
rotdegree     = 0  # pupil will be rotated by 11 degrees
pixscale      = 40./nRes  # this is pixel scale
dspider       = 0.51   # spiders are 51cm wide
gap           = 0.0   # this is the gap between segments

# GO !!!!.......
pup = generateEeltPupilReflectivity(refl, nRes, dspider, i0, j0, pixscale, gap, rotdegree, softGap=True)

# plot things
plt.figure()
plt.imshow(pup.T, origin='l')

#%% ------------------------------------------- COMPUTE M4 IFS ----------------------------------------------------

# Initialize the mis-registrations 
misReg = MisRegistration()
misReg.shiftX               = 0
misReg.shiftY               = 0
misReg.rotationAngle        = 0
misReg.anamorphosisAngle    = 0
misReg.radialScaling        = 0
misReg.tangentialScaling    = 0
 
# path for the M4_IFs (cropped IFs defined on a 120 by 120 pix)
filename = '/Disk3/cheritier/psim/data_calibration/cropped_M4IF.fits'

M4_IF,M4_coord,nAct = makeM4influenceFunctions(pup      = pup,\
                                               filename = filename,\
                                               misReg   = misReg,\
                                               dm       = None,\
                                               nAct     = 1*892,\
                                               nJobs    = 4,\
                                               nThreads = 2,\
                                               order    = 1)
                                               
# sum the IF
sum_M4 = np.reshape(np.sum((M4_IF)**1,axis=1),[nRes,nRes])
#sum_M4 /= sum_M4.max()

# sum the cube of the IFs
sum_M4_cube=np.reshape(np.sum((M4_IF)**3,axis=1),[nRes,nRes])
#sum_M4_cube /= sum_M4_cube.max()

#%% plot things
plt.figure()
plt.subplot(1,2,1)
plt.imshow(sum_M4.T*pup.T, origin='l')
plt.title('Sum of the influence functions')
plt.subplot(1,2,2)
plt.imshow(sum_M4_cube.T*pup.T, origin='l')
plt.plot(M4_coord[:,0],M4_coord[:,1],'.r')
plt.title('Sum of cube of the influence functions')



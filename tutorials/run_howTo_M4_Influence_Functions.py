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
import time
#%% ------------------------------------------- COMPUTE M1 PUPIL ----------------------------------------------------

# This is an example with uniform reflectivities of segments
refl = np.ones(798)

dead = 0
refl[(np.random.rand(dead)*797).astype(int)] = 0.
# The generated image will be 752 pixels wide

nRes          = 384 # resolution in pix of the pupil
i0            = nRes/2+0.5-1        # the centre will be at the centre of the image
j0            = nRes/2+0.5-1        # the centre will be at the centre of the image
rotdegree     = 20                  # pupil will be rotated by 20 degrees
pixscale      = 40./nRes            # this is pixel scale
dspider       = 0.51                # spiders are 51cm wide
gap           = 0.0                 # this is the gap between segments

# GO !!!!.......
pup = generateEeltPupilReflectivity(refl, nRes, dspider, i0, j0, pixscale, gap, rotdegree, softGap=True)

# plot things
plt.figure()
plt.imshow(pup)

#%% ------------------------------------------- COMPUTE M4 IFS I ----------------------------------------------------
# 1) First case using cropped M4 IFs

# Initialize the mis-registrations 
misReg = MisRegistration()
misReg.shiftX               = 0
misReg.shiftY               = 0
misReg.rotationAngle        = rotdegree
misReg.anamorphosisAngle    = 0
misReg.radialScaling        = 0
misReg.tangentialScaling    = 0
 



# path for the M4_IFs (cropped IFs defined on a 120 by 120 pix)
filename = '/Disk3/cheritier/psim/data_calibration/cropped_M4IF.fits'
a = time.time()
M4_IF,M4_coord,nAct = makeM4influenceFunctions(pup      = pup,\
                                               filename = filename,\
                                               misReg   = misReg,\
                                               dm       = None,\
                                               nAct     = 2*892,\
                                               order    = 1)
                                               
b = time.time()
print('Done in '+str(b-a)+' s!')
    
# sum the IF
sum_M4 = np.reshape(np.sum((M4_IF)**1,axis=1),[nRes,nRes])

# sum the cube of the IFs
sum_M4_cube=np.reshape(np.sum((M4_IF)**3,axis=1),[nRes,nRes])

#% plot things
plt.figure()
plt.subplot(1,2,1)
plt.imshow(sum_M4.T*pup.T)
plt.title('Sum of the influence functions')
plt.subplot(1,2,2)
plt.imshow(sum_M4_cube.T)
plt.plot(M4_coord[:,0],M4_coord[:,1],'.r')
plt.title('Sum of cube of the influence functions')



#%% ------------------------------------------- COMPUTE M4 IFS II ----------------------------------------------------
# 2) Second case using cropped M4 IFs but with output support different from the pupil resolution

# Initialize the mis-registrations 
misReg = MisRegistration()
misReg.shiftX               = 0
misReg.shiftY               = 0
misReg.rotationAngle        = rotdegree
misReg.anamorphosisAngle    = 0
misReg.radialScaling        = 0
misReg.tangentialScaling    = 0
 


# desired support
S = np.zeros([941,941])

# corresponding pixel size in m
pixel_size = 0.05

# corresponding diameter in m 
D = np.round(941.*pixel_size,4)

# path for the M4_IFs (cropped IFs defined on a 120 by 120 pix)
filename = '/Disk3/cheritier/psim/data_calibration/cropped_M4IF.fits'
a = time.time()
M4_IF,M4_coord,nAct = makeM4influenceFunctions(pup      = S,\
                                               filename = filename,\
                                               misReg   = misReg,\
                                               dm       = None,\
                                               nAct     = 2*892,\
                                               order    = 1,\
                                               floating_precision = 32,\
                                               D = D)
                                               
b = time.time()
print('Done in '+str(b-a)+' s!')
    
# sum the IF
sum_M4 = np.reshape(np.sum((M4_IF)**1,axis=1),[941,941])

# sum the cube of the IFs
sum_M4_cube=np.reshape(np.sum((M4_IF)**3,axis=1),[941,941])

#% plot things
plt.figure()
plt.subplot(1,2,1)
plt.imshow(sum_M4.T)
plt.title('Sum of the influence functions')
plt.subplot(1,2,2)
plt.imshow(sum_M4_cube.T)
plt.plot(M4_coord[:,0],M4_coord[:,1],'.r')
plt.title('Sum of cube of the influence functions')

#%% ------------------------------------------- COMPUTE M4 IFS III ----------------------------------------------------
# 3) Second case using already computed IFs
#   WARNING: when applying rotation on pre-computed IFs, some parts of the IFs outside of the support might have been cropped by the first computation type see 1) and use 2) to avoid this issue
#
from AO_modules.tools.interpolate_influence_functions import interpolate_influence_functions

M4_IF_2D = np.moveaxis(np.reshape(M4_IF,[nRes,nRes,M4_IF.shape[1]]),2,0)


misReg = MisRegistration()
misReg.shiftX               = 0
misReg.shiftY               = 0
misReg.rotationAngle        = 30
misReg.anamorphosisAngle    = 0
misReg.radialScaling        = 0
misReg.tangentialScaling    = 0


# parameters for the interpolation

pixel_size_in       = 40/nRes               # pixel size in [m] of the input influence function
resolution_out      = 400                  # resolution in pixels of the input influence function
pixel_size_out      = 40/resolution_out     # resolution in pixels of the output influence function
mis_registration    = misReg                # mis-registration object to apply with respect to the input influence function
coordinates_in      = M4_coord              # coordinated in [m] of the input influence function


# GO
a = time.time()
IF_out,coord_out = interpolate_influence_functions(M4_IF_2D,pixel_size_in,pixel_size_out, resolution_out, mis_registration,coordinates_in)
b = time.time()
print('Done in '+str(b-a)+' s!')

# corresponding rotated pupil

i0            = resolution_out/2+0.5-1             # the centre will be at the centre of the image
j0            = resolution_out/2+0.5-1             # the centre will be at the centre of the image
rotdegree     = 20+misReg.rotationAngle  # pupil will be rotated by 20 degrees
pixscale      = 40./resolution_out                 # this is pixel scale
dspider       = 0.51                     # spiders are 51cm wide
gap           = 0.0                      # this is the gap between segments

pup_rot = generateEeltPupilReflectivity(refl, resolution_out, dspider, i0, j0, pixscale, gap, rotdegree, softGap=True)

    
# sum the IF
sum_M4 = np.reshape(np.sum((IF_out)**1,axis=2),[resolution_out,resolution_out])

# sum the cube of the IFs
sum_M4_cube=np.reshape(np.sum((IF_out)**3,axis=2),[resolution_out,resolution_out])

#% plot things
plt.figure()
plt.subplot(1,2,1)
plt.imshow(sum_M4.T*pup_rot.T)
plt.title('Sum of the influence functions')
plt.subplot(1,2,2)
plt.imshow(sum_M4_cube.T)
plt.plot(coord_out[:,0],coord_out[:,1],'.r')
plt.title('Sum of cube of the influence functions')
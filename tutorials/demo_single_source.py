# OOPAO imports
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Zernike import Zernike
from OOPAO.Detector import Detector

# general imports
import numpy as np
import matplotlib.pyplot as plt
#%%

# number of sub apertures for the SH WFS
n_subaperture = 20

# telescope
tel = Telescope(resolution = 8*20,
                diameter = 10,
                centralObstruction = 0.15,
                fov = 100)

# boolean variable to set the type of source to use.
# If True, we use natural guide star, otherwise laser guide star is used
use_ngs = True

# guide star for the AO loop
if use_ngs:
    src = Source(optBand = "H",
                 magnitude = 8.0,
                 coordinates = [0, 0], # [arcsec, deg]
                 altitude = np.inf)
else:
    src = Source(optBand = "Na",
                 magnitude = 8.0,
                 coordinates = [0, 0], # [arcsec, deg]
                 altitude = 90e3)

# for now, we need to do this before creating the atmosphere
src*tel

# science star
sci_src = Source(optBand = "K",
                 magnitude = 14.0,
                 coordinates = [20, 90], # [arcsec, deg] --> slightly out of axis to see angular anisoplanatism
                 altitude = np.inf)

# atmosphere
atm = Atmosphere(telescope = tel,                              # Telescope
                 r0 = 0.15,                                    # Fried Parameter [m]
                 L0 = 25,                                      # Outer Scale [m]
                 fractionalR0 = [0.45, 0.1, 0.1, 0.25, 0.1],   # Cn2 Profile
                 windSpeed = [10, 12, 11, 15, 20],             # Wind Speed in [m]
                 windDirection = [0, 72, 144, 216, 288],       # Wind Direction in [degrees]
                 altitude = [0, 1000, 5000, 10000, 12000])     # Altitude Layers in [m]

# initialize atmosphere with current Telescope
atm.initializeAtmosphere(tel)

#%%

# atmosphere layers display without having to propagate through atmosphere

src*tel
# display the atmosphere layers: on-axis source
atm.display_atm_layers()
plt.show()

sci_src*tel
# display the atmosphere layers: science source out of axis
atm.display_atm_layers()
plt.show()

#%%

# propagation through atmosphere - telescope stores the same OPD as source --> useful for backward compatibility

for i in range(5):

    atm.generateNewPhaseScreen(seed=i)

    src**atm*tel

    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im = axes[0].imshow(src.OPD)
    axes[0].set_title("src.OPD")
    fig.colorbar(im, ax=axes[0])

    im1 = axes[1].imshow(tel.OPD)
    axes[1].set_title("tel.OPD")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(src.OPD - tel.OPD)
    axes[2].set_title("Difference")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

#%%

# reset OPD --> both src.resetOPD() and tel.resetOPD() are working as expected
# src.OPD and tel.OPD gets equal to None

# src.resetOPD()
tel.resetOPD()
print(src.OPD)
print(tel.OPD)

# propagate so that we can have numpy arrays. Otherwise, we get None type
src*tel

# Create a figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im = axes[0].imshow(src.OPD)
axes[0].set_title("src.OPD")
fig.colorbar(im, ax=axes[0])

im1 = axes[1].imshow(tel.OPD)
axes[1].set_title("tel.OPD")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(src.OPD - tel.OPD)
axes[2].set_title("Difference")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()

#%%

# applying a known OPD with Zernike combination

nModes = 100 # number of zernike to compute

#create a zernike object
Z = Zernike(telObject = tel, J = nModes)

# compute the zernike polynomials associated to the telescope tel
Z.computeZernike(tel)

# Amplitude of the mode (in m RMS)
amp = np.zeros(nModes)

amp[0] = 0e-9
amp[1] = 0e-9
amp[2] = 200e-9
amp[4] = 200e-9

# set OPD with src.OPD --> tel.OPD is also automatically set
tel.OPD = np.squeeze(Z.modesFullRes@amp)

# Create a figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im = axes[0].imshow(src.OPD)
axes[0].set_title("src.OPD")
fig.colorbar(im, ax=axes[0])

im1 = axes[1].imshow(tel.OPD)
axes[1].set_title("tel.OPD")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(src.OPD - tel.OPD)
axes[2].set_title("Difference")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()

#%%

# propagation towards a detector

cam = Detector(nRes=32,
               integrationTime=tel.samplingTime*1, # with different than tel.sampingTime does not work well
               psf_sampling=4.0,
               readoutNoise=5,
               photonNoise=True)

# changing magnitude, we can see the noise effects as expected
# src.magnitude = 8

# diffraction-limited
src.resetOPD()

# # with Zernike modes applied
# src.OPD = np.squeeze(Z.modesFullRes@amp) # --> this raises an error because OPD_no_pupil is not updated !!!

src*tel*cam

plt.imshow(cam.frame)
plt.title("Diffraction-limited image")
plt.colorbar()
plt.show()

# with turbulence from atmosphere
src*atm*tel*cam

plt.imshow(cam.frame)
plt.title("With residuals")
plt.colorbar()
plt.show()


#%%


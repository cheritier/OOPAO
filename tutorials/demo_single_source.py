# OOPAO imports
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Zernike import Zernike
from OOPAO.Detector import Detector
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.calibration.InteractionMatrix import InteractionMatrix

# general imports
import numpy as np
import matplotlib.pyplot as plt
import time
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

src.resetOPD()

# set OPD with src.OPD --> tel.OPD is also automatically set
# the other way around also
src.OPD = np.squeeze(Z.modesFullRes@amp)

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

##################################################

src.resetOPD()

print(type(src.OPD))
print(type(tel.OPD))
print(type(src.OPD_no_pupil))
print(type(tel.OPD_no_pupil))

# set OPD_no_pupil with src --> tel.OPD_no_pupil is also automatically set
# the other way around also
src.OPD_no_pupil = np.squeeze(Z.modesFullRes@amp)*tel.pupil

# Create a figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im = axes[0].imshow(src.OPD_no_pupil)
axes[0].set_title("src.OPD_no_pupil")
fig.colorbar(im, ax=axes[0])

im1 = axes[1].imshow(tel.OPD_no_pupil)
axes[1].set_title("tel.OPD_no_pupil")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(src.OPD_no_pupil - tel.OPD_no_pupil)
axes[2].set_title("Difference")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()

#%%

# propagation towards a detector

cam = Detector(nRes=32,
               integrationTime=tel.samplingTime*1,
               psf_sampling=4.0,
               readoutNoise=5,
               photonNoise=True)

# changing magnitude, we can see the noise effects as expected
# src.magnitude = 8

# diffraction-limited
tel.resetOPD()

print(type(src.OPD))
print(type(tel.OPD))
print(type(src.OPD_no_pupil))
print(type(tel.OPD_no_pupil))

src*tel*cam

plt.imshow(cam.frame)
plt.title("Diffraction-limited image")
plt.colorbar()
plt.show()

# # with Zernike modes applied
# if we only update OPD, that would raise an error because OPD_no_pupil is not updated !!!
src.OPD = np.squeeze(Z.modesFullRes@amp)
# updating also OPD_no_pupil to avoid error
src.OPD_no_pupil = src.OPD*tel.pupil

plt.imshow(src.OPD)
plt.title("OPD")
plt.colorbar()
plt.show()

plt.imshow(src.OPD_no_pupil)
plt.title("OPD_no_pupil")
plt.colorbar()
plt.show()

src*tel*cam

plt.imshow(cam.frame)
plt.title("With Zernike aberrations")
plt.colorbar()
plt.show()

# with turbulence from atmosphere
src*atm*tel*cam

plt.imshow(cam.frame)
plt.title("With atmospheric residuals")
plt.colorbar()
plt.show()


#%%

# testing the SH wfs

wfs = ShackHartmann(nSubap=20,
                    telescope=tel,
                    lightRatio=0.5,
                    threshold_cog=0.01,
                    is_geometric=False,
                    shannon_sampling=True)

print(wfs.d_subap)

src.resetOPD()

src*tel*wfs

print(np.shape(wfs.signal))

plt.plot(wfs.signal)
plt.title("WFS signal")
plt.grid()
plt.show()

plt.imshow(wfs.cam.frame)
plt.colorbar()
plt.show()

#%%

# testing zernike modes propagation towards SH

# boolean variable to change between geometric and diffractive SH
use_geo_sh = False
# with geometric SH, there are no edge effects as expected
# with diffractive they are visible as expected

if use_geo_sh:
    wfs.is_geometric = True
else:
    wfs.is_geometric = False

# Amplitude of the mode (in m RMS)
amp = np.zeros(nModes)

amp[0] = 0e-9
amp[1] = 100e-9
amp[2] = 0e-9
amp[4] = 0e-9

src.resetOPD()
src.OPD = np.squeeze(Z.modesFullRes@amp)
src.OPD_no_pupil = src.OPD*tel.pupil  ## --> if I do not do this, I get an error!!
# This should be automatically set when we set the opd


src*tel*wfs

plt.plot(wfs.signal)
plt.title("WFS signal")
plt.grid()
plt.show()

plt.imshow(wfs.signal_2D)
plt.title("WFS signal 2D")
plt.show()

if not use_geo_sh:
    plt.imshow(wfs.cam.frame)
    plt.colorbar()
    plt.show()

#%%

# deformable mirror in Fried geometry

dm = DeformableMirror(nSubap=20,
                      telescope=tel,
                      mechCoupling=0.3,
                      altitude=None)

plt.imshow(dm.modes.T @ dm.modes)
plt.title("IF.T @ IF",pad=10,fontsize=14)
plt.colorbar()
plt.show()

poke = np.zeros(dm.nValidAct,dtype=float)
poke[106] = 1e-9

dm.coefs = poke

src**tel*dm

plt.imshow(src.OPD)
plt.title("OPD after poke")
plt.colorbar()
plt.show()

plt.imshow(src.OPD_no_pupil)
plt.title("OPD_no_pupil after poke")
plt.colorbar()
plt.show()

src*tel*wfs

plt.plot(wfs.signal)
plt.title("WFS signal after poke")
plt.grid()
plt.show()

#%% KL modes

from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.tools.displayTools import cl_plot, displayMap
# use the default definition of the KL modes with forced Tip and Tilt. For more complex KL modes, consider the use of the compute_KL_basis function.
M2C_KL = compute_KL_basis(tel, atm, dm,lim = 1e-2) # matrix to apply modes on the DM

#%%

# src.resetOPD() # if I don't reset the OPD, and use src*tel*dm I get an error

# when using src**tel*dm, the reset is automatically done, so no error occurs

# apply a KL mode
dm.coefs = M2C_KL[:,:10]*1e-9
# propagate through the DM
src**tel*dm
# show the KL mode applied on the DM
displayMap(tel.OPD)
plt.show()

#%% -----------------------     Calibration: Interaction Matrix  ----------------------------------

# amplitude of the modes in m
stroke=1e-9
# zonal Interaction Matrix
M2C_zonal = np.eye(dm.nValidAct)
# modal Interaction Matrix for 300 modes
M2C_modal = M2C_KL[:,:350]

# just to test the phase offset in the interaction matrix function
phase_off = np.squeeze(Z.modesFullRes@amp)*(2*np.pi)/src.wavelength

plt.imshow(phase_off)
plt.colorbar()
plt.show()

wfs.is_geometric = True # using geometric SH for calibration

calib_zonal = True

if calib_zonal:
    M2C = M2C_zonal
else:
    M2C = M2C_modal

src.resetOPD()

# for now, nMeasurements needs to be 1, otherwise an issue with OPD_no_pupil arises

# zonal interaction matrix
calib = InteractionMatrix(ngs            = src,
                          tel            = tel,
                          dm             = dm,
                          wfs            = wfs,
                          M2C            = M2C,       # M2C matrix used
                          stroke         = stroke,    # stroke for the push/pull in M2C units
                          nMeasurements  = 1,        # number of simultaneous measurements
                          noise          = 'off',     # disable wfs.cam noise
                          display        = True,      # display the time using tqdm
                          single_pass    = True,      # only push to compute the interaction matrix instead of push-pull
                          nTrunc         = 10,
                          phaseOffset    = 0)


plt.imshow(calib.D)
plt.title("Interaction matrix",pad=10,fontsize=14)
plt.colorbar()
plt.show()

plt.plot(calib.D[:,0])
plt.grid()
plt.show()

#%%

wfs.is_geometric = False # back to diffractive SH

# loop parameters
gainCL                  = 0.4
wfs.cam.photonNoise     = True
frame_delay             = 1
nLoop = 500

plt.imshow(calib.M)
plt.title("Reconstruction matrix",pad=10,fontsize=14)
plt.colorbar()
plt.show()

plt.imshow(calib.Mtrunc)
plt.title("Reconstruction matrix (truncated SVD)",pad=10,fontsize=14)
plt.colorbar()
plt.show()

print("Conditioning number: {}".format(calib.cond))

reconstructor = M2C@calib.Mtrunc

# allocate memory to save data
total = np.zeros(nLoop)
residual_SRC = np.zeros(nLoop)
residual_NGS = np.zeros(nLoop)

wfsSignal = np.arange(0,wfs.nSignal)*0

#%%

dm.coefs = 0

for i in range(nLoop):

    a = time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    # propagate light from the NGS through the atmosphere
    src**atm*tel
    # save phase variance
    total[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
    # propagate light from the NGS through the atmosphere, with dm commands applied
    src**atm*tel*dm*wfs*cam
    # save residuals corresponding to the NGS
    residual_NGS[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9

    # plt.imshow(cam.frame)
    # plt.colorbar()
    # plt.show()

    if frame_delay == 1:
        wfsSignal = wfs.signal

    # apply the commands on the DM
    dm.coefs = dm.coefs - gainCL * np.matmul(reconstructor, wfsSignal)

    # store the slopes after computing the commands => 2 frames delay
    if frame_delay == 2:
        wfsSignal = wfs.signal

    # print('Elapsed time: ' + str(time.time() - a) + ' s')
    print("Residuals: {} nm rms".format(residual_NGS[i]))

#%%

plt.plot(residual_NGS,label='NGS residuals')
plt.plot(total,label='Total')
plt.title("AO performance on-axis (NGS): {:.0f} nm rms".format(np.mean(residual_NGS[10:])), fontsize=14,pad=10)
plt.ylabel("Residuals (nm rms)",fontsize=12,labelpad=10)
plt.xlabel("Frames",fontsize=12,labelpad=10)
plt.grid()
plt.show()

#%%

src.resetOPD()

dm.coefs = poke

src**tel*dm
tel.src.OPD += np.squeeze(Z.modesFullRes@amp)

plt.imshow(tel.src.phase)
plt.colorbar()
plt.show()

plt.imshow(src.OPD)
plt.colorbar()
plt.show()

plt.imshow(tel.OPD)
plt.colorbar()
plt.show()

#%%

# Amplitude of the mode (in m RMS)
amp = np.zeros(nModes)

amp[0] = 0e-9
amp[1] = 100e-9
amp[2] = 0e-9
amp[4] = 0e-9

dm.coefs = poke

# src.resetOPD()
src**tel*dm

plt.imshow(src.OPD)
plt.title("src.OPD")
plt.colorbar()
plt.show()

src.OPD += np.squeeze(Z.modesFullRes@amp)
src.OPD_no_pupil = src.OPD*tel.pupil

plt.imshow(src.phase)
plt.title("src.phase")
plt.colorbar()
plt.show()

plt.imshow(src.phase_no_pupil)
plt.title("src.phase_no_pupil")
plt.colorbar()
plt.show()

plt.imshow(src.OPD)
plt.title("src.OPD")
plt.colorbar()
plt.show()

plt.imshow(tel.OPD)
plt.title("tel.OPD")
plt.colorbar()
plt.show()

plt.imshow(tel.OPD_no_pupil)
plt.title("tel.OPD_no_pupil")
plt.colorbar()
plt.show()

src**tel*wfs

plt.plot(wfs.signal)
plt.grid()
plt.show()
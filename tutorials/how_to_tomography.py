" Tomography tutorial"

#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Asterism import Asterism
from OOPAO.FieldTransformer import FieldTransformer
import tomoAO
from OOPAO.tools.displayTools import cl_plot, displayMap


#%% ### Telescope ###

n_subaperture = 20
n_pixel_per_subaperture = 8
 
tel = Telescope(diameter          = 8.0,   # diameter [m]
               resolution         = n_pixel_per_subaperture * n_subaperture,  # resolution [pix.]
               centralObstruction = 0.2,   # central obstruction
               samplingTime       = 0.001, # sampling time [s]
               fov                = 40)    # FOV of telescope [arcsec]

# N = 2
# n_subaperture += (2*N)
# tel.pad(padding_values=N*n_pixel_per_subaperture)

plt.figure()
plt.imshow(tel.pupil)
plt.title("Telescope pupil",fontsize=16,pad=10)
plt.colorbar()
plt.show()

#%% ### Asterism ###

n_lgs = 4
lgs_zenith = [10]*n_lgs
lgs_azimuth = np.linspace(0,360,n_lgs,endpoint=False)
lgs_altitude = 90e3 #np.inf # lower altitude to simulate LGS extended spots
lgs_is_extended = False

if lgs_is_extended:
    n_Na = 21 # number of points to model the Na Profile
    lgs_Na_profile = np.vstack([np.linspace(lgs_altitude-5000,lgs_altitude+5000,n_Na),np.ones(n_Na)/n_Na]) # [altitude coordinates, Na Profile]
    fwhm_spot = 1. # spots FWHM arcsec
    # case with extension of the LGS spots (Cone effect + spots elongation included)
    lgs_asterism = Asterism([Source(optBand='Na',
                                    magnitude=0,
                                    coordinates=[lgs_zenith[kLgs], lgs_azimuth[kLgs]],
                                    altitude=lgs_altitude,
                                    Na_profile=lgs_Na_profile,
                                    fwhm_spot_up=fwhm_spot,
                                    laser_coordinates=[tel.D/2 * np.cos(np.deg2rad(lgs_azimuth[kLgs])),tel.D/2 * np.sin(np.deg2rad(lgs_azimuth[kLgs])) ]) for kLgs in range(n_lgs)])
else:
    # Case with LGS treated as point sources (Cone effect included)
    lgs_asterism = Asterism([Source(optBand='Na',magnitude=0, coordinates=[lgs_zenith[kLgs], lgs_azimuth[kLgs]], altitude=lgs_altitude) for kLgs in range(n_lgs)])
    
# sensing wavelength of the WFS
sensing_wavelength  = lgs_asterism.src[0].wavelength

# telescope fov needs to be in accordance with the asterism used
assert tel.fov >= 2*lgs_zenith[0]

lgs_asterism*tel



#%%

r0 = 0.15

# atm = Atmosphere(telescope=tel,  # Telescope
#                   r0=r0,  # Fried Parameter [m]
#                   L0=25,  # Outer Scale [m]
#                   fractionalR0=[0.5,0.2,0.3],  # Cn2 Profile
#                   windSpeed=[10,15,13],  # Wind Speed in [m]
#                   windDirection=[0,90,120],  # Wind Direction in [degrees]
#                   altitude=[0,2000,10000]) # Altitude Layers in [m]

atm = Atmosphere(telescope=tel,  # Telescope
                  r0=r0,  # Fried Parameter [m]
                  L0=25,  # Outer Scale [m]
                  fractionalR0=[1],  # Cn2 Profile
                  windSpeed=[10],  # Wind Speed in [m]
                  windDirection=[0],  # Wind Direction in [degrees]
                  altitude=[0]) # Altitude Layers in [m]

atm.initializeAtmosphere(telescope=tel)

atm.display_atm_layers()

#%%

# n_pix = n_pixel_per_subaperture/4
n_pix = 2.0
shift_x = [n_pix,-n_pix,n_pix,-n_pix]
shift_y = [n_pix,-n_pix,-n_pix,n_pix]

### Field transformer class to simulate different shifts between the 4 SHs
# this operator allows to perform a geometrical transformation to the EM field
# which will be different for each guide star --> SH pair (line of sight)
# in this case only shifts are being considered, but rotations, magnifications, and other are possible

ft = FieldTransformer(src=lgs_asterism,
                      shift_x=shift_x,
                      shift_y=shift_y)

lgs_asterism**tel

# diffractive SH WFS (is_geometric = False), with Nyquist sampled spots
wfs = ShackHartmann(telescope          = tel,
                    nSubap             = n_subaperture,
                    lightRatio         = 1.0,
                    is_geometric       = False,
                    shannon_sampling   = False,
                    threshold_cog      = 0.01,
                    n_pixel_per_subaperture=6,
                    em_field_transform=ft)

# assert we indeed created a diffractive SH
assert wfs.is_geometric == False

# ast --> tel --> EM transform --> SH
lgs_asterism**tel*ft*wfs

displayMap(wfs.cam.frame,axis=0)
plt.show()
plt.close()

plt.figure()
plt.plot(wfs.signal)
plt.grid()
plt.show()
plt.close()

# Each SH will have its own set of valid_subapertures, and reference signal
valid_sh_subapertures = np.zeros((lgs_asterism.n_source,wfs.nSubap,wfs.nSubap),dtype=float)
# valid_sh_subapertures = []

for i in range(lgs_asterism.n_source):

    plt.figure()
    plt.title("Valid subapertures masks for source {}".format(i))
    plt.imshow(wfs.sh_data['src_' + str(i)].valid_subapertures)
    plt.colorbar()
    plt.show()
    plt.close()

    print(np.sum(wfs.sh_data['src_' + str(i)].valid_subapertures))

    valid_sh_subapertures[i] = wfs.sh_data['src_' + str(i)].valid_subapertures
    # valid_sh_subapertures.append(wfs.sh_data['src_' + str(i)].valid_subapertures)

displayMap(wfs.reference_signal_2D,axis=0)
plt.show()
plt.close()

lgs_asterism**tel

# geometric SH WFS (is_geometric = True)
wfs_geom = ShackHartmann(telescope     = tel,
                         nSubap             = n_subaperture,
                         lightRatio         = 1.0,
                         is_geometric       = True,
                         shannon_sampling   = True,
                         threshold_cog      = 0.01,
                         em_field_transform=ft)

assert wfs_geom.is_geometric == True

#%%

### Deformable mirror ###

dm = DeformableMirror(telescope = tel,
                      nSubap = n_subaperture*2,
                      mechCoupling = 0.3,
                      pitch = None,
                      actuator_selection=None)

print("DM pitch: {} m".format(dm.pitch))
print(tel.D/(dm.nAct-1))

## DM unfiltered actuator mask --> needed for the tomographic reconstruction ##
M = np.zeros(dm.nAct**2,dtype=bool)
M[dm.validAct]=True
unfiltered_act_mask = M.reshape(dm.nAct,dm.nAct)

plt.imshow(unfiltered_act_mask)
plt.title("DM unfiltered actuator mask",fontsize=14,pad=10)
plt.colorbar()
plt.show()

dm.unfiltered_act_mask = unfiltered_act_mask

#%%

# natural guide star giving the optimization direction we want
# Giving coordinates [0,0] to ngs, means we are optimizing the tomographic reconstruction in this direction
ngs = Source(optBand="Na",        # Optical band (see photometry.py)
             magnitude=6.0,       # Source Magnitude
             altitude=np.inf,     # altitude
             coordinates=[0, 0])  # Source coordinated [arcsec,deg]

# Define a scientific source:
science = Source(optBand='H', magnitude=0)

ngs*tel
science*tel

#%% ### Tomography ###
# Dictionary to be filled with parameters necessary for the tomographic reconstructor
config_vars = {}

# oversampling factor
config_vars["os"] = 2
# number of subapertures for the SH wfs
config_vars["nSubaperture"] = n_subaperture
# mechanical coupling of the Deformable mirror
config_vars['mechanicalCoupling'] = dm.mechCoupling
# resolution for the reconstructed phase
config_vars["dm_resolution"] = config_vars["os"] * config_vars["nSubaperture"] + 1
# resolution of the original simulated phase screens
config_vars["resolution"] = tel.resolution

## add offsets to mmse star by hand ##
ngs.offset_x = 0
ngs.offset_y = 0

#%%

# AO system. We give the different AO objects created in OOPAO as input
aoSys = tomoAO.Simulation.AOSystem(param=config_vars,
                                   tel=tel,               # telescope
                                   atm=atm,               # atmosphere
                                   dm=dm,                 # deformable mirror
                                   lgsAst=lgs_asterism,   # asterism
                                   filtered_subap_mask=valid_sh_subapertures, # list with valid subapertures mask for
                                   # each SH
                                   mmse_star=ngs,   # natural guide star
                                   # (giving optimization direction for mmse reconstruction)
                                   wfs=wfs,   # SH wfs
                                   sci_src=science,   # science source
                                   filtered_subap_mask_operation="union",
                                   os=config_vars["os"])

# misReg information to tomoAO model
for i in range(lgs_asterism.n_source):
    ## put offsets into meters ##
    aoSys.lgsAst[i].offset_x = -shift_x[i]*tel.pixelSize
    aoSys.lgsAst[i].offset_y = -shift_y[i]*tel.pixelSize

print(lgs_asterism.src[0].offset_x)
print(lgs_asterism.src[0].offset_y)

print(aoSys.lgsAst[0].offset_x)
print(aoSys.lgsAst[0].offset_y)

plt.imshow(aoSys.outputReconstructiongrid)
plt.title("Reconstruction grid",fontsize=14,pad=10)
plt.colorbar()
plt.show()

plt.imshow(aoSys.filtered_subap_mask^aoSys.unfiltered_subap_mask)
plt.colorbar()
plt.show()

plt.imshow(aoSys.filtered_subap_mask)
plt.title("Filtered subap mask",fontsize=14,pad=10)
plt.colorbar()
plt.show()

print(np.shape(aoSys.outputReconstructiongrid))
print(np.shape(aoSys.filtered_subap_mask))
print(np.shape(aoSys.unfiltered_subap_mask))

#%% ## Spatio-angular Tomographic reconstructor ###

from tomoAO.Reconstruction.reconClassType import tomoReconstructor

inital_time = time.time()

rec = tomoReconstructor(ao_sys=aoSys,                 # AO system object (as crea:ted before)
                        alpha=10,                     # constant used to compute the noise covariance
                        # noise_covariance=Cn_slope,
                        os=config_vars["os"],         # oversampling factor (used to compute the reconstruction grid)
                        indexation="xxyy",            # related with slopes ordering (in oopao they are xxyy)
                        remove_tt_focus = False,      # wether or not to remove TT from reconstruction
                        filter_subapertures = True)   # wether or not to filter SH subapertures (in simulation the
                                                      # valid subapertures are constant, so this is set False
                                                      
final_time = time.time()

print("Time spent to build tomographic reconstructor: {:.2f} seconds".format(final_time-inital_time))

reconstructor = rec.reconstructor.copy()
atm.r0 = r0 # this is important since the atm.r0 is changed inside the rec function
plt.figure()
plt.imshow(reconstructor)
plt.title("Spatio-angular reconstructor",fontsize=14,pad=10)
plt.colorbar()
plt.show()

#%% Calibration of the SH WFS units to adapt to LGS gains, sampling, etc WITH tomographic reconstructor

lgs_asterism**tel*ft*wfs*wfs_geom


wfs.set_slopes_units(tomographic_reconstructor = reconstructor, src = lgs_asterism, dm = dm)
wfs_geom.set_slopes_units(tomographic_reconstructor = reconstructor, src = lgs_asterism, dm = dm)

print(wfs.slopes_units)
print(wfs_geom.slopes_units)

#%% Test tomographic reconstruction:
plt.close('all')

# propagate to geometric SHWFS 
lgs_asterism**atm*tel*ft*wfs*wfs_geom
# save both geometric and diffractive SHWFS signals
wfs_signal = np.hstack(wfs.signal)
wfs_signal_geo = np.hstack(wfs_geom.signal)

# apply the correction and show the residuals
dm.coefs = -reconstructor@wfs_signal
science**atm*tel*dm
plt.figure(),plt.imshow(science.OPD)
plt.title('Diffractive SHWFS - Residual WFE: ' +str(np.round(1e9*np.std(science.OPD[science.mask]),2)) + ' nm')
plt.show()

# apply the correction and show the residuals
dm.coefs = -reconstructor@wfs_signal_geo
science**atm*tel*dm
plt.figure(),plt.imshow(science.OPD)
plt.title('Geometric SHWFS - Residual WFE: ' +str(np.round(1e9*np.std(science.OPD[science.mask]),2)) + ' nm')
plt.show()

#%%
displayMap(wfs_geom.signal_2D,axis=0)
plt.colorbar()
plt.show()

displayMap(wfs.signal_2D,axis=0)
plt.colorbar()
plt.show()

#%%
# Interaction matrix using a zonal approach and the geometric SH, with a stroke equal to calib_src.wavelength/2/np.pi
# This matrix is necessary for the pseudo-open loop reconstruction

# calib_src = Source('Na', 0)
# calib_src**tel*wfs_geom

dm_eye = np.eye(dm.nValidAct)
imat = np.zeros((wfs_geom.nSignal, dm.nValidAct))

for i_act in tqdm(range(dm.nValidAct)):

    dm.coefs = dm_eye[:, i_act]*lgs_asterism.wavelength/2/np.pi
    lgs_asterism**tel*dm*ft*wfs_geom

    wfsSignal = np.hstack(wfs_geom.signal)

    imat[:, i_act] = wfsSignal

imat = imat*2*np.pi/lgs_asterism.wavelength
# imat = np.vstack([imat]*n_lgs)

plt.imshow(imat)
plt.title("Interaction matrix", fontsize=14,pad=10)
plt.colorbar()
plt.show()

#%%

from OOPAO.calibration.CalibrationVault import CalibrationVault

# IntMat inversion using the SVD
D_inv = CalibrationVault(imat,
                         nTrunc = 50,
                         invert=True)

invIntMat = D_inv.M
invIntMat_trunc = D_inv.Mtrunc

print("Conditioning number : {}".format(D_inv.cond))

noise_propag_coefs = np.diag(invIntMat@invIntMat.T)

plt.plot(noise_propag_coefs)
plt.title("Noise propag. coefficients",pad=10,fontsize=14)
plt.yscale('log')
plt.xscale('log')
plt.grid()
plt.show()

normalized_s_values = D_inv.eigenValues/np.max(D_inv.eigenValues)

plt.plot(normalized_s_values)
plt.title("IM singular value decomposition",pad=10,fontsize=14)
plt.ylabel("Normalized singular values",fontsize=12,labelpad=10)
plt.xlabel("Mode index",fontsize=12,labelpad=10)
plt.yscale('log')
# plt.xscale('log')
plt.grid()
plt.show()

#%%
from OOPAO.tools.tools import strehlMeter
plt.close('all')
dm.coefs=0

# To make sure to always replay the same turbulence, generate a new phase screen for the atmosphere and combine it with the Telescope
atm.generateNewPhaseScreen(seed=10)

from OOPAO.Detector import Detector
# science detector
science_cam = Detector(tel.resolution*4)
science_cam.psf_sampling = 4
science_cam.integrationTime = tel.samplingTime*1

science**tel*science_cam
science_psf_ref = science_cam.frame.copy()

# NGS detector
ngs_cam = Detector(tel.resolution*2)
ngs_cam.psf_sampling = 4
ngs_cam.integrationTime = tel.samplingTime

# Save reference frame
ngs**tel*ngs_cam
ngs_psf_ref = ngs_cam.frame.copy()


# propagate both sources
lgs_asterism**atm*tel
ngs**atm*tel*ngs_cam
science**atm*tel*science_cam

# loop parameters
n_loop = 500  # number of iterations
gain_cl = 0.4  # integrator gain
wfs.cam.photonNoise = False  # enable photon noise on the WFS camera
display = False  # enable the display
frame_delay = 1  # number of frame delay

# variables used to save closed-loop data
SR_ngs = np.zeros(n_loop)
SR_science = np.zeros(n_loop)

wfe_atmosphere = np.zeros(n_loop)
wfe_residual_science = np.zeros(n_loop)
wfe_residual_NGS = np.zeros(n_loop)
wfsSignal = np.arange(0, wfs.nSignal*lgs_asterism.n_source)*0  # buffer to simulate the loop delay

# configure the display pannel
plot_obj = cl_plot(list_fig=[atm.OPD,  # list of data for the different subplots
                              ngs.OPD,
                              science.OPD,
                              np.block([[wfs.cam.frame[0,:,:],wfs.cam.frame[1,:,:]],[wfs.cam.frame[2,:,:],wfs.cam.frame[3,:,:]]]),
                              [dm.coordinates[:, 0], np.flip(dm.coordinates[:, 1]), dm.coefs],
                              [[0, 0], [0, 0], [0, 0]],
                              np.log10(ngs_cam.frame),
                              np.log10(science_cam.frame)],
                    type_fig=['imshow',  # type of figure for the different subplots
                              'imshow',
                              'imshow',
                              'imshow',
                              'scatter',
                              'plot',
                              'imshow',
                              'imshow'],
                    list_title=['Turbulence [nm]',  # list of title for the different subplots
                                'NGS residual [m]',
                                'science residual [m]',
                                'wfs Detector',
                                'DM Commands',
                                None,
                                None,
                                None],
                    list_legend=[None,  # list of legend labels for the subplots
                                None,
                                None,
                                None,
                                None,
                                ['science@'+str(science.coordinates[0])+'"', 'NGS@'+str(ngs.coordinates[0])+'"'],
                                None,
                                None],
                    list_label=[None,  # list of axis labels for the subplots
                                None,
                                None,
                                None,
                                None,
                                ['Time', 'WFE [nm]'],
                                ['NGS PSF@' + str(ngs.coordinates[0]) + '" -- FOV: ' + str(np.round(ngs_cam.fov_arcsec, 2)) + '"', ''],
                                ['science PSF@' + str(science.coordinates[0]) + '" -- FOV: ' + str(np.round(science_cam.fov_arcsec, 2)) + '"', '']],
                    n_subplot=[4, 2],
                    list_display_axis=[None,  # list of the subplot for which axis are displayed
                                      None,
                                      None,
                                      None,
                                      None,
                                      True,
                                      None,
                                      None],
                    list_ratio=[[0.95, 0.95, 0.1],
                                [1, 1, 1, 1]],
                    s=20)  # size of the scatter markers


for i in range(n_loop):
    a = time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.science.phase
    atm.update()
    ngs**atm*tel
    # save the wave-front error of the incoming turbulence within the pupil
    wfe_atmosphere[i] = np.std(ngs.OPD[np.where(tel.pupil > 0)])*1e9
    # propagate light from the ngs through the atmosphere, telescope, DM to the wfs and ngs camera
    ngs**atm*tel*dm*ngs_cam
    lgs_asterism**atm*tel*dm*ft*wfs

    # save residuals corresponding to the ngs
    wfe_residual_NGS[i] = np.std(ngs.OPD[np.where(tel.pupil > 0)])*1e9
    # save Strehl ratio from the PSF image
    SR_ngs[i] = strehlMeter(PSF=ngs_cam.frame, tel=tel, PSF_ref=ngs_psf_ref, display=False)
    # save the OPD seen by the ngs
    OPD_NGS = ngs.OPD.copy()
    if display:
        NGS_PSF = np.log10(np.abs(ngs_cam.frame))

    # propagate light from the science through the atmosphere, telescope, DM to the science camera
    science**atm*tel*dm*science_cam
    # save residuals corresponding to the science
    wfe_residual_science[i] = np.std(science.OPD[np.where(tel.pupil > 0)])*1e9
    # save the OPD seen by the science
    OPD_science = science.OPD.copy()
    # save Strehl ratio from the PSF image
    SR_science[i] = strehlMeter(PSF=science_cam.frame, tel=tel, PSF_ref=science_psf_ref, display=False)

    # store the slopes after propagating to the wfs <=> 1 frames delay
    if frame_delay == 1:
        wfsSignal = np.hstack(wfs.signal)
        wfsSignal = wfsSignal-imat@dm.coefs


    # apply the commands on the DM (POLC)
    dm.coefs = (1-gain_cl)*dm.coefs - gain_cl * (reconstructor@wfsSignal)

    # store the slopes after computing the commands <=> 2 frames delay
    if frame_delay == 2:
        wfsSignal = np.hstack(wfs.signal)
        wfsSignal = wfsSignal-imat@dm.coefs

    # print('Elapsed time: ' + str(time.time()-a) + ' s')

    # update displays if required
    if display and i > 1:
        science_PSF = np.log10(np.abs(science_cam.frame))
        # update range for PSF images
        plot_obj.list_lim = [None,
                              None,
                              None,
                              None,
                              None,
                              None,
                              [NGS_PSF.max()-6, NGS_PSF.max()],
                              [science_PSF.max()-6, science_PSF.max()]]
        # update title
        plot_obj.list_title = ['Turbulence '+str(np.round(wfe_atmosphere[i]))+'[nm]',
                                'NGS residual '+str(np.round(wfe_residual_NGS[i]))+'[nm]',
                                'science residual '+str(np.round(wfe_residual_science[i]))+'[nm]',
                                'wfs Detector',
                                'DM Commands',
                                None,
                                None,
                                None]

        cl_plot(list_fig=[1e9*atm.OPD,
                          1e9*OPD_NGS,
                          1e9*OPD_science,
                          np.block([[wfs.cam.frame[0,:,:],wfs.cam.frame[1,:,:]],[wfs.cam.frame[2,:,:],wfs.cam.frame[3,:,:]]]),
                          dm.coefs,
                          [np.arange(i+1), wfe_residual_science[:i+1], wfe_residual_NGS[:i+1]],
                          NGS_PSF,
                          science_PSF],
                plt_obj=plot_obj)
        plt.pause(0.01)
        if plot_obj.keep_going is False:
            break
    print('-----------------------------------')
    print('Loop'+str(i) + '/' + str(n_loop))
    print('NGS: Strehl ratio [%] : ', np.round(SR_ngs[i],1), ' WFE [nm] : ', np.round(wfe_residual_NGS[i],2))
    print('science: Strehl ratio [%] : ', np.round(SR_science[i],1), ' WFE [nm] : ', np.round(wfe_residual_science[i],2))

#%%
plt.figure()

plt.plot(wfe_residual_NGS,label='Residuals')
plt.plot(wfe_atmosphere,label="Total")
plt.title('Residuals',fontsize=14,pad=10)
plt.ylabel('Residual [nm rms]',fontsize=12,labelpad=10)
plt.legend(fontsize=12)
plt.grid()
plt.show()


plt.figure()
plt.plot(SR_ngs)
plt.title("Strehl ratio",fontsize=14,pad=10)
plt.grid()
plt.show()

# %%

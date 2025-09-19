# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:51:32 2020

@author: cheritie
"""

from scipy import signal
from OOPAO.tools.displayTools import makeSquareAxes
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
import time

import matplotlib.pyplot as plt
import numpy as np

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap
# %% -----------------------     read parameter file   ----------------------------------
from parameter_files.parameterFile_VLT_I_Band_SHWFS import initializeParameterFile

param = initializeParameterFile()

# %%
plt.ion()

# %% -----------------------     TELESCOPE   ----------------------------------

param['resolution'] = 4*8
# create the Telescope object
tel = Telescope(resolution=param['resolution'],
                diameter=2,
                samplingTime=param['samplingTime'],
                centralObstruction=param['centralObstruction'], display_optical_path=True)
tel2 = Telescope(resolution=param['resolution'],
                diameter=2,
                samplingTime=2*param['samplingTime'],
                centralObstruction=param['centralObstruction'], display_optical_path=True)

# tel.pupil = np.ones((tel.resolution,tel.resolution))

# %% -----------------------     NGS   ----------------------------------
# create the Source object
ngs = Source(optBand=param['opticalBand'],
             magnitude=param['magnitude'])

# combine the NGS to the telescope using '*' operator:
ngs*tel

tel.computePSF(zeroPaddingFactor=6)
plt.figure()
plt.imshow(np.log10(np.abs(tel.PSF)), extent=[
           tel.xPSF_arcsec[0], tel.xPSF_arcsec[1], tel.xPSF_arcsec[0], tel.xPSF_arcsec[1]])
plt.clim([-1, 3])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()


src = Source(optBand='K',
             magnitude=param['magnitude'])


#%%

ngs*tel
plt.figure()
plt.imshow(tel.src.fluxMap),plt.colorbar()

ngs*tel2

plt.figure()
plt.imshow(tel.src.fluxMap),plt.colorbar()

# combine the NGS to the telescope using '*' operator:
# src*tel
# %% -----------------------     ATMOSPHERE   ----------------------------------

# create the Atmosphere object
atm = Atmosphere(telescope=tel,
                 r0=0.08,
                 L0=param['L0'],
                 windSpeed=[50],
                 fractionalR0=[1],
                 windDirection=[0],
                 altitude=[0])
# initialize atmosphere
atm.initializeAtmosphere(tel)

atm.update()

plt.figure()
plt.imshow(atm.OPD*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()


tel+atm
tel.computePSF(8)
plt.figure()
plt.imshow((np.log10(tel.PSF)), extent=[
           tel.xPSF_arcsec[0], tel.xPSF_arcsec[1], tel.xPSF_arcsec[0], tel.xPSF_arcsec[1]])
plt.clim([-1, 3])

plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()

# %% -----------------------     DEFORMABLE MIRROR   ----------------------------------
# mis-registrations object
misReg = MisRegistration(param)
# if no coordonates specified, create a cartesian dm
dm = DeformableMirror(telescope=tel,
                      nSubap=param['nSubaperture'],
                      mechCoupling=param['mechanicalCoupling'],
                      misReg=misReg)

plt.figure()
plt.plot(dm.coordinates[:, 0], dm.coordinates[:, 1], 'x')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates')

# %% -----------------------     SH WFS   ----------------------------------

# make sure tel and atm are separated to initialize the PWFS
tel-atm
tel.resetOPD()
wfs = ShackHartmann(nSubap=2,
                    telescope=tel,
                    lightRatio=param['lightThreshold'],
                    binning_factor=1,
                    is_geometric=False, shannon_sampling=True)


# %
wfs.cam.photonNoise = False

tel*wfs
plt.close('all')
plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS Camera Frame - Without Noise')

wfs.cam.photonNoise = False
tel*wfs
plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS Camera Frame - With Noise')


# %%

d_subap = (tel.D/wfs.nSubap)
pixel_scale_rad = (ngs.wavelength / d_subap) / (wfs.shannon_sampling*2)
pixel_scale_arcsec = 206265*pixel_scale_rad

print(wfs.slopes_units)

[Tip, Tilt] = np.meshgrid(np.linspace(
    0, tel.resolution-1, tel.resolution), np.linspace(0, tel.resolution-1, tel.resolution))

Tip *= 1/np.std(Tip[tel.pupil])

mean_slope = np.zeros(5)

tel.resetOPD()
ngs*tel*wfs
ref = wfs.cam.frame
plt.close('all')

for i in range(4):
    
    amp = i *1e-9

    tel.OPD = tel.pupil*Tip*amp
    tel.OPD_no_pupil = Tip*amp
    print(np.std(tel.OPD))
    # print(np.max(tel.src.phase))
    # # 
    ngs*tel*wfs
    print(wfs.signal)

    plt.figure()
    plt.imshow(wfs.cam.frame)

# %%


# %% -----------------------     Modal Basis   ----------------------------------
# compute the modal basis
M2C_KL = compute_KL_basis(tel, atm, dm)
# foldername_M2C  = None  # name of the folder to save the M2C matrix, if None a default name is used
# filename_M2C    = None  # name of the filename, if None a default name is used
# # KL Modal basis
# M2C_KL = compute_M2C(telescope            = tel,\
#                                  atmosphere         = atm,\
#                                  deformableMirror   = dm,\
#                                  param              = param,\
#                                  nameFolder         = None,\
#                                  nameFile           = None,\
#                                  remove_piston      = True,\
#                                  HHtName            = None,\
#                                  baseName           = None ,\
#                                  mem_available      = 8.1e9,\
#                                  minimF             = False,\
#                                  nmo                = 300,\
#                                  ortho_spm          = True,\
#                                  SZ                 = np.int(2*tel.OPD.shape[0]),\
#                                  nZer               = 3,\
#                                  NDIVL              = 1)
#
# ao_calib =  ao_calibration(param            = param,\
#                            ngs              = ngs,\
#                            tel              = tel,\
#                            atm              = atm,\
#                            dm               = dm,\
#                            wfs              = wfs,\
#                            nameFolderIntMat = None,\
#                            nameIntMat       = None,\
#                            nameFolderBasis  = None,\
#                            nameBasis        = None,\
#                            nMeasurements    = 100)

# %% ZERNIKE Polynomials
# create Zernike Object
Z = Zernike(tel, 300)
# compute polynomials for given telescope
Z.computeZernike(tel)

# mode to command matrix to project Zernike Polynomials on DM
M2C_zernike = np.linalg.pinv(np.squeeze(dm.modes[tel.pupilLogical, :]))@Z.modes

# show the first 10 zernikes
dm.coefs = M2C_zernike[:, :10]
tel*dm
displayMap(tel.OPD)

# %% to manually measure the interaction matrix

# amplitude of the modes in m
stroke = 1e-9
# Modal Interaction Matrix

# %%
wfs.is_geometric = True

M2C_zonal = np.eye(dm.nValidAct)
# zonal interaction matrix
calib_zonal = InteractionMatrix(ngs=ngs,
                                atm=atm,
                                tel=tel,
                                dm=dm,
                                wfs=wfs,
                                M2C=M2C_zonal,
                                stroke=stroke,
                                nMeasurements=100,
                                noise='off')

plt.figure()
plt.plot(np.std(calib_zonal.D, axis=0))
plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')

# %%
# Modal interaction matrix

# Modal interaction matrix
calib_zernike = CalibrationVault(calib_zonal.D@M2C_KL[:, :300])

plt.figure()
plt.plot(np.std(calib_zernike.D, axis=0))
plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')

# %% switch to a diffractive SH-WFS
tel.resetOPD()
ngs*tel*wfs
wfs.is_geometric = True
# #%%
# KL_DM = tel.OPD.reshape(tel.resolution**2,dm.nValidAct)

# covmat = KL_DM.T@KL_DM

# %%
tel.resetOPD()
dm.coefs = M2C_KL

tel*dm

KL_DM = tel.OPD.reshape(tel.resolution**2, M2C_KL.shape[1])

covmat = KL_DM.T@KL_DM


projector = np.diag(1/np.diag(covmat))@KL_DM.T


# %%

tel-atm
tel.resetOPD()
# initialize DM commands
dm.coefs = 0
ngs*tel*dm*wfs

wfs.is_geometric = True

tel+atm

# dm.coefs[100] = -1

tel.computePSF(4)
plt.close('all')

# These are the calibration data used to close the loop
calib_CL = calib_zernike
M2C_CL = M2C_KL[:, :300]


# combine telescope with atmosphere
tel+atm

# initialize DM commands
dm.coefs = 0
ngs*tel*dm*wfs


plt.show()

param['nLoop'] = 20000
# allocate memory to save data
SR = np.zeros(param['nLoop'])
total = np.zeros(param['nLoop'])
residual = np.zeros(param['nLoop'])
slopes = np.zeros([param['nLoop'], wfs.nSignal])
commands = np.zeros([param['nLoop'], dm.nValidAct])
modal_in = np.zeros([param['nLoop'], M2C_KL.shape[1]])
modal_out = np.zeros([param['nLoop'], M2C_KL.shape[1]])


wfsSignal = np.arange(0, wfs.nSignal)*0
SE_PSF = []
LE_PSF = np.log10(tel.PSF_norma_zoom)

plot_obj = cl_plot(list_fig=[atm.OPD, tel.mean_removed_OPD, wfs.cam.frame, [dm.coordinates[:, 0], np.flip(dm.coordinates[:, 1]), dm.coefs], [[0, 0], [0, 0]], np.log10(tel.PSF_norma_zoom), np.log10(tel.PSF_norma_zoom)],
                   type_fig=['imshow', 'imshow', 'imshow',
                             'scatter', 'plot', 'imshow', 'imshow'],
                   list_title=['Turbulence OPD', 'Residual OPD',
                               'WFS Detector', 'DM Commands', None, None, None],
                   list_lim=[None, None, None, None, None, [-4, 0], [-4, 0]],
                   list_label=[None, None, None, None, ['Time', 'WFE [nm]'], [
                       'Short Exposure PSF', ''], ['Long Exposure_PSF', '']],
                   n_subplot=[4, 2],
                   list_display_axis=[None, None,
                                      None, None, True, None, None],
                   list_ratio=[[0.95, 0.95, 0.1], [1, 1, 1, 1]], s=20)
# loop parameters
gainCL = 0.35
wfs.cam.photonNoise = False
display = False

reconstructor = M2C_CL@calib_CL.M

for i in range(param['nLoop']):
    a = time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    # save phase variance
    total[i] = np.std(tel.OPD[np.where(tel.pupil > 0)])*1e9

    modal_in[i, :] = projector@atm.OPD.reshape(tel.resolution**2)
    # save turbulent phase
    turbPhase = tel.src.phase
    # propagate to the WFS with the CL commands applied
    ngs*tel*dm*wfs
    src*tel
    tel.print_optical_path()
    dm.coefs = dm.coefs-gainCL*np.matmul(reconstructor, wfsSignal)

    modal_out[i, :] = projector@tel.OPD.reshape(tel.resolution**2)

    commands[i, :] = dm.coefs.copy()
    slopes[i, :] = wfs.signal.copy()
    # store the slopes after computing the commands => 2 frames delay
    wfsSignal = wfs.signal
    b = time.time()
    print('Elapsed time: ' + str(b-a) + ' s')
    # update displays if required
    if display == True:
        tel.computePSF(4)
        if i > 15:
            SE_PSF.append(np.log10(tel.PSF_norma_zoom))
            LE_PSF = np.mean(SE_PSF, axis=0)

        cl_plot(list_fig=[atm.OPD, tel.mean_removed_OPD, wfs.cam.frame, dm.coefs, [np.arange(i+1), residual[:i+1]], np.log10(tel.PSF_norma_zoom), LE_PSF],
                plt_obj=plot_obj)
        plt.pause(0.1)
        if plot_obj.keep_going is False:
            break

    SR[i] = np.exp(-np.var(tel.src.phase[np.where(tel.pupil == 1)]))
    residual[i] = np.std(tel.OPD[np.where(tel.pupil > 0)])*1e9
    OPD = tel.OPD[np.where(tel.pupil > 0)]

    print('Loop'+str(i)+'/'+str(param['nLoop'])+' Turbulence: ' +
          str(total[i])+' -- Residual:' + str(residual[i]) + '\n')

# %%
plt.figure()
plt.plot(total)
plt.plot(residual)
plt.xlabel('Time')
plt.ylabel('WFE [nm]')
# %%

# convert commands to modes

modal_coefs_in = np.linalg.pinv(M2C_CL)@commands.T

end_v = 9990
start_v = 100
delta = 2
modal_coefs_out = calib_CL.M@slopes[:, :].T


plt.figure()
plt.loglog(np.std(modal_coefs_in[:, 100:], axis=1))
plt.loglog(np.std(modal_coefs_out[:, 100:], axis=1))
plt.loglog(np.std(modal_in[100:, :], axis=0))
plt.loglog(np.std(modal_out[100:, :], axis=0))
makeSquareAxes(plt.gca())

# %%
plt.figure()
plt.plot(modal_coefs_in[10, :])
# %%


def TF_Her_Hcl_Hol_Hn(fp, loop_gain, Ti, Tau, Tdm):
    dfp = fp[1]-fp[0]
    I = 1j
    S = I*2.*np.pi*fp
    #H_WFS = (1.-np.exp(-S*Ti)) / (S*Ti)
    H_WFS = np.exp(-1.*Ti/2*S)  # ; like in simulation
    H_RTC = np.exp(-1.*Tau*S)
    H_DM = np.exp(-Tdm*S)
    H_DAC = (1.-np.exp(-S*Ti)) / (S*Ti)
    CC = loop_gain / (1-np.exp(-S*Ti))
    H_OL = H_WFS*H_RTC*H_DAC*H_DM*CC
    H_CL = H_OL/(1+H_OL)
    H_ER = 1./(1.+H_OL)
    H_N = H_CL/H_WFS
    return H_ER, H_CL, H_OL, H_N


out = modal_out.T
inp = modal_in.T


def TF_Her_Hcl_Hol_Hn(fp, loop_gain, Ti, Tau, Tdm):
    dfp = fp[1]-fp[0]
    I = 1j
    S = I*2.*np.pi*fp
    #H_WFS = (1.-np.exp(-S*Ti)) / (S*Ti)
    H_WFS = np.exp(-1.*Ti/2*S)  # ; like in simulation
    H_RTC = np.exp(-1.*Tau*S)
    H_DM = np.exp(-Tdm*S)
    H_DAC = (1.-np.exp(-S*Ti)) / (S*Ti)
    CC = loop_gain / (1-np.exp(-S*Ti))
    H_OL = H_WFS*H_RTC*H_DAC*H_DM*CC
    H_CL = H_OL/(1+H_OL)
    H_ER = 1./(1.+H_OL)
    H_N = H_CL/H_WFS
    return H_ER, H_CL, H_OL, H_N


mod_test = [10, 100]
plt.close('all')


fs = 1/tel.samplingTime
end = 23000-100
start = 100
mod_test = [10]

# scipy.signal.welch(x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')


nperseg = 1024
noverlap = 128
nfft = nperseg
for i_mode in range(1):
    mode = mod_test[i_mode]
    f, Poo = signal.csd(out[mode, start:end], out[mode, start:end],
                        fs, nperseg=nperseg, window='hann', noverlap=noverlap, nfft=nfft)
    f, Poi = signal.csd(out[mode, start:end], inp[mode, start:end],
                        fs, nperseg=nperseg, window='hann', noverlap=noverlap, nfft=nfft)
    f, Pii = signal.csd(inp[mode, start:end], inp[mode, start:end],
                        fs, nperseg=nperseg, window='hann', noverlap=noverlap, nfft=nfft)

    # f, Poo = signal.welch(out[mode,start:end],fs,nperseg=4*128,average ='median')

    # f, Pii = signal.welch(inp[mode,start:end],fs,nperseg=4*128,average ='median')

    loop_gain = gainCL
    fp = (f.copy())[1:]
    Ti = 1*tel.samplingTime
    Tau = 1*tel.samplingTime/2
    Tdm = 1*tel.samplingTime/2

    Her, Hcl, Hol, Hn = TF_Her_Hcl_Hol_Hn(fp, loop_gain, Ti, Tau, Tdm)

    Her_dB = 20.*np.log10(np.abs(Her))

    FTR = 20. * np.log10(np.abs((Poi)/(Pii)))
    FTR2 = 10. * np.log10(np.abs((Poo)/(Pii)))

    DSP = 20.*np.log10(np.abs(Pii))
    DSP_CL = 20.*np.log10(np.abs(Poo))

#     plt.figure(9)
#     plt.subplot(1,1,i_mode+1)
#     plt.plot(np.abs(Her)/np.abs(Poi[1:]/Pii[1:]))
#     plt.title('KL Mode ' + str(mod_test[i_mode]))
# #    plt.grid(which='both')
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel(r'20 log($\frac{\Phi_{res}}{\Phi_{turb}}$) [dB]')
# #    plt.ylim([-35,10])
#     makeSquareAxes(plt.gca())

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.semilogx(f[1:], (FTR[1:]), linewidth=2,
                 label='loop gain '+str(loop_gain)+' -- OOPAO')
    plt.semilogx(f[1:], (FTR2[1:]), linewidth=2,
                 label='loop gain '+str(loop_gain)+' -- OOPAO')

    plt.semilogx(f[1:], Her_dB, '--', alpha=0.8, linewidth=2,
                 label='loop gain '+str(loop_gain)+' -- Theory')

    plt.title('Rejection Transfer Function')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'20 log($\frac{\Phi_{res}}{\Phi_{turb}}$) [dB]')
    plt.ylim([-50, 20])
    makeSquareAxes(plt.gca())
    plt.legend(['OOPAO', 'Theory'])

    plt.subplot(1, 2, 2)

    plt.semilogx(f, (DSP), label='OL -- KL Mode '+str(mod_test[i_mode]+1))
    plt.semilogx(f, (DSP_CL), label='CL -- KL Mode '+str(mod_test[i_mode]+1))
    plt.title('Wind-Speed = 50 m/s')
    plt.grid(which='both')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectrum Density')
    plt.legend()
#    plt.ylim([-35,10])
    makeSquareAxes(plt.gca())
plt.grid(which='both')
plt.legend()

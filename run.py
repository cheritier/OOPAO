from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import tqdm as tqdm

import __load__psim

__load__psim.load_psim()
from AO_modules.Atmosphere import Atmosphere
from AO_modules.DeformableMirror import DeformableMirror
from AO_modules.MisRegistration import MisRegistration
from AO_modules.Pyramid import Pyramid
from AO_modules.Source import Source
from AO_modules.Telescope import Telescope
from AO_modules.Zernike import Zernike
from AO_modules.calibration.InteractionMatrix import interactionMatrix
from tutorials.parameter_files.parameterFile_VLT_I_Band_PWFS import initializeParameterFile


class SimDisplay:
    def __init__(self, sim: Simulation, init: bool = True):
        self.sim = sim

        self.fig = None
        self.ax = None
        self.im_atm = None
        self.im_dm = None
        self.im_psf_ol = None
        self.im_residual = None
        self.im_wfs_cl = None
        self.im_psf = None

        if init:
            self._init()

    def _init(self):
        with plt.ion():
            self.fig, self.ax = plt.subplots(2, 3)
            self.fig.tight_layout()

            self.ax_atm = self.ax[0, 0]
            self.im_atm = self.ax_atm.imshow(self.sim.tel.src.phase)
            plt.colorbar(self.im_atm)
            self.ax_atm.set_title("Turbulence phase [rad]")

            self.ax_dm = self.ax[0, 1]
            self.im_dm = self.ax_dm.imshow(self.sim.dm.OPD * self.sim.tel.pupil)
            plt.colorbar(self.im_dm)
            self.ax_dm.set_title("DM phase [rad]")

            self.sim.tel.computePSF(zeroPaddingFactor=self.sim.zeroPaddingFactor)
            self.ax_psf_ol = self.ax[0, 2]
            self.im_psf_ol = self.ax_psf_ol.imshow(self.sim.tel.PSF)
            plt.colorbar(self.im_psf_ol)
            self.ax_psf_ol.set_title("OL PSF")

            self.ax_residual = self.ax[1, 0]
            self.im_residual = self.ax_residual.imshow(self.sim.tel.src.phase)
            plt.colorbar(self.im_residual)
            self.ax_residual.set_title("Residual phase [rad]")

            self.ax_wfs_cl = self.ax[1, 1]
            self.im_wfs_cl = self.ax_wfs_cl.imshow(self.sim.wfs.cam.frame)
            plt.colorbar(self.im_wfs_cl)
            self.ax_wfs_cl.set_title("Pyramid Frame CL")

            self.ax_psf = self.ax[1, 2]
            self.im_psf = self.ax_psf.imshow(self.sim.tel.PSF)
            plt.colorbar(self.im_psf)
            self.ax_psf.set_title("CL PSF")

            plt.show()

    def update_ol_psf(self):
        with plt.ion():
            # compute the OL PSF and update the display
            self.sim.tel.computePSF(zeroPaddingFactor=self.sim.zeroPaddingFactor)
            self.im_psf_ol.set_data(np.log(self.sim.tel.PSF / self.sim.tel.PSF.max()))
            self.im_psf_ol.set_clim(vmin=-3, vmax=0)

    def update(self):
        with plt.ion():
            # Turbulence
            turb_phase = self.sim.tel.src.phase
            self.im_atm.set_data(turb_phase)
            self.im_atm.set_clim(vmin=turb_phase.min(), vmax=turb_phase.max())

            # WFS frame
            cam = self.sim.wfs.cam.frame
            self.im_wfs_cl.set_data(cam)
            self.im_wfs_cl.set_clim(vmin=cam.min(), vmax=cam.max())

            # DM OPD
            dm_opd = self.sim.tel.pupil * self.sim.dm.OPD * 2 * np.pi / self.sim.ngs.wavelength
            self.im_dm.set_data(dm_opd)
            self.im_dm.set_clim(vmin=dm_opd.min(), vmax=dm_opd.max())

            # residual phase
            D = self.sim.tel.src.phase
            D = D - np.mean(D[self.sim.tel.pupil])
            self.im_residual.set_data(D)
            self.im_residual.set_clim(vmin=D.min(), vmax=D.max())

            self.sim.tel.computePSF(zeroPaddingFactor=self.sim.zeroPaddingFactor)
            self.im_psf.set_data(np.log(self.sim.tel.PSF / self.sim.tel.PSF.max()))
            self.im_psf.set_clim(vmin=-4, vmax=0)
            plt.draw()
            plt.show()
            plt.pause(0.001)


class Simulation:

    def __init__(self, param):
        self.tel = Telescope(resolution=param['resolution'],
                             diameter=param['diameter'],
                             samplingTime=param['samplingTime'],
                             centralObstruction=param['centralObstruction'])

        # create the Source object
        self.ngs = Source(optBand=param['opticalBand'],
                          magnitude=param['magnitude'])

        # combine the NGS to the telescope using '*' operator:
        self.ngs * self.tel

        # compute the associated diffraction limited PSF
        self.zeroPaddingFactor = 6  # zero padding factor for the FFT to prevent the effects of aliasing
        self.tel.computePSF(zeroPaddingFactor=self.zeroPaddingFactor)

        # create the Atmosphere object
        self.atm = Atmosphere(telescope=self.tel,
                              r0=param['r0'],
                              L0=param['L0'],
                              windSpeed=param['windSpeed'],
                              fractionalR0=param['fractionnalR0'],
                              windDirection=param['windDirection'],
                              altitude=param['altitude'])

        # initialize atmosphere
        self.atm.initializeAtmosphere(self.tel)
        self.atm.update()

        #
        self.tel + self.atm
        self.tel.computePSF(8)

        # mis-registrations object
        self.misReg = MisRegistration(param)
        # if no coordinates specified, create a cartesian dm
        self.dm = DeformableMirror(telescope=self.tel,
                                   nSubap=param['nSubaperture'],
                                   mechCoupling=param['mechanicalCoupling'],
                                   misReg=self.misReg)

        # make sure tel and atm are separated to initialize the PWFS
        self.tel - self.atm

        self.wfs = Pyramid(nSubap=param['nSubaperture'],
                           telescope=self.tel,
                           modulation=param['modulation'],
                           lightRatio=param['lightThreshold'],
                           n_pix_separation=param['n_pix_separation'],
                           psfCentering=param['psfCentering'],
                           postProcessing=param['postProcessing'])

        self.tel * self.wfs

    def calibrate_modal_cl(self,
                           n_modes: int = 300,
                           stroke: float = 1e-9,
                           n_measurements: int = 20,
                           noise: str = "off"):
        # create Zernike Object
        z = Zernike(self.tel, n_modes)

        # compute polynomials for given telescope
        z.computeZernike(self.tel)
        m2c_zernike = np.linalg.pinv(np.squeeze(self.dm.modes[self.tel.pupilLogical, :])) @ z.modes

        self.dm.coefs = m2c_zernike[:, :10]
        self.tel * self.dm

        calib_zernike = interactionMatrix(ngs=self.ngs,
                                          atm=self.atm,
                                          tel=self.tel,
                                          dm=self.dm,
                                          wfs=self.wfs,
                                          M2C=m2c_zernike,
                                          stroke=stroke,
                                          nMeasurements=n_measurements,
                                          noise=noise)

        return calib_zernike, m2c_zernike

    def run_cl(self, n_loop: int, gain_cl=0.6, *, photon_noise=True, display=False):
        self.wfs.cam.photonNoise = photon_noise

        # Calibrate
        calib_cl, m2c_cl = self.calibrate_modal_cl(n_modes=300)
        reconstructor = m2c_cl@calib_cl.M

        # combine telescope with atmosphere
        self.tel + self.atm

        # initialize DM commands
        self.dm.coefs = 0
        self.ngs * self.tel * self.dm * self.wfs

        # allocate memory to save data
        sr = np.zeros(n_loop)
        total = np.zeros(n_loop)
        residual = np.zeros(n_loop)
        wfs_signal = np.arange(0, self.wfs.nSignal) * 0

        if display:
            display_obj = SimDisplay(self, init=True)

        loop_it = tqdm.tqdm(range(n_loop))
        for i in loop_it:
            # update phase screens => overwrite tel.OPD and consequently tel.src.phase
            self.atm.update()

            # save phase variance
            total[i] = np.std(self.tel.OPD[np.where(self.tel.pupil > 0)]) * 1e9

            if display:
                display_obj.update_ol_psf()

            # propagate to the WFS with the CL commands applied
            self.tel * self.dm * self.wfs

            # save the DM OPD shape
            self.dm.coefs = self.dm.coefs - gain_cl * np.matmul(reconstructor, wfs_signal)

            # store the slopes after computing the commands => 2 frames delay
            wfs_signal = self.wfs.signal

            if display:
                display_obj.update()

            sr[i] = np.exp(-np.var(self.tel.src.phase[np.where(self.tel.pupil == 1)]))
            residual[i] = np.std(self.tel.OPD[np.where(self.tel.pupil > 0)]) * 1e9

            loop_it.set_description(f"Turbulence: {total[i]:1.5e} -- Residual: {residual[i]:1.5e}")

        return sr, total, residual, wfs_signal


def main():
    param = initializeParameterFile()
    sim = Simulation(param)
    sim.run_cl(n_loop=100, display=True)


if __name__ == "__main__":
    main()

import numpy as np
from scipy import linalg as lg
from scipy import signal as sg

from numpy.random import Generator, PCG64

try:
    import cupy as cp
    from cupyx.scipy import signal as csg

    global_gpu_flag = True

except ImportError or ModuleNotFoundError:
    print('CuPy is not found, using NumPy backend...')
    cp = np
    csg = sg
    global_gpu_flag = False

from OOPAO.tools.tools import set_binning

class LiFT:
    def __init__(self, tel,
                 basis,
                 det,
                 diversity_OPD:float,
                 iterations:int,
                 ang_pixel:float,
                 img_resolution:int,
                 numerical:bool):

        """
        LiFT: Linearized Focal Plane Technique is a type of Focal Plane Wavefront Sensor

        :param tel:
            Telescope object coupled with a Source object
        :param basis:
            Modal basis: Zernike, KL etc. - data cube with the different basis elements
        :param det:
            Detector object
        :param diversity_OPD: float
            2D array representing the Diversity Optical Path Difference
        :param iterations: int
            Maximum number of iterations allowed for LiFT algorithm
        :param ang_pixel:float
            Angular pixel size in mas of the detector - this parameter allows changing the PSF sampling
        :param img_resolution:
            Resolution of the PSF: if img_resolution = 10, then we have a PSF with 10*10 pixels
            When under noisy conditions or under the presence of high-order residuals, choosing a low img_resolution can
            help LiFT performance; This parameter defines the FoV
        :param numerical:bool
            If True, the interaction matrices of LiFT are calculated numerically; if False, which is the default value,
            the interaction matrices of LiFT are calculated analytically


        Examples of usage of this function are provided in the tutorial how_to_LiFT.ipynb
        """

        global global_gpu_flag
        self.tel            = tel
        self.det            = det
        self.basis          = basis
        self.diversity_OPD  = diversity_OPD
        self.iterations     = iterations
        self.ang_pixel      = ang_pixel  # mas
        self.img_resolution = img_resolution
        self.object         = None

        self.gpu = False and global_gpu_flag

        if self.gpu:
            self.diversity_OPD = cp.array(self.diversity_OPD, dtype=cp.float32)

        self.ang_pixel_rad      = self.ang_pixel/((180/np.pi)*3600*1000)
        self.zeroPaddingFactor  = (self.tel.src.wavelength / self.tel.D) * (1/(self.ang_pixel_rad))

        self.numerical = numerical


    def print_modes(self, A_vec):
        xp = cp if self.gpu else np
        for i in range(A_vec.shape[0]):
            val = A_vec[i]
            if val != None and not xp.isnan(val): val = xp.round(val, 4)
            print('Mode #', i, val)
            

    def obj_convolve(self, mat):
        xg = csg if self.gpu else sg
        if self.object is not None:
            return xg.convolve2d(mat, self.object, boundary='symm', mode='same') / self.object.sum()
        else:
            return mat


    def generateLIFTinteractionMatrices(self, coefs, modes_ids, flux_norm=1.0):
        xp = cp if self.gpu else np

        if isinstance(coefs, list):
            coefs = xp.array(coefs)

        initial_OPD = np.squeeze(self.basis @ coefs) + self.diversity_OPD

        H = []
        if not self.numerical:
            wavelength = self.tel.src.wavelength

            initial_amplitude = xp.sqrt(self.tel.pupilReflectivity * self.tel.src.nPhoton * flux_norm * self.tel.samplingTime * (
                        self.tel.D / self.tel.resolution) ** 2)
            # initial_amplitude = xp.sqrt(self.tel.src.fluxMap)
            k = 2 * xp.pi / wavelength

            initial_phase = k * initial_OPD

            _ = self.tel.PropagateField(initial_amplitude, initial_phase, self.zeroPaddingFactor,
                               self.img_resolution)
            Pd = xp.conj(self.tel.focal_EMF)

            H_spectral = []
            for i in modes_ids:

                aux_oversampling = self.tel.PropagateField(np.squeeze(
                    self.basis[:, :, i]) * initial_amplitude, initial_phase, \
                                                        self.zeroPaddingFactor, self.img_resolution)
                buf = self.tel.focal_EMF
                derivative = 2 * set_binning((xp.real(1j * buf * Pd)), aux_oversampling) * k
                derivative = self.obj_convolve(derivative)

                H_spectral.append(derivative.flatten())
            H.append(xp.vstack(H_spectral).T)
        else:
            delta = 1e-9  # [nm]
            H_spectral = []

            wavelength = self.tel.src.wavelength
            k = 2 * xp.pi / wavelength

            for i in modes_ids:
                self.tel.OPD = (np.squeeze(self.basis[:, :, i]) * delta) + initial_OPD

                self.tel.PropagateField(xp.sqrt(self.tel.src.fluxMap), k * self.tel.OPD,
                                      self.zeroPaddingFactor, self.img_resolution)
                tmp1 = self.tel.PSF * flux_norm

                self.tel.OPD = -(np.squeeze(self.basis[:, :, i]) * delta) + initial_OPD

                self.tel.PropagateField(xp.sqrt(self.tel.src.fluxMap), k * self.tel.OPD,
                                      self.zeroPaddingFactor, self.img_resolution)
                tmp2 = self.tel.PSF * flux_norm

                derivative = self.obj_convolve((tmp1 - tmp2) / 2 / delta)

                H_spectral.append(derivative.flatten())
            H.append(np.vstack(H_spectral).T)

        return xp.dstack(H).sum(axis=2)  # sum all spectral interaction matricies


    def Reconstruct(self, PSF_inp, R_n, mode_ids, A_0=None, verbous=False, optimize_norm='sum',check_convergence=True):
        """
        Function to reconstruct modal coefficients from the input PSF image using LIFT

        Parameters:
            PSF (ndarray):                   2-d array of the input PSF image to reconstruct.

            R_n (ndarray or string or None): The pixel weighting matrix for LIFT. It can be passed to the function.
                                             from outside, modeled ('model'), updated dynamically ('iterative'), or
                                             assumed to be just detector's readout noise ('None').
            A_0 (ndarray):                   initial assumtion for the coefficient values. In some sense, it acts as an additional
                                             phase diversity on top of the main phase diversity which is passed when class is initialized.

            verbous (bool):                  Set 'True' to print the intermediate reconstruction results.

            optimize_norm (string or None):  Recomputes the flux of the recontructed PSF iteratively. If 'None', the flux is not recomputed,
                                             this is recommended only if the target brightness is precisely known. In mosyt of the case it is
                                             recommended to switch it on. When 'sum', the reconstructed PSF is normalized to the sum of the pixels
                                             of the input PSF. If 'max', then the reconstructed PSF is normalized to the maximal value of the input PSF.

            ckeck_convergence (bool):        If True, we check both convergence criteria at each iteration, which gives the possibility to exit the cycle
                                             before reaching the maximum number of iterations (self.iterations). If False, the convergence criteria are ignored
                                             and the cycle continues until we reach the maximum number of iterations.

        """
        if self.gpu:
            xp = cp
            convert = lambda x: cp.asnumpy(x)
        else:
            xp = np
            convert = lambda x: x

        def PSF_from_coefs(coefs):

            OPD = np.squeeze(self.basis @ coefs)
            self.tel.OPD = self.diversity_OPD + OPD
            wavelength = self.tel.src.wavelength
            k = 2 * np.pi / wavelength
            # PSF = self.PropagateField(xp.sqrt(self.tel.src.fluxMap), k * self.tel.OPD,
            #                      return_intensity=True, oversampling=1)
            self.tel.PropagateField(xp.sqrt(self.tel.src.fluxMap), k * self.tel.OPD,
                                 self.zeroPaddingFactor, self.img_resolution)
            PSF = self.tel.PSF

            return self.obj_convolve(PSF)

        C = []  # optimization criterion
        Hs = []  # interaction matrices for every iteration
        P_MLs = []  # estimators for every iteration
        A_ests = []  # history of estimated coefficients

        PSF = xp.array(PSF_inp, dtype=xp.float32)
        modes = xp.sort(xp.array(mode_ids, dtype=xp.int32))

        # Account for the intial assumtion for the coefficients values
        if A_0 is None:
            # A_est = xp.zeros(modes.max().item() + 1, dtype=xp.float32)
            A_est = xp.zeros(self.basis.shape[-1], dtype=xp.float32)
        else:
            # A_est = xp.array(A_0, dtype=xp.float32)
            A_est = xp.zeros(self.basis.shape[-1], dtype=xp.float32)
            A_est[:len(A_0)] = xp.array(A_0, dtype=xp.float32)
        A_ests.append(xp.copy(A_est))

        def normalize_PSF(PSF_in):
            if optimize_norm is not None and optimize_norm is not False:
                if optimize_norm == 'max': return (PSF_in / PSF_in.max(), PSF_in.max())
                if optimize_norm == 'sum': return (PSF_in / PSF_in.sum(), PSF_in.sum())
            else: return (PSF_in, 1.0)

        PSF_0, flux_cap = normalize_PSF(PSF_from_coefs(A_est))  # initial PSF assumtion, normalized to 1.0

        flux_scale = 1.0
        if optimize_norm is not None and optimize_norm is not False:
            if optimize_norm == 'max': flux_scale = PSF.max()
            elif optimize_norm == 'sum': flux_scale = PSF.sum()

        PSF_cap = xp.copy(PSF_0) * flux_scale

        criterion = lambda i: xp.abs(C[i] - C[i - 1]) / C[i]
        coefs_norm = lambda v: xp.linalg.norm(v[modes], ord=2)

        def inverse_Rn(Rn):
            return 1. / xp.clip(Rn.flatten(), a_min=1e-6, a_max=Rn.max())

        if R_n is not None:
            if isinstance(R_n, str):  # basically if it's 'model' or 'iterative':
                inv_R_n = inverse_Rn(PSF_0 * flux_scale + self.det.readoutNoise ** 2)
            else:
                inv_R_n = inverse_Rn(xp.array(R_n))
        else:
            inv_R_n = inverse_Rn(xp.ones_like(PSF) * self.det.readoutNoise ** 2)

        for i in range(self.iterations):
            dI = (PSF - PSF_cap).flatten()

            C.append(xp.dot(dI * inv_R_n, dI))  # check the convergence
            if i > 0 and (criterion(i) < 1e-6 or coefs_norm(A_ests[i] - A_ests[i - 1]) < 1e-12):
                if verbous:
                    print('Criterion', criterion(i), 'is reached at iter.', i)
                if check_convergence == True:
                    break

            # Generate interaction matricies
            H = self.generateLIFTinteractionMatrices(A_est, modes, flux_scale / flux_cap)

            # Maximum likelyhood estimation
            P_ML = xp.linalg.pinv(H.T * inv_R_n @ H) @ H.T * inv_R_n
            d_A = P_ML @ dI
            A_est[modes] += d_A

            # Save the intermediate results for history
            Hs.append(H)
            P_MLs.append(P_ML)
            A_ests.append(xp.copy(A_est))

            if verbous:
                print('Criterion:', criterion(i))
                self.print_modes(d_A)
                print()

            # Update the PSF image with the estimated coefficients
            PSF_cap, flux_cap = normalize_PSF(PSF_from_coefs(A_est))
            PSF_cap *= flux_scale

            if isinstance(R_n, str):
                if R_n == 'iterative':
                    inv_R_n = inverse_Rn(PSF_cap + self.det.readoutNoise ** 2)

        history = {  # contains intermediate data saved at every iteration
            'P_ML': convert(xp.dstack(P_MLs)),
            'H': convert(xp.dstack(Hs)),
            'A_est': convert(xp.squeeze(xp.dstack(A_ests), axis=0)),
            'C': convert(xp.array(C))
        }
        return convert(A_est), convert(PSF_cap), history
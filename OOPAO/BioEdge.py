# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:48:49 2022

@author: cheritier
"""

import inspect
import multiprocessing
import sys
import time
from .tools.tools import warning
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sp
from joblib import Parallel, delayed

from OOPAO.Detector import Detector

try:
    #error
    import cupy as xp
except:
    import numpy as xp
    # print('NO GPU available!')

class BioEdge:
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def __init__(self,nSubap,telescope,modulation,grey_width,lightRatio, grey_length=False, postProcessing='slopesMaps',psfCentering=True, n_pix_separation = 2, calibModulation=50, n_pix_edge=None,extraModulationFactor=0,zeroPadding=None,pupilSeparationRatio=None,edgePixel = None,binning =1,nTheta_user_defined=None,userValidSignal=None,old_mask=False,rooftop = None,delta_theta = 0, user_modulation_path = None):
        """
        A Bi-O Edge object consists in defining a 2D phase mask located at the focal plane of the telescope to perform the Fourier Filtering of the EM-Field. 
        By default the Bi-O Edge detector is considered to be noise-free (for calibration purposes). These properties can be switched on and off on the fly (see properties)
        ************************** REQUIRED PARAMETERS **************************
        It requires the following parameters: 
        _ nSubap                : the number of subapertures (ie the diameter of the Bi-O Edge Pupils in pixels)
        _ telescope             : the telescope object to which the Bi-O Edge is associated. This object carries the phase, flux and pupil information
        _ modulation            : the Tip-Tilt modulation in [lambda/D] where lambda is the NGS wavelength and D the telescope diameter
        _ grey_width            : half width of the central area of the masks in [lambda/D] over which a gradient of transmissivity from 0 to 1 is applied.
        _ lightRatio            : criterion to select the valid subaperture based on flux considerations
        _ n_pix_separation      : number of pixels separating the Bi-O Edge Pupils in number of pixels of the detector    -- default value is 2 pixels
        _ n_pix_edge            : number of pixel at the edge of the Bi-O Edge Pupils in number of pixels of the detector -- default value is n_pix_separation's value
        _ postProcessing        : processing of the signals ('fullFrame' or 'slopesMaps')                               -- default value is 'slopesMaps'
        
        DEPRECATED PARAMETERS:
        _ pupilSeparationRatio  : Separation ratio of the PWFS pupils (Diameter/Distance Center to Center) -- DEPRECATED -> use n_pix_separation instead)
        _ edgePixel             : number of pixel at the edge of the Bi-O Edge Pupils
        
        ************************** OPTIONAL PARAMETERS **************************
        
        _ calibModulation       : Defines the modulation used to select the valid subapertures  -- default value is 50 lambda/D
        _ extraModulationFactor : Extra Factor to increase/reduce the number of modulation point (extraModulationFactor = 1 means 4 modulation points added, 1 for each quadrant) 
        _ psfCentering          : If False, the Bi-O Edge mask is centered on 1 pixel, if True, the Bi-O Edge mask is centered on 4 pixels -- default value is True 
        _ binning               : binning factor of the PWFS detector signals -- default Value is 1
        _ nTheta_user_defined   : user-defined number of Tip/Tilt modulation points 
        _ delta_theta           : delta angle for the modulation points, default value is 0 (on the edge between two sides of the Bi-O Edge)
        _ userValidSignal       : user-defined valid pixel mask for the signals computation
        _ rooftop               : if different to None, allows to compute a two-sided Bi-O Edge ("V" corresponds to a vertical split, "H" corresponds to an horizontal split)  
        _ zeroPadding           : User-defined zero-padding value in pixels that will be added to each side of the arrays. Consider using the n_pix_edge parameter that allows to do the same thing. 
        _ pupilReflectivcty     : Defines the reflectivity of the Telescope object. If not set to 1, it can be input as a 2D map of uneven reflectivy correspondong to the pupil mask. 
        _ user_modulation_path  : user-defined modulation path ( a list of [x,y] coordinates in lambda/D units is expected)
            
        ************************** PROPAGATING THE LIGHT TO THE Bi-O Edge OBJECT **************************
        The light can be propagated from a telescope object tel through the Bi-O Edge object wfs using the * operator:        
        _ tel*wfs
        This operation will trigger:
            _ propagation of the tel.src light through the PWFS detector (phase and flux)
            _ binning of the Bi-O Edge signals
            _ addition of eventual photon noise and readout noise
            _ computation of the Bi-O Edge signals
        
    
        ************************** PROPERTIES **************************
        
        The main properties of a Telescope object are listed here: 
        _ wfs.nSignal                    : the length of the signal measured by the Bi-O Edge
        _ wfs.signal                     : signal measured by the Bi-O Edge of length wfs.nSignal
        _ wfs.signal_2D                  : 2D map of the signal measured by the Bi-O Edge
        _ wfs.apply_shift_wfs            : apply a tip tilt to each quadrant to move the Bi-O Edge pupils
        _ wfs.random_state_photon_noise  : a random state cycle can be defined to reproduces random sequences of noise -- default is based on the current clock time 
        _ wfs.random_state_readout_noise : a random state cycle can be defined to reproduces random sequences of noise -- default is based on the current clock time   
        _ wfs.random_state_background    : a random state cycle can be defined to reproduces random sequences of noise -- default is based on the current clock time   
        _ wfs.fov                        : Field of View of the Bi-O Edge in arcsec

        the following properties can be updated on the fly:
            _ wfs.modulation            : update the modulation radius and update the reference signal
            _ wfs.cam.photonNoise       : Photon noise can be set to True or False
            _ wfs.cam.readoutNoise      : Readout noise can be set to True or False
            _ wfs.backgroundNoise       : Background noise can be set to True or False
            _ wfs.lightRatio            : reset the valid subaperture selection considering the new value
        
        """        
        try:
            import cupy as xp
            self.gpu_available = True
            self.convert_for_gpu = xp.asarray
            self.convert_for_numpy = xp.asnumpy
            self.nJobs = 1
            self.mempool = xp.get_default_memory_pool()
            from .tools.tools import get_gpu_memory
            self.mem_gpu = get_gpu_memory()

            print('GPU available!')
            for i in range(len(self.mem_gpu)):
                print('GPU device '+str(i)+' : ' +
                      str(self.mem_gpu[i]/1024) + 'GB memory')
        except:
            import numpy as xp

            def no_function(input_matrix):
                return input_matrix
            self.gpu_available = False
            self.convert_for_gpu = no_function
            self.convert_for_numpy = no_function

        OOPAO_path = [s for s in sys.path if "OOPAO" in s]
        l = []
        for i in OOPAO_path:
            l.append(len(i))
        path = OOPAO_path[np.argmin(l)]
        precision = np.load(path+'/precision_oopao.npy')
        if precision == 64:
            self.precision = np.float64
        else:
            self.precision = np.float32
        if self.precision is xp.float32:
            self.precision_complex = xp.complex64
        else:
            self.precision_complex = xp.complex128
        
        # initialize the Bi-O Edge Object 
        self.telescope                  = telescope                                         # telescope attached to the wfs
        if self.telescope.resolution/nSubap <2 or (self.telescope.resolution/nSubap)%2 !=0:
            raise ValueError('The resolution should be an even number and be a multiple of 2**i where i>=2')
        if self.telescope.src is None:
            raise AttributeError('The telescope was not coupled to any source object! Make sure to couple it with an src object using src*tel')
        # delta theta in degree to change the position of the modulation point (default is 0 <=> modulation point on the edge of two sides of the pyramid)
        self.delta_theta = delta_theta
        # user defined number of modulation point
        self.nTheta_user_defined = nTheta_user_defined
        # Extra Factor to increase/reduce the number of modulation point (extraModulationFactor = 1 means 4 modulation points added, 1 for each quadrant)
        self.extraModulationFactor = extraModulationFactor
        # Number of subaperture
        self.nSubap = nSubap
        # Number of pixel on the edges of the PWFS pupils
        self.edgePixel = n_pix_edge
        # Value used for the centering for the slopes-maps computation
        self.centerPixel = 0
        # type of processing of the signals (see self.postProcessing)
        self.postProcessing = postProcessing
        # user defined mask for the valid pixel selection
        self.userValidSignal = userValidSignal
        # tag for the PSF centering on  or 4 pixels
        self.psfCentering = psfCentering
        # binning factor for the detector
        self.binning = binning
        self.old_mask = old_mask
        # half width of the grey area of the mask in [l/D]
        self.grey_width                 = grey_width                                        # half width of the grey area of the mask
        self.grey_length                = grey_length                                       # half length of the grey area of the mask (set to false by default -> traditional Grey Bi-O-Edge)
        self.random_state_photon_noise  = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        self.random_state_readout_noise = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        self.random_state_background    = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        self.user_modulation_path       = user_modulation_path                              # user defined modulation path
        self.pupilSeparationRatio       = pupilSeparationRatio                              # Separation ratio of the PWFS pupils (Diameter/Distance Center to Center) -- DEPRECATED -> use n_pix_separation instead)
        self.joblib_prefer_masks        = 'threads'
        self.nJobs_masks                = 1 
        self.weight_vector = None
        if edgePixel is not None:
            raise AttributeError(
                'The use of the edgePixel property has been deprecated. Consider using n_pix_edge instead')
        if pupilSeparationRatio is not None:
            raise AttributeError(
                'The use of the pupilSeparationRatio property has been deprecated. Consider using n_pix_separation instead')
        else:
            self.n_pix_separation = n_pix_separation
            self.sx = [0, 0, 0, 0]
            self.sy = [0, 0, 0, 0]
        if n_pix_edge is None:
            self.n_pix_edge = self.n_pix_separation//2
        else:
            self.n_pix_edge = n_pix_edge
            if n_pix_edge != self.n_pix_separation//2:
                warning('The recommanded value for n_pix_edge is ' +
                      str(self.n_pix_separation//2) + ' instead of ' + str(n_pix_edge))
        if self.gpu_available:
            self.joblib_setting = 'processes'
        else:
            self.joblib_setting = 'threads'
        self.rooftop = rooftop    

        if zeroPadding is not None:
            raise AttributeError(
                'The use of the zeroPadding property has been deprecated')
        # Case where the zero-padding is not specificed => taking the smallest value ensuring to get edgePixel space from the edge.
        self.nRes = int((self.nSubap*2+self.n_pix_separation +
                        self.n_pix_edge*2)*self.telescope.resolution/self.nSubap)
        # zero-Padding Factor
        self.zeroPaddingFactor = self.nRes/self.telescope.resolution
        # zero-Padding Factor
        self.zeroPadding = (self.nRes - self.telescope.resolution)//2

        # Tag of the object
        self.tag                        = 'bioEdge'
        
        # WFS detector object (see Detector class)
        self.cam = Detector(round(2*nSubap*self.zeroPaddingFactor))
        # WFS focal plane detector object (see Detector class)
        self.focal_plane_camera         = Detector(self.nRes)
        self.focal_plane_camera.is_focal_plane_camera = True
        # Light ratio for the valid pixels selection 
        self.lightRatio = lightRatio
        if calibModulation >= self.telescope.resolution/2:
            self.calibModulation = self.telescope.resolution/2 - 1
        else:
            # Modulation used for the valid pixel selection
            self.calibModulation = calibModulation
        # Flag for the initialization of the WFS
        self.isInitialized = False
        # Flag for the initialization of the WFS
        self.isCalibrated = False
        # delta Tip for the modulation
        self.delta_Tip = 0
        # delta Tilt for the modulation
        self.delta_Tilt = 0
        # Center of the zero-Padded array
        self.center = self.nRes//2
        self.supportPadded = self.convert_for_gpu(np.pad(self.telescope.pupil.astype(self.precision_complex()),
                                                         ((self.zeroPadding, self.zeroPadding),
                                                          (self.zeroPadding, self.zeroPadding)), 'constant'))        
        # case where a spatial filter is considered
        self.spatialFilter = None
        self.fov = 206265*self.nRes/self.zeroPaddingFactor * \
            (self.telescope.src.wavelength/self.telescope.D)  # fov in arcsec
        self.fov_l_d = self.nRes/self.zeroPaddingFactor  # fov in arcsec

        n_cpu = multiprocessing.cpu_count()
        # joblib settings for parallization
        if self.gpu_available is False:
            if n_cpu > 16:
                self.nJobs                      = 32                                                                 # number of jobs for the joblib package
            else:
                self.nJobs                      = 8
            self.n_max = 20*500
        else:
            # quantify GPU max memory usage
            A = np.ones([self.nRes,self.nRes]) + 1j*np.ones([self.nRes,self.nRes])
            self.n_max = int(0.75*(np.min(self.mem_gpu)/1024)/(A.nbytes/1024/1024/1024))
            del A

        # Prepare the Tip Tilt for the modulation -- normalized to apply the modulation in terms of lambda/D
        [self.Tip, self.Tilt] = np.meshgrid(np.linspace(-np.pi, np.pi, self.telescope.resolution),
                                            np.linspace(-np.pi, np.pi, self.telescope.resolution))
        self.Tilt *= self.telescope.pupil
        self.Tip *= self.telescope.pupil

        # compute the phasor to center the PSF on 4 pixels
        [xx, yy] = np.meshgrid(np.linspace(0, self.nRes-1, self.nRes),
                               np.linspace(0, self.nRes-1, self.nRes))
        # phasor for the FFT centering
        self.phasor = self.convert_for_gpu(np.exp(-(1j*np.pi*(self.nRes+1)/self.nRes)*(xx+yy)))
        
        # initialize the reference signal and units 
        self.slopesUnits                = 1     
        self.referenceSignal            = 0
        self.referenceSignal_2D         = 0
        self.referenceBioFrame          = 0 
        self.modulation                 = modulation                                        # Modulation radius (in lambda/D)
        # Creating the PWFS mask
        self.mask_computation()
        # Select the valid pixels
        print('Selection of the valid pixels...')
        self.initialization(self.telescope)
        print('Acquisition of the reference slopes and units calibration...')
        # set the modulation radius and propagate light
        self.modulation = modulation
        self.wfs_calibration(self.telescope)
        self.telescope.resetOPD()
        self.wfs_measure(phase_in=self.telescope.src.phase)
        self.print_properties()

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS INITIALIZATION PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def mask_computation(self,sx = [0,0,0,0],sy = [0,0,0,0]):
        print('Bio-Edge Mask initialization...')        
        A = np.zeros([self.nRes,self.nRes],dtype=self.precision_complex)
        B = np.zeros([self.nRes,self.nRes],dtype=self.precision_complex)
        C = np.copy(A.T)
        D = np.copy(B.T)
        
        n_tot = int((self.nSubap*2+self.n_pix_separation+self.n_pix_edge*2) * self.telescope.resolution/self.nSubap)

        # mask centered on 4 pixel
        lim     = np.pi
        norma = (self.nSubap *2)*(self.telescope.resolution/self.nSubap) * (1/(self.nSubap*2))

        # create a Tip/Tilt combination for each quadrant
        [Tip, Tilt] = np.meshgrid(np.linspace(-lim, lim, n_tot),
                                  np.linspace(-lim, lim, n_tot))        
        
        if self.grey_width == 0:
            BW                                = np.zeros([self.nRes])
            BW[0:self.nRes//2]                = 1.
            
        elif self.grey_length != False:
            r_grey              = self.zeroPaddingFactor
            x                   = (np.asarray(range(self.nRes))-self.nRes//2)/self.grey_width
            r                                 =  int(np.round(r_grey*self.grey_width))
            BW                                = np.zeros([self.nRes])
            BW[0:self.nRes//2]                = 1.
            BW[self.nRes//2-r:self.nRes//2+r] = -1.*x[self.nRes//2-r:self.nRes//2+r]*1/(2.*r_grey)+0.5
        else:
            r_grey              = self.zeroPaddingFactor
            x                   = (np.asarray(range(self.nRes))-self.nRes//2)/self.grey_width
            r                                 =  int(np.round(r_grey*self.grey_width))
            BW                                = np.zeros([self.nRes])
            BW[0:self.nRes//2]                = 1.
            BW[self.nRes//2-r:self.nRes//2+r] = -1.*x[self.nRes//2-r:self.nRes//2+r]*1/(2.*r_grey)+0.5
        
        [X,Y]   = np.meshgrid(BW,BW)
        A[:,:]  = np.sqrt(X)
        B       = np.sqrt(1-A**2)
        C       = np.copy(A.T)
        D       = np.copy(B.T)
        
        if len(sx)==4 and len(sy)==4:
            A_TT =( sx[0]*Tip*norma + sy[0]*Tilt*norma) 
            B_TT =( sx[1]*Tip*norma + sy[1]*Tilt*norma)
            C_TT =( sx[2]*Tip*norma + sy[2]*Tilt*norma) 
            D_TT =( sx[3]*Tip*norma + sy[3]*Tilt*norma)
            
            
            A = A*np.exp(1j*A_TT)
            B = B*np.exp(1j*B_TT)
            C = C*np.exp(1j*C_TT)
            D = D*np.exp(1j*D_TT)
        
            self.mask = [A,B,C,D]   
            self.mask_TT = [A_TT,B_TT,C_TT,D_TT]   

        else:
            self.mask = [A,B,C,D]   
        # convert for GPU
        self.mask = self.convert_for_gpu(self.mask)
        # Save a copy of the initial mask
        self.initial_mask = np.copy(self.mask)
        print('Done!')
             
    def apply_shift_wfs(self,sx =None,sy=None,mis_reg = None,units = 'pixels'):
        if sx is None:
            sx = 0
        if sy is None:
            sy = 0
        if mis_reg is not None:
            sx = [mis_reg.dX_1,mis_reg.dX_2,mis_reg.dX_3,mis_reg.dX_4]
            sy = [mis_reg.dY_1,mis_reg.dY_2,mis_reg.dY_3,mis_reg.dY_4] 
        # apply a TIP/TILT of the PWFS mask to shift the pupils
        if units == 'pixels':
            factor = 1
        if units == 'm':
            factor = 1/(self.telescope.pixelSize*(self.telescope.resolution/self.nSubap))
        # sx and sy are the units of displacements in pixels
        if np.isscalar(sx) and np.isscalar(sy):
            shift_x = [factor*sx,factor*sx,factor*sx,factor*sx]
            shift_y = [factor*sy,factor*sy,factor*sy,factor*sy]
        else:
            if len(sx)==4 and len(sy)==4:
                shift_x = []
                shift_y = []
                [shift_x.append(i_x*factor) for i_x in sx]
                [shift_y.append(i_y*factor) for i_y in sy]
            else:
                raise ValueError('Wrong size for sx and/or sy, a list of 4 values is expected.')
        if np.max(np.abs(shift_x))>self.n_pix_edge or np.max(np.abs(shift_y))>self.n_pix_edge:
            warning('The Bi-O Edge pupils have been shifted outside of the detector!! Wrapping of the signal is currently occuring!!')

        self.sx = shift_x
        self.sy = shift_y
        self.mask_computation(self.sx,self.sy)  
        # print('Updating the reference slopes and Wavelength Calibration for the new modulation...')
        self.slopesUnits                = 1     
        self.referenceSignal            = 0
        self.referenceSignal_2D         = 0
        self.wfs_calibration(self.telescope)

    def get_phase_mask(self, resolution, n_subap, n_pix_separation, n_pix_edge, psf_centering=False, sx=[0, 0, 0, 0], sy=[0, 0, 0, 0]):
        # size of the mask in pixel
        n_tot = int((n_subap*2+n_pix_separation+n_pix_edge*2) * self.telescope.resolution/self.nSubap)

        # normalization factor for the Tip/Tilt
        norma = (n_subap + n_pix_separation) * (self.telescope.resolution/self.nSubap)/2

        # support for the mask
        m = np.zeros([n_tot, n_tot])
        if psf_centering:
            # mask centered on 4 pixel
            lim = np.pi/2
            # create a Tip/Tilt combination for each quadrant
            [Tip, Tilt] = np.meshgrid(np.linspace(-lim, lim, n_tot//2),
                                      np.linspace(-lim, lim, n_tot//2))

            m[:n_tot//2, :n_tot//2] = Tip * (1 + sx[0]/(n_subap+n_pix_separation/2))*norma + Tilt * (1 - sy[0]/(n_subap+n_pix_separation/2))*norma
            m[:n_tot//2, -n_tot//2:] = -Tip * (1 - sx[1]/(n_subap+n_pix_separation/2))*norma + Tilt * (1 - sy[1]/(n_subap+n_pix_separation/2))*norma
            m[-n_tot//2:, -n_tot//2:] = -Tip * (1 - sx[2]/(n_subap+n_pix_separation/2))*norma + -Tilt * (1 + sy[2]/(n_subap+n_pix_separation/2))*norma
            m[-n_tot//2:, :n_tot//2] = Tip * (1 + sx[3]/(n_subap+n_pix_separation/2))*norma + -Tilt * (1 + sy[3]/(n_subap+n_pix_separation/2))*norma

        else:
            # mask centered on 1 pixel => different normalization for each Tip/tilt
            d_pix = (np.pi) / (n_tot)     # size of a pixel in angle
            lim_p = np.pi/2
            lim_m = np.pi/2 - 2*d_pix

            # create a Tip/Tilt combination for each quadrant
            [Tip_1, Tilt_1] = np.meshgrid(np.linspace(-lim_p, lim_p, n_tot//2 + 1),
                                          np.linspace(-lim_p, lim_p, n_tot//2 + 1))
            [Tip_2, Tilt_2] = np.meshgrid(np.linspace(-lim_p, lim_p, n_tot//2 + 1), np.linspace(-lim_m, lim_m, n_tot//2 - 1))
            [Tip_3, Tilt_3] = np.meshgrid(np.linspace(-lim_m, lim_m, n_tot//2 - 1), np.linspace(-lim_m, lim_m, n_tot//2 - 1))
            [Tip_4, Tilt_4] = np.meshgrid(np.linspace(-lim_m, lim_m, n_tot//2 - 1), np.linspace(-lim_p, lim_p, n_tot//2 + 1))

            m[:n_tot//2 + 1, :n_tot//2+1] = Tip_1 * (1 + sx[0]/(n_subap+n_pix_separation/2))*norma + Tilt_1 * (1 - sy[0]/(n_subap+n_pix_separation/2))*norma
            m[:n_tot//2 + 1, -n_tot//2+1:] = -Tip_4 * (1 - sx[1]/(n_subap+n_pix_separation/2))*norma + Tilt_4 * (1 - sy[1]/(n_subap+n_pix_separation/2))*norma
            m[-n_tot//2 + 1:, -n_tot//2 + 1:] = -Tip_3 * (1 - sx[2]/(n_subap+n_pix_separation/2))*norma + -Tilt_3 * (1 + sy[2]/(n_subap+n_pix_separation/2))*norma
            m[-n_tot//2 + 1:, :n_tot//2 + 1] = Tip_2 * (1 + sx[3]/(n_subap+n_pix_separation/2))*norma + -Tilt_2 * (1 + sy[3]/(n_subap+n_pix_separation/2))*norma

        return -m  # sign convention for backward compatibility

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
     
    def initialization(self,telescope):
        telescope.resetOPD()
        if self.userValidSignal is None:
            print('The valid pixel are selected on flux considerations')
            self.modulation = self.calibModulation                  # set the modulation to a large value   
            self.wfs_measure(phase_in=self.telescope.src.phase)
            # save initialization frame
            self.initFrame = self.cam.frame
            
            # save the number of signals depending on the case    
            if self.postProcessing == 'slopesMaps' or self.postProcessing == 'slopesMaps_incidence_flux':
                # select the valid pixels of the detector according to the flux (case slopes-maps)
                I1 = self.grabQuadrant(1)
                I2 = self.grabQuadrant(2)
                I3 = self.grabQuadrant(3)
                I4 = self.grabQuadrant(4)
                
                # sum of the 4 quadrants
                self.I4Q            = I1+I2+I3+I4
                # valid pixels to consider for the slopes-maps computation
                self.validI4Q       = (self.I4Q>=self.lightRatio*self.I4Q.max()) 
                self.validSignal    = np.concatenate((self.validI4Q,self.validI4Q))
                self.nSignal        = int(np.sum(self.validSignal))
                
            if self.postProcessing == 'fullFrame' or self.postProcessing == 'fullFrame_incidence_flux':
                # select the valid pixels of the detector according to the flux (case full-frame)
                self.validSignal = (self.initFrame>=self.lightRatio*self.initFrame.max())   
                self.nSignal        = int(np.sum(self.validSignal))
        else:
            print('You are using a user-defined mask for the selection of the valid pixel')
            if self.postProcessing == 'slopesMaps' or self.postProcessing == 'slopesMaps_incidence_flux':
                
                # select the valid pixels of the detector according to the flux (case full-frame)
                self.validI4Q       =  self.userValidSignal
                self.validSignal    = np.concatenate((self.validI4Q,self.validI4Q))
                self.nSignal        = int(np.sum(self.validSignal))
                
            if self.postProcessing == 'fullFrame' or self.postProcessing == 'fullFrame_incidence_flux':            
                self.validSignal    = self.userValidSignal  
                self.nSignal        = int(np.sum(self.validSignal))
                    
        # Tag to indicate that the wfs is initialized
        self.isInitialized = True
            
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS CALIBRATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        
    def wfs_calibration(self,telescope):
        # reference slopes acquisition 
        telescope.OPD = telescope.pupil.astype(float)
        # compute the refrence slopes
        self.wfs_measure(phase_in=self.telescope.src.phase)
        self.referenceSignal_2D,self.referenceSignal = self.signalProcessing()
      
        # 2D reference Frame before binning with detector
        self.referenceBioFrame         = np.copy(self.bioFrame)
        if self.isCalibrated is False:
            print('WFS calibrated!')
        self.isCalibrated= True
        telescope.OPD = telescope.pupil.astype(float)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bio TRANSFORM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def convolution(self,mask,em_field_ft):
            # case with mask centered on 4 pixels
            if self.psfCentering:
                em_field_pwfs   = np.fft.ifft2(em_field_ft*mask)
                I               = np.abs(em_field_pwfs)**2
            # case with mask centered on 1 pixel
            else:
                em_field_pwfs   = np.fft.ifft2(em_field_ft*mask)
                I               = np.abs(em_field_pwfs)**2
            self.modulation_camera_em.append(em_field_ft)
            return I 
        
    def bio_edge_transform(self,phase_in):
        # copy of the support for the zero-padding
        support = self.supportPadded.copy()
        # em field corresponding to phase_in
        if np.ndim(self.telescope.OPD)==2:  
            if self.modulation==0:
                em_field     = 0.5*self.maskAmplitude*np.exp(1j*(phase_in))
            else:
                em_field     = 0.5*self.maskAmplitude*np.exp(1j*(self.convert_for_gpu(self.telescope.src.phase)+phase_in))
        else:
            em_field     = 0.5*self.maskAmplitude*np.exp(1j*phase_in)
        # zero-padding for the FFT computation
        support[self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2,self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2] = em_field
        del em_field
        # case with mask centered on 4 pixels
        if self.psfCentering:
            em_field_ft     = xp.fft.fft2(support*self.phasor)
        # case with mask centered on 1 pixel
        else:
            em_field_ft     = xp.fft.fftshift(xp.fft.fft2(support)) 

        support_list = [em_field_ft,em_field_ft,em_field_ft,em_field_ft]

        frame = np.zeros([2*self.nRes,2*self.nRes])
                         
        def job_loop_mask():
            Q = Parallel(n_jobs=self.nJobs_masks,prefer=self.joblib_prefer_masks)(delayed(self.convolution)(i,j) for i,j in zip(self.mask,support_list))
            return Q 
        
        out  = self.convert_for_numpy(xp.asarray(job_loop_mask()))
        
        
        count = 0
        
        for i in range(2):
            for j in range(2):
                frame[i*self.nRes:(i+1)*self.nRes,j*self.nRes:(j+1)*self.nRes] = out[count,:,:]
                count+=1    
        return frame    

    def setPhaseBuffer(self,phaseIn):
        B = self.phaseBuffModulationLowres_CPU+phaseIn
        return B    
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bio PROPAGATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    def wfs_measure(self,phase_in=None):
        if phase_in is not None:
            self.telescope.src.phase = phase_in
        # mask amplitude for the light propagation
        self.maskAmplitude = self.convert_for_gpu(self.telescope.pupilReflectivity)
        
        if self.spatialFilter is not None:
            if np.ndim(phase_in)==2:
                support_spatial_filter = np.copy(self.supportPadded)   
                em_field = self.maskAmplitude*np.exp(1j*(self.telescope.src.phase))
                support_spatial_filter[self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2,self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2] = em_field
                self.em_field_spatial_filter       = (np.fft.fft2(support_spatial_filter*self.phasor))
                self.pupil_plane_spatial_filter    = (np.fft.ifft2(self.em_field_spatial_filter*self.spatialFilter))

        # modulation camera
        self.modulation_camera_em=[]

        if self.modulation==0:
            if np.ndim(phase_in)==2:
                self.bioFrame = self.convert_for_numpy(self.bio_edge_transform(self.convert_for_gpu(self.telescope.src.phase)))              
                self*self.cam
                if self.isInitialized and self.isCalibrated:
                    self.bioSignal_2D,self.bioSignal=self.signalProcessing()
            else:
                nModes = phase_in.shape[2]
                # move axis to get the number of modes first
                self.phase_buffer = self.convert_for_gpu(np.moveaxis(self.telescope.src.phase,-1,0))
                
                #define the parallel jobs
                def job_loop_multiple_modes_non_modulated():
                    Q = Parallel(n_jobs=self.nJobs,prefer=self.joblib_setting)(delayed(self.bio_edge_transform)(i) for i in self.phase_buffer)
                    return Q 
                # apply the bio transform in parallel
                maps = job_loop_multiple_modes_non_modulated()
                
                self.bioSignal_2D    = np.zeros([self.validSignal.shape[0],self.validSignal.shape[1],nModes])
                self.bioSignal       = np.zeros([self.nSignal,nModes])
                
                for i in range(nModes):
                    self.bioFrame = self.convert_for_numpy(maps[i])
                    self*self.cam
                    if self.isInitialized:
                        self.bioSignal_2D[:,:,i],self.bioSignal[:,i] = self.signalProcessing()             
                del maps

        else:
            if np.ndim(phase_in)==2:       
                n_max_ = self.n_max
                if self.nTheta>n_max_:
                    # break problem in pieces:                     
                    nCycle = int(np.ceil(self.nTheta/n_max_))
                    # print(self.nTheta)
                    maps = self.convert_for_numpy(xp.zeros([self.nRes,self.nRes]))
                    for i in range(nCycle):
                        if self.gpu_available:
                            try:
                                self.mempool = xp.get_default_memory_pool()
                                self.mempool.free_all_blocks()
                            except:
                                print('could not free the memory')
                        if i<nCycle-1:
                            def job_loop_single_mode_modulated():
                                Q = Parallel(n_jobs=self.nJobs,prefer=self.joblib_setting)(delayed(self.bio_edge_transform)(i) for i in self.convert_for_gpu(self.phaseBuffModulationLowres[i*n_max_:(i+1)*n_max_,:,:]))
                                return Q 
                            maps+=self.convert_for_numpy(xp.sum(xp.asarray(job_loop_single_mode_modulated()),axis=0))
                        else:
                            def job_loop_single_mode_modulated():
                                Q = Parallel(n_jobs=self.nJobs,prefer=self.joblib_setting)(delayed(self.bio_edge_transform)(i) for i in self.convert_for_gpu(self.phaseBuffModulationLowres[i*n_max_:,:,:]))
                                return Q 
                            maps+=self.convert_for_numpy(xp.sum(xp.asarray(job_loop_single_mode_modulated()),axis=0))
                    self.bioFrame=maps/self.nTheta
                    del maps
                else:
                    #define the parallel jobs
                    def job_loop_single_mode_modulated():
                        Q = Parallel(n_jobs=self.nJobs,prefer=self.joblib_setting)(delayed(self.bio_edge_transform)(i) for i in self.phaseBuffModulationLowres)
                        return Q 
                    # apply the bio transform in parallel
                    self.maps=xp.asarray(job_loop_single_mode_modulated())
                    # compute the sum of the bio frames for each modulation points
                    if self.weight_vector is None:
                        self.bioFrame=self.convert_for_numpy(xp.sum((self.maps),axis=0))/self.nTheta
                    else:
                        weighted_map = np.reshape(self.maps,[self.nTheta,self.nRes**2])
                        self.weighted_map = np.diag(self.weight_vector)@weighted_map
                        self.bioFrame= np.reshape(self.convert_for_numpy(xp.sum((self.weighted_map),axis=0))/self.nTheta,[self.nRes,self.nRes])
                        

                #propagate to the detector
                self*self.cam
                
                if self.isInitialized and self.isCalibrated:
                    self.bioSignal_2D,self.bioSignal=self.signalProcessing()
            else:
                if np.ndim(phase_in)==3:
                    nModes = phase_in.shape[2]
                    # move axis to get the number of modes first
                    self.phase_buffer = np.moveaxis(self.telescope.src.phase,-1,0)

                    def jobLoop_setPhaseBuffer():
                        Q = Parallel(n_jobs=self.nJobs,prefer=self.joblib_setting)(delayed(self.setPhaseBuffer)(i) for i in self.phase_buffer)
                        return Q                   
                    
                    self.phaseBuffer    = (np.reshape(np.asarray(jobLoop_setPhaseBuffer()),[nModes*self.nTheta,self.telescope.resolution,self.telescope.resolution]))
                    n_measurements      = nModes*self.nTheta
                    n_max               = self.n_max
                    n_measurement_max   = int(np.floor(n_max/self.nTheta))
                    maps                = xp.zeros([n_measurements,self.nRes,self.nRes])
                    
                    if n_measurements >n_max:
                        nCycle = int(np.ceil(nModes/n_measurement_max))
                        for i in range(nCycle):
                            if self.gpu_available:
                                try:
                                    self.mempool = xp.get_default_memory_pool()
                                    self.mempool.free_all_blocks()
                                except:
                                    print('could not free the memory')
                            if i<nCycle-1:
                                def job_loop_multiple_mode_modulated():
                                    Q = Parallel(n_jobs=self.nJobs,prefer=self.joblib_setting)(delayed(self.bio_edge_transform)(i) for i in self.convert_for_gpu(self.phaseBuffer[i*n_measurement_max*self.nTheta:(i+1)*n_measurement_max*self.nTheta,:,:]))
                                    return Q 
                                maps[i*n_measurement_max*self.nTheta:(i+1)*n_measurement_max*self.nTheta,:,:] = xp.asarray(job_loop_multiple_mode_modulated())
                            else:
                                def job_loop_multiple_mode_modulated():
                                    Q = Parallel(n_jobs=self.nJobs,prefer=self.joblib_setting)(delayed(self.bio_edge_transform)(i) for i in self.convert_for_gpu(self.phaseBuffer[i*n_measurement_max*self.nTheta:,:,:]))
                                    return Q 
                                maps[i*n_measurement_max*self.nTheta:,:,:] = xp.asarray(job_loop_multiple_mode_modulated())
                        self.bufferBioFrames = self.convert_for_numpy(maps)
                        del self.phaseBuffer
                        del maps
                        if self.gpu_available:
                            try:
                                self.mempool = xp.get_default_memory_pool()
                                self.mempool.free_all_blocks()
                            except:
                                print('could not free the memory')
                    else:
                        def job_loop_multiple_mode_modulated():
                            Q = Parallel(n_jobs=self.nJobs,prefer=self.joblib_setting)(delayed(self.bio_edge_transform)(i) for i in self.convert_for_gpu(self.phaseBuffer))
                            return Q 
                        
                        self.bufferBioFrames  = self.convert_for_numpy(xp.asarray(job_loop_multiple_mode_modulated()))
                        
                    self.bioSignal_2D     = np.zeros([self.validSignal.shape[0],self.validSignal.shape[1],nModes])
                    self.bioSignal        = np.zeros([self.nSignal,nModes])
                    
                    for i in range(nModes):
                        self.bioFrame = xp.sum(self.bufferBioFrames[i*(self.nTheta):(self.nTheta)+i*(self.nTheta)],axis=0)/self.nTheta
                        self*self.cam
                        if self.isInitialized:
                            self.bioSignal_2D[:,:,i],self.bioSignal[:,i] = self.signalProcessing()                   
                    del self.bufferBioFrames      
                else:
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    print('Error - Wrong dimension for the input phase. Aborting....')
                    print('Aborting...')
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    sys.exit(0)
                if self.gpu_available:
                    try:
                        self.mempool = xp.get_default_memory_pool()
                        self.mempool.free_all_blocks()
                    except:
                        print('could not free the memory')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS SIGNAL PROCESSING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
            
    def signalProcessing(self,cameraFrame=None):
        if cameraFrame is None:
            cameraFrame=self.cam.frame
        if self.postProcessing == 'slopesMaps':
            # slopes-maps computation
            I1              = self.grabQuadrant(1,cameraFrame=0)*self.validI4Q
            I2              = self.grabQuadrant(2,cameraFrame=0)*self.validI4Q
            I3              = self.grabQuadrant(3,cameraFrame=0)*self.validI4Q
            I4              = self.grabQuadrant(4,cameraFrame=0)*self.validI4Q
            # global normalisation
            I4Q        = I1+I2+I3+I4
            norma      = np.mean(I4Q[self.validI4Q])
            # slopesMaps computation cropped to the valid pixels
            Sx         = (I1-I2)            
            Sy         = (I3-I4)          
            # 2D slopes maps      
            slopesMaps = (np.concatenate((Sx,Sy)/norma) - self.referenceSignal_2D) *self.slopesUnits
            # slopes vector
            slopes     = slopesMaps[np.where(self.validSignal==1)]
            return slopesMaps,slopes
    
        if self.postProcessing == 'slopesMaps_incidence_flux':
            # slopes-maps computation
            I1              = self.grabQuadrant(1,cameraFrame=0)*self.validI4Q
            I2              = self.grabQuadrant(2,cameraFrame=0)*self.validI4Q
            I3              = self.grabQuadrant(3,cameraFrame=0)*self.validI4Q
            I4              = self.grabQuadrant(4,cameraFrame=0)*self.validI4Q

            # global normalisation
            I4Q         = I1+I2+I3+I4
            subArea     = (self.telescope.D / self.nSubap)**2
            norma       = np.float64(self.telescope.src.nPhoton*self.telescope.samplingTime*subArea)/2 # factor 2 to account for flux separation

            # slopesMaps computation cropped to the valid pixels
            Sx         = (I1-I2)         
            Sy         = (I3-I4)   
            
            # 2D slopes maps      
            slopesMaps = (np.concatenate((Sx,Sy)/norma) - self.referenceSignal_2D) *self.slopesUnits
            
            # slopes vector
            slopes     = slopesMaps[np.where(self.validSignal==1)]
            return slopesMaps,slopes
        
        if self.postProcessing == 'fullFrame':
            # global normalization
            norma = np.sum(cameraFrame[self.validSignal])
            # 2D full-frame
            fullFrameMaps  = (cameraFrame / norma )  - self.referenceSignal_2D
            # full-frame vector
            fullFrame  = fullFrameMaps[np.where(self.validSignal==1)]
            
            return fullFrameMaps,fullFrame
        
        if self.postProcessing == 'fullFrame_incidence_flux':
            # global normalization
            subArea     = (self.telescope.D / self.nSubap)**2
            norma       = np.float64(self.telescope.src.nPhoton/2*self.telescope.samplingTime*subArea)/4/2 # factor 2 to account for flux separation
            # 2D full-frame
            fullFrameMaps  = (cameraFrame / norma )  - self.referenceSignal_2D
            # full-frame vector
            fullFrame  = fullFrameMaps[np.where(self.validSignal==1)]
            
            
            return fullFrameMaps,fullFrame
        
    def get_modulation_frame(self, radius = 6, norma = True):
        self.modulation_camera_frame = np.sum(np.abs(self.modulation_camera_em)**2,axis=0)
       
        N_trunc = int(self.nRes/2 - radius*self.modulation*self.zeroPaddingFactor )
        
        modulation_camera_frame_zoom = self.modulation_camera_frame[N_trunc:-N_trunc,N_trunc:-N_trunc]
        
        if norma:
            modulation_camera_frame_zoom/= modulation_camera_frame_zoom.max()
        
        return modulation_camera_frame_zoom
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GRAB QUADRANTS FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    def grabQuadrant(self,n,cameraFrame=0):
        
        centerPixel = int(np.round((self.cam.resolution/self.binning)/2))
        n_pixels    = centerPixel
        if cameraFrame==0:
            cameraFrame=self.cam.frame
            
        if n==3:
            I=cameraFrame[centerPixel:(centerPixel+n_pixels),centerPixel:(centerPixel+n_pixels)]
        if n==4:
            I=cameraFrame[centerPixel:(centerPixel+n_pixels),centerPixel-n_pixels:(centerPixel)]
        if n==1:
            I=cameraFrame[centerPixel-n_pixels:(+centerPixel),centerPixel-n_pixels:(centerPixel)]
        if n==2:
            I=cameraFrame[centerPixel-n_pixels:(+centerPixel),centerPixel:(centerPixel+n_pixels)]
        return I

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #properties required for backward compatibility (20/10/2020)
    @property
    def bioSignal(self):
        return self._bioSignal
    
    @bioSignal.setter
    def bioSignal(self,val):
        self._bioSignal = val
        self.signal = val
    
    @property
    def bioSignal_2D(self):
        return self._bioSignal_2D
    
    @bioSignal_2D.setter
    def bioSignal_2D(self,val):
        self._bioSignal_2D = val
        self.signal_2D = val
        
    @property
    def lightRatio(self):
        return self._lightRatio
    
    @lightRatio.setter
    def lightRatio(self,val):
        self._lightRatio = val
        if hasattr(self,'isInitialized'):
            if self.isInitialized:
                print('Updating the map if valid pixels ...')
                self.validI4Q           = (self.I4Q>=self._lightRatio*self.I4Q.max())   
                self.validSignal        = np.concatenate((self.validI4Q,self.validI4Q))
                self.validPix           = (self.initFrame>=self.lightRatio*self.initFrame.max())   
                
                # save the number of signals depending on the case    
                if self.postProcessing == 'slopesMaps' or self.postProcessing == 'slopesMaps_incidence_flux':
                    self.nSignal        = np.sum(self.validSignal)
                    # display
                    xPix,yPix = np.where(self.validI4Q==1)
                    plt.figure()
                    plt.imshow(self.I4Q.T)
                    plt.plot(xPix,yPix,'+')
                
                if self.postProcessing == 'fullFrame' or self.postProcessing == 'fullFrame_incidence_flux':
                    self.nSignal        = np.sum(self.validPix)  
                print('Done!')

    @property
    def spatialFilter(self):
        return self._spatialFilter
    
    @spatialFilter.setter
    def spatialFilter(self,val):
        self._spatialFilter = val
        if self.isInitialized:
            if val is None:         
                print('No spatial filter considered')
                self.mask = self.initial_mask
                if self.isCalibrated:
                    print('Updating the reference slopes and Wavelength Calibration for the new modulation...')
                    self.slopesUnits                = 1     
                    self.referenceSignal            = 0
                    self.referenceSignal_2D         = 0
                    self.wfs_calibration(self.telescope)
                    print('Done!')
            else:
                tmp                             = np.ones([self.nRes,self.nRes])
                tmp[:,0]                        = 0
                Tip                             = (sp.morphology.distance_transform_edt(tmp))
                Tilt                            = (sp.morphology.distance_transform_edt(np.transpose(tmp)))
                
                # normalize the TT to apply the modulation in terms of lambda/D
                self.Tip_spatial_filter                        = (((Tip/Tip.max())-0.5)*2*np.pi)
                self.Tilt_spatial_filter                       = (((Tilt/Tilt.max())-0.5)*2*np.pi)
                if val.shape == self.mask.shape:
                    print('A spatial filter is now considered')
                    self.mask = self.initial_mask * val
                    plt.figure()
                    plt.imshow(np.real(self.mask))
                    plt.title('Spatial Filter considered')
                    if self.isCalibrated:
                        print('Updating the reference slopes and Wavelength Calibration for the new modulation...')
                        self.slopesUnits                = 1     
                        self.referenceSignal            = 0
                        self.referenceSignal_2D         = 0
                        self.wfs_calibration(self.telescope)
                        print('Done!')
                else:
                    print('ERROR: wrong shape for the spatial filter. No spatial filter attached to the mask')
                    self.mask = self.initial_mask
                
    @property
    def delta_Tip(self):
        return self._delta_Tip
    
    @delta_Tip.setter
    def delta_Tip(self,val):
        self._delta_Tip = val
        if self.isCalibrated:
            self.modulation = self.modulation     
    @property
    def delta_Tilt(self):
        return self._delta_Tilt
    
    @delta_Tilt.setter
    def delta_Tilt(self,val):
        self._delta_Tilt = val
        if self.isCalibrated:
            self.modulation = self.modulation        
    @property
    def modulation(self):
        return self._modulation
    
    @modulation.setter
    def modulation(self,val):
        self._modulation = val
        if self._modulation>=(self.telescope.resolution//2):
            raise ValueError('Error the modulation radius is too large for this resolution! Consider using a larger telescope resolution!')
        if val !=0:
            self.modulation_path = []
            if self.user_modulation_path is not None:
                self.modulation_path = self.user_modulation_path.copy()
                self.nTheta          = len(self.user_modulation_path)
            else:
                # define the modulation points
                perimeter                       = np.pi*2*self._modulation    
                if self.nTheta_user_defined is None:
                    self.nTheta                     = 4*int((self.extraModulationFactor+np.ceil(perimeter/4)))
                else:
                    self.nTheta = self.nTheta_user_defined       
                
                self.thetaModulation            = np.linspace(0+self.delta_theta,2*np.pi+self.delta_theta,self.nTheta,endpoint=False)
                for i in range(self.nTheta):
                    dTheta                                  = self.thetaModulation[i]                
                    self.modulation_path.append([self.modulation*np.cos(dTheta)+self.delta_Tip, self.modulation*np.sin(dTheta)+self.delta_Tilt])

            self.phaseBuffModulation        = np.zeros([self.nTheta,self.nRes,self.nRes]).astype(xp.float32)    
            self.phaseBuffModulationLowres  = np.zeros([self.nTheta,self.telescope.resolution,self.telescope.resolution]).astype(xp.float32)          
            
            for i in range(self.nTheta):
                self.TT                                                                                                                                         = (self.modulation_path[i][0]*self.Tip+self.modulation_path[i][1]*self.Tilt)*self.telescope.pupil
                self.phaseBuffModulation[i,self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2,self.center-self.telescope.resolution//2:self.center+self.telescope.resolution//2]         = self.TT
                self.phaseBuffModulationLowres[i,:,:]                                                                                                           = self.TT
            self.phaseBuffModulationLowres_CPU = self.phaseBuffModulationLowres.copy()
            if self.gpu_available:
                if self.nTheta<=self.n_max:                        
                    self.phaseBuffModulationLowres = self.convert_for_gpu(self.phaseBuffModulationLowres)
        else:
            self.nTheta = 1

        if hasattr(self,'isCalibrated'):
            if self.isCalibrated:
                print('Updating the reference slopes and Wavelength Calibration for the new modulation...')
                self.slopesUnits                = 1     
                self.referenceSignal            = 0
                self.referenceSignal_2D         = 0
                self.wfs_calibration(self.telescope)
                print('Done!')

    @property
    def backgroundNoise(self):
        return self._backgroundNoise
    
    @backgroundNoise.setter
    def backgroundNoise(self,val):
        self._backgroundNoise = val
        if val == True:
            self.backgroundNoiseMap = []
    
    def __mul__(self,obj): 
        if obj.tag=='detector':
            
            if obj.resolution == self.nRes:
                    frame = np.sum(np.abs(self.modulation_camera_em)**2,axis=0)                
            else:
                    I = self.bioFrame
                    frame = (obj.rebin(I,(obj.resolution,obj.resolution)))
            if self.binning != 1:
                try:
                    frame = (obj.rebin(obj.frame,(obj.resolution//self.binning,obj.resolution//self.binning)))    
                except:
                    print('ERROR: the shape of the detector ('+str(obj.frame.shape)+') is not valid with the binning value requested:'+str(self.binning)+'!')
            frame = frame *(self.telescope.src.fluxMap.sum())/frame.sum()
            
            obj.integrate(frame)

        else:
            print('Error light propagated to the wrong type of object')
        return -1
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
    def print_properties(self):
        
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BIO-EDGE WFS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('{: ^18s}'.format('Pupils Diameter')      + '{: ^18s}'.format(str(self.nSubap))                       +'{: ^18s}'.format('[pixels]'   ))
        print('{: ^18s}'.format('Pupils Separation')    + '{: ^18s}'.format(str(self.n_pix_separation))             +'{: ^18s}'.format('[pixels]'   ))
        print('{: ^18s}'.format('FoV')                  + '{: ^18s}'.format(str(np.round(self.fov,2)))              +'{: ^18s}'.format('[arcsec]'   ))
        print('{: ^18s}'.format('TT Modulation')        + '{: ^18s}'.format(str(self.modulation))                   +'{: ^18s}'.format('[lamda/D]'  ))
        print('{: ^18s}'.format('Grey Area')            + '{: ^18s}'.format(str(2*self.grey_width))                 +'{: ^18s}'.format('[lamda/D]'  ))
        print('{: ^18s}'.format('PSF Core Sampling')    + '{: ^18s}'.format(str(1+self.psfCentering*3))             +'{: ^18s}'.format('[pixel(s)]' ))
        print('{: ^18s}'.format('Valid Pixels')         + '{: ^18s}'.format(str(self.nSignal))                      +'{: ^18s}'.format('[pixel(s)]' ))
        print('{: ^18s}'.format('Signal Computation')   + '{: ^18s}'.format(str(self.postProcessing)                                                ))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    def __repr__(self):
        self.print_properties()
        return ' '
                                   
        

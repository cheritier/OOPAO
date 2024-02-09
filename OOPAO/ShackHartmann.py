# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:52:09 2021

@author: cheritie
"""

import inspect
import time

import numpy as np

from .Detector import Detector
from .tools.tools import bin_ndarray

try:
    from joblib import Parallel, delayed
except:
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('WARNING: The joblib module is not installed. This would speed up considerably the operations.')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

# import ctypes
# try : 
#     mkl_rt = ctypes.CDLL('libmkl_rt.so')
#     mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
#     mkl_set_num_threads(6)
# except:
#     try:
#         mkl_rt = ctypes.CDLL('./mkl_rt.dll')
#         mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
#         mkl_set_num_threads(6)
#     except:
#         print('Could not optimize the parallelisation of the code ')


class ShackHartmann:
    def __init__(self,nSubap:float,telescope,lightRatio:float,threshold_cog:float = 0.01,\
                 is_geometric:bool = False, binning_factor:int = 1,padding_extension_factor:int = 1,\
                     threshold_convolution:float = 0.05,shannon_sampling:bool = False,unit_P2V = True):
        """SHACK-HARTMANN
        A Shack Hartmann object consists in defining a 2D grd of lenslet arrays located in the pupil plane of the telescope to estimate the local tip/tilt seen by each lenslet. 
        By default the Shack Hartmann detector is considered to be noise-free (for calibration purposes). These properties can be switched on and off on the fly (see properties)
        It requires the following parameters: 

        Parameters
        ----------
        nSubap : float
            The number of subapertures (micro-lenses) along the diameter defined by the telescope.pupil.
        telescope : TYPE
            The telescope object to which the Shack Hartmann is associated. 
            This object carries the phase, flux and pupil information.
        lightRatio : float
            Criterion to select the valid subaperture based on flux considerations.
        threshold_cog : float, optional
            Threshold (with respect to the maximum value of the image) 
            to apply to compute the center of gravity of the spots.
            The default is 0.01.
        is_geometric : bool, optional
            Flag to enable the geometric WFS. 
            If True, enables the geometric Shack Hartmann (direct measurement of gradient).
            If False, the diffractive computation is considered.
            The default is False.
        binning_factor : int, optional
            Binning factor of the detector.
            The default is 1.
        padding_extension_factor : int, optional
            Zero-padding factor on the spots intensity images. 
            This is a fast way to provide a larger field of view before the convolution 
            with LGS spots is achieved and allow to prevent wrapping effects.
            The default is 1.
        threshold_convolution : float, optional
            Threshold considered to force the gaussian spots (elungated spots) to go to zero on the edges.
            The default is 0.05.
        shannon_sampling : bool, optional
            If True, the lenslet array spots are sampled at the same sampling as the FFT (2 pix per FWHM).
            If False, the sampling is 1 pix per FWHM (default).
            The default is False.
        unit_P2V : bool, optional
                If True, the slopes units are calibrated using a Tip/Tilt normalized to 2 Pi peak-to-valley (Default).
                If False, the slopes units are calibrated using a Tip/Tilt normalized to 1 in the pupil.
                The default is True.

        Raises
        ------
        AttributeError
            DESCRIPTION.

        Returns
        -------
        None.

        ************************** PROPAGATING THE LIGHT TO THE SH OBJECT **************************
        The light can be propagated from a telescope object tel through the Shack Hartmann object wfs using the * operator:        
        _ tel*wfs
        This operation will trigger:
            _ propagation of the tel.src light through the Shack Hartmann detector (phase and flux)
            _ binning of the SH signals
            _ addition of eventual photon noise and readout noise
            _ computation of the Shack Hartmann signals
        
    
        ************************** PROPERTIES **************************
        
        The main properties of a Telescope object are listed here: 
        _ wfs.signal                     : signal measured by the Shack Hartmann
        _ wfs.signal_2D                  : 2D map of the signal measured by the Shack Hartmann
        _ wfs.random_state_photon_noise  : a random state cycle can be defined to reproduces random sequences of noise -- default is based on the current clock time 
        _ wfs.random_state_readout_noise : a random state cycle can be defined to reproduces random sequences of noise -- default is based on the current clock time   
        _ wfs.random_state_background    : a random state cycle can be defined to reproduces random sequences of noise -- default is based on the current clock time   
        _ wfs.fov_lenslet_arcsec         : Field of View of the subapertures in arcsec
        _ wfs.fov_pixel_binned_arcsec    : Field of View of the pixel in arcsec
        
        The main properties of the object can be displayed using :
            wfs.print_properties()
        
        the following properties can be updated on the fly:
            _ wfs.is_geometric          : switch between diffractive and geometric shackHartmann
            _ wfs.cam.photonNoise       : Photon noise can be set to True or False
            _ wfs.cam.readoutNoise      : Readout noise can be set to True or False
            _ wfs.lightRatio            : reset the valid subaperture selection considering the new value
        
        """ 
        self.tag                            = 'shackHartmann'
        self.telescope                      = telescope
        if self.telescope.src is None:
            raise AttributeError('The telescope was not coupled to any source object! Make sure to couple it with an src object using src*tel')
        self.is_geometric                   = is_geometric
        self.nSubap                         = nSubap
        self.lightRatio                     = lightRatio
        self.binning_factor                 = binning_factor
        self.zero_padding                   = 2
        self.padding_extension_factor       = padding_extension_factor
        self.threshold_convolution          = threshold_convolution
        self.threshold_cog                  = threshold_cog
        self.shannon_sampling               = shannon_sampling
        self.unit_P2V                       = unit_P2V
        # case where the spots are zeropadded to provide larger fOV
        if padding_extension_factor>2:
            self.n_pix_subap            = int(padding_extension_factor*self.telescope.resolution// self.nSubap)            
            self.is_extended            = True
            self.binning_factor         = padding_extension_factor
            self.zero_padding           = 1 
        else:
            self.n_pix_subap            = self.telescope.resolution// self.nSubap 
            self.is_extended            = False
        

        # different resolutions needed
        self.n_pix_subap_init           = self.telescope.resolution// self.nSubap    
        self.extra_pixel                = (self.n_pix_subap-self.n_pix_subap_init)//2         
        self.n_pix_lenslet_init         = self.n_pix_subap_init*self.zero_padding 
        self.n_pix_lenslet              = self.n_pix_subap*self.zero_padding 
        self.center                     = self.n_pix_lenslet//2 
        self.center_init                = self.n_pix_lenslet_init//2 
        self.lenslet_frame              = np.zeros([self.n_pix_subap*self.zero_padding,self.n_pix_subap*self.zero_padding], dtype =complex)
        self.outerMask                  = np.ones([self.n_pix_subap_init*self.zero_padding, self.n_pix_subap_init*self.zero_padding ])
        self.outerMask[1:-1,1:-1]       = 0
        
        # Compute camera frame in case of multiple measurements
        self.get_camera_frame_multi     = False
        # detector camera
        self.cam                        = Detector(round(nSubap*self.n_pix_subap))                     # WFS detector object
        self.cam.photonNoise            = 0
        self.cam.readoutNoise           = 0        # single lenslet
        # noies random states
        self.random_state_photon_noise      = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        self.random_state_readout_noise     = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        self.random_state_background        = np.random.RandomState(seed=int(time.time()))      # random states to reproduce sequences of noise 
        
        # field of views
        self.fov_lenslet_arcsec         = (self.n_pix_subap*206265*self.binning_factor/self.padding_extension_factor*self.telescope.src.wavelength/(self.telescope.D/self.nSubap))/(1+self.shannon_sampling)
        self.fov_pixel_arcsec           = self.fov_lenslet_arcsec/ self.n_pix_subap
        self.fov_pixel_binned_arcsec    = self.fov_lenslet_arcsec/ self.n_pix_subap_init

        X_map, Y_map= np.meshgrid(np.arange(self.n_pix_subap//self.binning_factor),np.arange(self.n_pix_subap//self.binning_factor))
        self.X_coord_map = np.atleast_3d(X_map).T
        self.Y_coord_map = np.atleast_3d(Y_map).T
        
        if telescope.src.type == 'LGS':
            self.is_LGS                 = True
        else:
            self.is_LGS                 = False
        
        # joblib parameter
        self.nJobs                  = 1
        self.joblib_prefer          = 'processes'
        
        # camera frame 
        self.camera_frame           = np.zeros([self.n_pix_subap*(self.nSubap)//self.binning_factor,self.n_pix_subap*(self.nSubap)//self.binning_factor], dtype =float)

        # cube of lenslet zero padded
        self.cube                   = np.zeros([self.nSubap**2,self.n_pix_lenslet_init,self.n_pix_lenslet_init])
        self.cube_flux              = np.zeros([self.nSubap**2,self.n_pix_subap_init,self.n_pix_subap_init],dtype=(complex))
        self.index_x                = []
        self.index_y                = []

        # phasor to center spots in the center of the lenslets
        [xx,yy]                    = np.meshgrid(np.linspace(0,self.n_pix_lenslet_init-1,self.n_pix_lenslet_init),np.linspace(0,self.n_pix_lenslet_init-1,self.n_pix_lenslet_init))
        self.phasor                = np.exp(-(1j*np.pi*(self.n_pix_lenslet_init+1)/self.n_pix_lenslet_init)*(xx+yy))
        self.phasor_tiled          = np.moveaxis(np.tile(self.phasor[:,:,None],self.nSubap**2),2,0)
        
        # Get subapertures index and flux per subaperture        
        [xx,yy]                    = np.meshgrid(np.linspace(0,self.n_pix_lenslet-1,self.n_pix_lenslet),np.linspace(0,self.n_pix_lenslet-1,self.n_pix_lenslet))
        self.phasor_expanded       = np.exp(-(1j*np.pi*(self.n_pix_lenslet+1)/self.n_pix_lenslet)*(xx+yy))
        self.phasor_expanded_tiled          = np.moveaxis(np.tile(self.phasor_expanded[:,:,None],self.nSubap**2),2,0)

        self.initialize_flux()
        for i in range(self.nSubap):
            for j in range(self.nSubap):
                self.index_x.append(i)
                self.index_y.append(j)

        self.current_nPhoton = self.telescope.src.nPhoton
        self.index_x = np.asarray(self.index_x)
        self.index_y = np.asarray(self.index_y)
        
        print('Selecting valid subapertures based on flux considerations..')

        self.photon_per_subaperture_2D = np.reshape(self.photon_per_subaperture,[self.nSubap,self.nSubap])
                
        self.valid_subapertures = np.reshape(self.photon_per_subaperture >= self.lightRatio*np.max(self.photon_per_subaperture), [self.nSubap,self.nSubap])
        
        self.valid_subapertures_1D = np.reshape(self.valid_subapertures,[self.nSubap**2])

        [self.validLenslets_x , self.validLenslets_y] = np.where(self.photon_per_subaperture_2D >= self.lightRatio*np.max(self.photon_per_subaperture))
        
        # index of valid slopes X and Y
        self.valid_slopes_maps = np.concatenate((self.valid_subapertures,self.valid_subapertures))
        
        # number of valid lenslet
        self.nValidSubaperture = int(np.sum(self.valid_subapertures))
        
        self.nSignal = 2*self.nValidSubaperture        
 
        
        if self.is_LGS:
            self.get_convolution_spot() 

        # WFS initialization
        self.initialize_wfs()
        


        
    def initialize_wfs(self):
        self.isInitialized = False

        readoutNoise = np.copy(self.cam.readoutNoise)
        photonNoise = np.copy(self.cam.photonNoise)
        
        self.cam.photonNoise        = 0
        self.cam.readoutNoise       = 0       
        
        # reference signal
        self.sx0                    = np.zeros([self.nSubap,self.nSubap])
        self.sy0                    = np.zeros([self.nSubap,self.nSubap])
        # signal vector
        self.sx                     = np.zeros([self.nSubap,self.nSubap])
        self.sy                     = np.zeros([self.nSubap,self.nSubap])
        # signal map
        self.SX                     = np.zeros([self.nSubap,self.nSubap])
        self.SY                     = np.zeros([self.nSubap,self.nSubap])
        # flux per subaperture
        self.reference_slopes_maps  = np.zeros([self.nSubap*2,self.nSubap])
        self.slopes_units           = 1
        print('Acquiring reference slopes..')        
        self.telescope.resetOPD()    
        self.wfs_measure()        
        self.reference_slopes_maps = np.copy(self.signal_2D) 
        self.isInitialized = True
        print('Done!')
        
        print('Setting slopes units..')        
        [Tip,Tilt]                         = np.meshgrid(np.linspace(0,self.telescope.resolution-1,self.telescope.resolution),np.linspace(0,self.telescope.resolution-1,self.telescope.resolution))
        
        if self.unit_P2V:            
            # normalize to 2 pi p2v
            Tip                                = (((Tip/Tip.max())-0.5)*2*np.pi)
        else:
            # normalize to 1 m RMS in the pupil
            Tip *= 1/np.std(Tip[self.telescope.pupil])

        mean_slope = np.zeros(5)
        amp = 1e-9
        for i in range(5):
            self.telescope.OPD = self.telescope.pupil*Tip*(i-2)*amp
            self.telescope.OPD_no_pupil = Tip*(i-2)*amp

            self.wfs_measure()        
            mean_slope[i] = np.mean(self.signal[:self.nValidSubaperture])
        self.p = np.polyfit(np.linspace(-2,2,5)*amp,mean_slope,deg = 1)
        self.slopes_units = np.abs(self.p[0])
        print('Done!')
        self.cam.photonNoise        = readoutNoise
        self.cam.readoutNoise       = photonNoise
        self.telescope.resetOPD()
        
        self.print_properties()

    def centroid(self,image,threshold =0.01):
        im = np.atleast_3d(image.copy())    
        im[im<(threshold*im.max())] = 0
        centroid_out         = np.zeros([im.shape[0],2])
        X_map, Y_map= np.meshgrid(np.arange(im.shape[1]),np.arange(im.shape[2]))
        X_coord_map = np.atleast_3d(X_map).T
        Y_coord_map = np.atleast_3d(Y_map).T
        norma                   = np.sum(np.sum(im,axis=1),axis=1)
        centroid_out[:,0]    = np.sum(np.sum(im*X_coord_map,axis=1),axis=1)/norma
        centroid_out[:,1]    = np.sum(np.sum(im*Y_coord_map,axis=1),axis=1)/norma
        return centroid_out
#%% DIFFRACTIVE

    def initialize_flux(self,input_flux_map = None):
        if self.telescope.tag!='asterism':
            if input_flux_map is None:
                input_flux_map = self.telescope.src.fluxMap.T
            tmp_flux_h_split = np.hsplit(input_flux_map,self.nSubap)
            self.cube_flux = np.zeros([self.nSubap**2,self.n_pix_lenslet_init,self.n_pix_lenslet_init],dtype=float)
            for i in range(self.nSubap):
                tmp_flux_v_split = np.vsplit(tmp_flux_h_split[i],self.nSubap)
                self.cube_flux[i*self.nSubap:(i+1)*self.nSubap,self.center_init - self.n_pix_subap_init//2:self.center_init+self.n_pix_subap_init//2,self.center_init - self.n_pix_subap_init//2:self.center_init+self.n_pix_subap_init//2] = np.asarray(tmp_flux_v_split)
            self.photon_per_subaperture = np.apply_over_axes(np.sum, self.cube_flux, [1,2])
            self.current_nPhoton = self.telescope.src.nPhoton
        return
    
    def get_lenslet_em_field(self,phase):
        tmp_phase_h_split = np.hsplit(phase.T,self.nSubap)
        self.cube_em = np.zeros([self.nSubap**2,self.n_pix_lenslet_init,self.n_pix_lenslet_init],dtype=complex)
        for i in range(self.nSubap):
            tmp_phase_v_split = np.vsplit(tmp_phase_h_split[i],self.nSubap)
            self.cube_em[i*self.nSubap:(i+1)*self.nSubap,self.center_init - self.n_pix_subap_init//2:self.center_init+self.n_pix_subap_init//2,self.center_init - self.n_pix_subap_init//2:self.center_init+self.n_pix_subap_init//2] = np.exp(1j*np.asarray(tmp_phase_v_split))
        self.cube_em*=np.sqrt(self.cube_flux)*self.phasor_tiled
        return self.cube_em 
  
    def fill_camera_frame(self,ind_x,ind_y,I,index_frame=None):
        if index_frame is None:
            self.camera_frame[ind_x*self.n_pix_subap//self.binning_factor:(ind_x+1)*self.n_pix_subap//self.binning_factor,ind_y*self.n_pix_subap//self.binning_factor:(ind_y+1)*self.n_pix_subap//self.binning_factor] = I        
        else:
            self.camera_frame[index_frame,ind_x*self.n_pix_subap//self.binning_factor:(ind_x+1)*self.n_pix_subap//self.binning_factor,ind_y*self.n_pix_subap//self.binning_factor:(ind_y+1)*self.n_pix_subap//self.binning_factor] = I

    def compute_camera_frame_multi(self,maps_intensity):   
        self.ind_frame =np.zeros(maps_intensity.shape[0],dtype=(int))
        self.maps_intensity = maps_intensity
        index_x = np.tile(self.index_x[self.valid_subapertures_1D],self.phase_buffer.shape[0])
        index_y = np.tile(self.index_y[self.valid_subapertures_1D],self.phase_buffer.shape[0])
        
        for i in range(self.phase_buffer.shape[0]):
            self.ind_frame[i*self.nValidSubaperture:(i+1)*self.nValidSubaperture]=i
        
        def joblib_fill_camera_frame():
            Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.fill_camera_frame)(i,j,k,l) for i,j,k,l in zip(index_x,index_y,self.maps_intensity,self.ind_frame))
            return Q
        
        joblib_fill_camera_frame()
        return
        
    #%% GEOMETRIC    
         
    def gradient_2D(self,arr):
        res_x = (np.gradient(arr,axis=0)/self.telescope.pixelSize)*self.telescope.pupil
        res_y = (np.gradient(arr,axis=1)/self.telescope.pixelSize)*self.telescope.pupil
        return res_x,res_y
        
    def lenslet_propagation_geometric(self,arr):
        
        [SLx,SLy]  = self.gradient_2D(arr)
        
        sy = (bin_ndarray(SLx, [self.nSubap,self.nSubap], operation='sum'))
        sx = (bin_ndarray(SLy, [self.nSubap,self.nSubap], operation='sum'))
        
        return np.concatenate((sx,sy))
            
    #%% LGS
    def get_convolution_spot(self): 
        # compute the projection of the LGS on the subaperture to simulate 
        #  the spots elongation using a convulotion with gaussian spot
        [X0,Y0]             = [self.telescope.src.laser_coordinates[1],-self.telescope.src.laser_coordinates[0]]     # coordinates of the LLT in [m] from the center (sign convention adjusted to match display position on camera)
        
        # 3D coordinates
        coordinates_3D           = np.zeros([3,len(self.telescope.src.Na_profile[0,:])])
        coordinates_3D_ref       = np.zeros([3,len(self.telescope.src.Na_profile[0,:])])
        delta_dx                 = np.zeros([2,len(self.telescope.src.Na_profile[0,:])])
        delta_dy                 = np.zeros([2,len(self.telescope.src.Na_profile[0,:])])
        
        # coordinates of the subapertures
        
        x_subap                 = np.linspace(-self.telescope.D//2,self.telescope.D//2,self.nSubap)
        y_subap                 = np.linspace(-self.telescope.D//2,self.telescope.D//2,self.nSubap)  
        # pre-allocate memory for shift x and y to apply to the gaussian spot

        # number of pixel
        n_pix                   = self.n_pix_lenslet
        # size of a pixel in m
        d_pix                   = (self.telescope.D/self.nSubap)/self.n_pix_lenslet_init
        v                       = np.linspace(-n_pix*d_pix/2,n_pix*d_pix/2,n_pix)
        [alpha_x,alpha_y]       = np.meshgrid(v,v)
        
        # FWHM of gaussian converted into pixel in arcsec
        sigma_spot              = self.telescope.src.FWHM_spot_up/(2*np.sqrt(np.log(2)))
        for i in range(len(self.telescope.src.Na_profile[0,:])):
                        coordinates_3D[:2,i]           = (self.telescope.D/4)*([X0,Y0]/self.telescope.src.Na_profile[0,i])
                        coordinates_3D[2,i]            = self.telescope.D**2./(8.*self.telescope.src.Na_profile[0,i])/(2.*np.sqrt(3.))
                        coordinates_3D_ref[:,i]        = coordinates_3D[:,i]-coordinates_3D[:,len(self.telescope.src.Na_profile[0,:])//2]
        C_elung                       = []
        C_gauss                       = []
        shift_x_buffer                = []
        shift_y_buffer                = []

        C_gauss                       = []
        criterion_elungation          = self.n_pix_lenslet*(self.telescope.D/self.nSubap)/self.n_pix_lenslet_init

        valid_subap_1D = np.copy(self.valid_subapertures_1D[:])
        count = -1
        # gaussian spot (for calibration)
        I_gauss  = (self.telescope.src.Na_profile[1,:][0]/(self.telescope.src.Na_profile[0,:][0]**2)) * np.exp(- ((alpha_x)**2 + (alpha_y)**2)/(2*sigma_spot**2))
        I_gauss /= I_gauss.sum()

        for i_subap in range(len(x_subap)):
            for j_subap in range(len(y_subap)):
                count += 1
                if valid_subap_1D[count]:
                    I = np.zeros([n_pix,n_pix],dtype=(complex))
                    # I_gauss = np.zeros([n_pix,n_pix],dtype=(complex))
                    shift_X                 = np.zeros(len(self.telescope.src.Na_profile[0,:]))
                    shift_Y                 = np.zeros(len(self.telescope.src.Na_profile[0,:]))
                    for i in range(len(self.telescope.src.Na_profile[0,:])):
                        coordinates_3D[:2,i]           = (self.telescope.D/4)*([X0,Y0]/self.telescope.src.Na_profile[0,i])
                        coordinates_3D[2,i]            = self.telescope.D**2./(8.*self.telescope.src.Na_profile[0,i])/(2.*np.sqrt(3.))
                        
                        coordinates_3D_ref[:,i]        = coordinates_3D[:,i]-coordinates_3D[:,len(self.telescope.src.Na_profile[0,:])//2]

                        delta_dx[0,i]   = coordinates_3D_ref[0,i]*(4/self.telescope.D)
                        delta_dy[0,i]   = coordinates_3D_ref[1,i]*(4/self.telescope.D)
            
                        delta_dx[1,i]   = coordinates_3D_ref[2,i]*(np.sqrt(3)*(4/self.telescope.D)**2)*x_subap[i_subap]
                        delta_dy[1,i]   = coordinates_3D_ref[2,i]*(np.sqrt(3)*(4/self.telescope.D)**2)*y_subap[j_subap]
                        
                        # resulting shift + conversion from radians to pixels in m
                        shift_X[i]          = 206265*self.fov_pixel_arcsec*(delta_dx[0,i] + delta_dx[1,i])
                        shift_Y[i]          = 206265*self.fov_pixel_arcsec*(delta_dy[0,i] + delta_dy[1,i])
     
        
                        I_tmp               = (self.telescope.src.Na_profile[1,:][i]/(self.telescope.src.Na_profile[0,:][i]**2))*np.exp(- ((alpha_x-shift_X[i])**2 + (alpha_y-shift_Y[i])**2)/(2*sigma_spot**2))
                                                
                        I                   += I_tmp
                        
                    # truncation of the wings of the gaussian
                    I[I<self.threshold_convolution*I.max()] = 0 
                    # normalization to conserve energy
                    I /= I.sum()
                    # save 
                    shift_x_buffer.append(shift_X)
                    shift_y_buffer.append(shift_Y)

                    C_elung.append((np.fft.fft2(I.T)))
                    C_gauss.append((np.fft.fft2(I_gauss.T)))
                    
        self.shift_x_buffer = np.asarray(shift_x_buffer)
        self.shift_y_buffer = np.asarray(shift_y_buffer)

        self.shift_x_max_arcsec = np.max(shift_x_buffer,axis=1)
        self.shift_y_max_arcsec = np.max(shift_y_buffer,axis=1)
        
        self.shift_x_min_arcsec = np.min(shift_x_buffer,axis=1)
        self.shift_y_min_arcsec = np.min(shift_y_buffer,axis=1)

        self.max_elung_x = np.max(self.shift_x_max_arcsec-self.shift_x_min_arcsec)
        self.max_elung_y = np.max(self.shift_y_max_arcsec-self.shift_y_min_arcsec)
        self.elungation_factor = np.max([self.max_elung_x,self.max_elung_y])/criterion_elungation

        if self.max_elung_x>criterion_elungation or self.max_elung_y>criterion_elungation:            
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  WARNING  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Warning: The largest spot elongation is '+str(np.round(self.elungation_factor,3))+' times larger than a subaperture! Consider using a higher resolution or increasing the padding_extension_factor parameter')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        self.C = np.asarray(C_elung.copy())
        self.C_gauss = np.asarray(C_gauss)
        self.C_elung = np.asarray(C_elung)
        
        return
    
#%% SH Measurement 
    def sh_measure(self,phase_in):
        # backward compatibility with previous version
        self.wfs_measure(phase_in=phase_in)
        return
    
    def wfs_measure(self,phase_in = None):
        if phase_in is not None:
            self.telescope.src.phase = phase_in
        
        if self.current_nPhoton != self.telescope.src.nPhoton:
            print('updating the flux of the SHWFS object')
            self.initialize_flux()
   
            
        if self.is_geometric is False:
            ##%%%%%%%%%%%%  DIFFRACTIVE SH WFS %%%%%%%%%%%%
            if np.ndim(self.telescope.OPD)==2:
                #-- case with a single wave-front to sense--
                
                # reset camera frame to be filled up
                self.camera_frame   = np.zeros([self.n_pix_subap*(self.nSubap)//self.binning_factor,self.n_pix_subap*(self.nSubap)//self.binning_factor], dtype =float)

                # normalization for FFT
                norma = self.cube.shape[1]

                # compute spot intensity
                if self.telescope.spatialFilter is None:  
                    phase = self.telescope.src.phase   
                    self.initialize_flux()

                else:
                    phase = self.telescope.phase_filtered   
                    self.initialize_flux(self.telescope.amplitude_filtered.T*self.telescope.src.fluxMap.T)
                    
                        
                I = (np.abs(np.fft.fft2(np.asarray(self.get_lenslet_em_field(phase)),axes=[1,2])/norma)**2)
                
                # reduce to valid subaperture
                I = I[self.valid_subapertures_1D,:,:]
                
                self.sum_I   = np.sum(I,axis=0)
                self.edge_subaperture_criterion = np.sum(I*self.outerMask)/np.sum(I)
                if self.edge_subaperture_criterion>0.05:
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                    print('WARNING !!!! THE LIGHT IN THE SUBAPERTURE IS MAYBE WRAPPING !!!'+str(np.round(100*self.edge_subaperture_criterion,1))+' % of the total flux detected on the edges of the subapertures. You may want to increase the seeing value or the resolution')
                    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

                # if FoV is extended, zero pad the spot intensity
                if self.is_extended:
                    I = np.pad(I,[[0,0],[self.extra_pixel,self.extra_pixel],[self.extra_pixel,self.extra_pixel]]) 
                
                # in case of LGS sensor, convolve with LGS spots to create spot elungation
                if self.is_LGS:
                    I = np.fft.fftshift(np.abs((np.fft.ifft2(np.fft.fft2(I)*self.C))),axes = [1,2])

                # Crop to get the spot at shannon sampling
                if self.shannon_sampling:
                    self.maps_intensity =  I[:,self.n_pix_subap//2:-self.n_pix_subap//2,self.n_pix_subap//2:-self.n_pix_subap//2]
                    if self.binning_factor>1:
                        self.maps_intensity =  bin_ndarray(self.maps_intensity,[self.maps_intensity.shape[0], self.n_pix_subap//self.binning_factor,self.n_pix_subap//self.binning_factor], operation='sum')
                else:
                    self.maps_intensity =  bin_ndarray(I,[I.shape[0], self.n_pix_subap//self.binning_factor,self.n_pix_subap//self.binning_factor], operation='sum')

                # bin the 2D spots intensity to get the desired number of pixel per subaperture
                if self.binning_factor>1:
                    self.maps_intensity =  bin_ndarray(self.maps_intensity,[self.maps_intensity.shape[0], self.n_pix_subap//self.binning_factor,self.n_pix_subap//self.binning_factor], operation='sum')
            
                # add photon/readout noise to 2D spots
                if self.cam.photonNoise!=0:
                    self.maps_intensity  = self.random_state_photon_noise.poisson(self.maps_intensity)
                        
                if self.cam.readoutNoise!=0:
                    self.maps_intensity += np.int64(np.round(self.random_state_readout_noise.randn(self.maps_intensity.shape[0],self.maps_intensity.shape[1],self.maps_intensity.shape[2])*self.cam.readoutNoise))
                
                # fill camera frame with computed intensity (only valid subapertures)
                def joblib_fill_camera_frame():
                    Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.fill_camera_frame)(i,j,k) for i,j,k in zip(self.index_x[self.valid_subapertures_1D],self.index_y[self.valid_subapertures_1D],self.maps_intensity))
                    return Q
                joblib_fill_camera_frame()

                # compute the centroid on valid subaperture
                self.centroid_lenslets = self.centroid(self.maps_intensity,self.threshold_cog)
                
                # discard nan and inf values
                val_inf = np.where(np.isinf(self.centroid_lenslets))
                val_nan = np.where(np.isnan(self.centroid_lenslets)) 
                
                if np.shape(val_inf)[1] !=0:
                    print('Warning! some subapertures are giving inf values!')
                    self.centroid_lenslets[np.where(np.isinf(self.centroid_lenslets))] = 0
                
                if np.shape(val_nan)[1] !=0:
                    print('Warning! some subapertures are giving nan values!')
                    self.centroid_lenslets[np.where(np.isnan(self.centroid_lenslets))] = 0
                    
                # compute slopes-maps
                self.SX[self.validLenslets_x,self.validLenslets_y] = self.centroid_lenslets[:,0]
                self.SY[self.validLenslets_x,self.validLenslets_y] = self.centroid_lenslets[:,1]
                
                signal_2D                           = np.concatenate((self.SX,self.SY)) - self.reference_slopes_maps
                signal_2D[~self.valid_slopes_maps]  = 0
                self.signal_2D                      = signal_2D/self.slopes_units
                self.signal                         = self.signal_2D[self.valid_slopes_maps]

                # assign camera_fram to sh.cam.frame
                self*self.cam
            else:
                #-- case with multiple wave-fronts to sense--

                # set phase buffer
                self.phase_buffer = np.moveaxis(self.telescope.src.phase_no_pupil,-1,0)

                # tile the valid subaperture vector to match the number of input phase
                valid_subap_1D_tiled = np.tile(self.valid_subapertures_1D,self.phase_buffer.shape[0])

                # tile the valid LGS convolution spot to match the number of input phase
                if self.is_LGS:
                    self.lgs_exp =  np.tile(self.C,[self.phase_buffer.shape[0],1,1])
                    
                # reset camera frame
                self.camera_frame   = np.zeros([self.phase_buffer.shape[0],self.n_pix_subap*(self.nSubap)//self.binning_factor,self.n_pix_subap*(self.nSubap)//self.binning_factor], dtype =float)
                
                # compute 2D intensity for multiple input wavefronts
                def compute_diffractive_signals_multi():
                    Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.get_lenslet_em_field)(i) for i in self.phase_buffer)
                    return Q                 
                emf_buffer = np.reshape(np.asarray(compute_diffractive_signals_multi()),[self.phase_buffer.shape[0]*self.nSubap**2,self.n_pix_lenslet_init,self.n_pix_lenslet_init])
                
                
                # reduce to valid subaperture
                emf_buffer = emf_buffer[valid_subap_1D_tiled,:,:]
                
                # normalization for FFT
                norma = emf_buffer.shape[1]
                # get the spots intensity
                I = np.abs(np.fft.fft2(emf_buffer)/norma)**2
                
                # if FoV is extended, zero pad the spot intensity
                if self.is_extended:
                    I = np.pad(I,[[0,0],[self.extra_pixel,self.extra_pixel],[self.extra_pixel,self.extra_pixel]]) 
                    
                # if lgs convolve with gaussian
                if self.is_LGS:
                    I = np.real(np.fft.ifft2(np.fft.fft2(I)*self.lgs_exp))
                # bin the 2D spots arrays
                self.maps_intensity = (bin_ndarray(I, [I.shape[0],self.n_pix_subap,self.n_pix_subap], operation='sum'))
                # add photon/readout noise to 2D spots
                if self.cam.photonNoise!=0:
                    self.maps_intensity  = self.random_state_photon_noise.poisson(self.maps_intensity)
                        
                if self.cam.readoutNoise!=0:
                    self.maps_intensity += np.int64(np.round(self.random_state_readout_noise.randn(self.maps_intensity.shape[0],self.maps_intensity.shape[1],self.maps_intensity.shape[2])*self.cam.readoutNoise))
                
                # fill up camera frame if requested (default is False)
                if self.get_camera_frame_multi is True:
                    self.compute_camera_frame_multi(self.maps_intensity)
                                    
                # normalization for centroid computation
                norma = np.sum(np.sum(self.maps_intensity,axis=1),axis=1)

                # centroid computation
                self.centroid_lenslets = self.centroid(self.maps_intensity,self.threshold_cog)

                # re-organization of signals according to number of wavefronts considered
                self.signal_2D = np.zeros([self.phase_buffer.shape[0],self.nSubap*2,self.nSubap])
                
                for i in range(self.phase_buffer.shape[0]):
                    self.SX[self.validLenslets_x,self.validLenslets_y] = self.centroid_lenslets[i*self.nValidSubaperture:(i+1)*self.nValidSubaperture,0]
                    self.SY[self.validLenslets_x,self.validLenslets_y] = self.centroid_lenslets[i*self.nValidSubaperture:(i+1)*self.nValidSubaperture,1]
                    signal_2D = np.concatenate((self.SX,self.SY)) - self.reference_slopes_maps
                    signal_2D[~self.valid_slopes_maps] = 0
                    self.signal_2D[i,:,:] = signal_2D/self.slopes_units

                self.signal = self.signal_2D[:,self.valid_slopes_maps].T
                # assign camera_fram to sh.cam.frame
                self*self.cam

        else:
            ##%%%%%%%%%%%%  GEOMETRIC SH WFS %%%%%%%%%%%%
            if np.ndim(self.telescope.src.phase)==2:
                self.camera_frame   = np.zeros([self.n_pix_subap*(self.nSubap)//self.binning_factor,self.n_pix_subap*(self.nSubap)//self.binning_factor], dtype =float)

                self.signal_2D = self.lenslet_propagation_geometric(self.telescope.src.phase_no_pupil)*self.valid_slopes_maps/self.slopes_units
                    
                self.signal = self.signal_2D[self.valid_slopes_maps]
                
                self*self.cam

            else:
                self.phase_buffer = np.moveaxis(self.telescope.src.phase_no_pupil,-1,0)

                def compute_geometric_signals():
                    Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.lenslet_propagation_geometric)(i) for i in self.phase_buffer)
                    return Q
                maps = compute_geometric_signals()
                self.signal_2D = np.asarray(maps)/self.slopes_units
                self.signal = self.signal_2D[:,self.valid_slopes_maps].T

    def print_properties(self):
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SHACK HARTMANN WFS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('{: ^20s}'.format('Subapertures')         + '{: ^18s}'.format(str(self.nSubap))                                   )
        print('{: ^20s}'.format('Subaperture Size')     + '{: ^18s}'.format(str(np.round(self.telescope.D/self.nSubap,2)))      +'{: ^18s}'.format('[m]'   ))
        print('{: ^20s}'.format('Pixel FoV')            + '{: ^18s}'.format(str(np.round(self.fov_pixel_binned_arcsec,2)))      +'{: ^18s}'.format('[arcsec]'   ))
        print('{: ^20s}'.format('Subapertue FoV')       + '{: ^18s}'.format(str(np.round(self.fov_lenslet_arcsec,2)))           +'{: ^18s}'.format('[arcsec]'  ))
        print('{: ^20s}'.format('Valid Subaperture')    + '{: ^18s}'.format(str(str(self.nValidSubaperture))))                   
        print('{: ^20s}'.format('Binning Factor')    + '{: ^18s}'.format(str(str(self.binning_factor))))                   

        if self.is_LGS:    
            print('{: ^20s}'.format('Spot Elungation')    + '{: ^18s}'.format(str(100*np.round(self.elungation_factor,3)))      +'{: ^18s}'.format('% of a subap' ))
        print('{: ^20s}'.format('Geometric WFS')    + '{: ^18s}'.format(str(self.is_geometric)))
        print('{: ^20s}'.format('Shannon Sampling')    + '{: ^18s}'.format(str(self.shannon_sampling)))

        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')        
        if self.is_geometric:
            print('WARNING: THE PHOTON AND READOUT NOISE ARE NOT CONSIDERED FOR GEOMETRIC SH-WFS')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    @property
    def is_geometric(self):
        return self._is_geometric
    
    @is_geometric.setter
    def is_geometric(self,val):
        self._is_geometric = val
        if hasattr(self,'isInitialized'):
            if self.isInitialized:
                print('Re-initializing WFS...')
                self.initialize_wfs()
            
    @property
    def C(self):
        return self._C
    
    @C.setter
    def C(self,val):
        self._C = val
        if hasattr(self,'isInitialized'):
            if self.isInitialized:
                print('Re-initializing WFS...')
                self.initialize_wfs()      
    @property
    def lightRatio(self):
        return self._lightRatio
    
    @lightRatio.setter
    def lightRatio(self,val):
        self._lightRatio = val
        if hasattr(self,'isInitialized'):
            if self.isInitialized:
                print('Selecting valid subapertures based on flux considerations..')

                self.valid_subapertures = np.reshape(self.photon_per_subaperture >= self.lightRatio*np.max(self.photon_per_subaperture), [self.nSubap,self.nSubap])
        
                self.valid_subapertures_1D = np.reshape(self.valid_subapertures,[self.nSubap**2])

                [self.validLenslets_x , self.validLenslets_y] = np.where(self.photon_per_subaperture_2D >= self.lightRatio*np.max(self.photon_per_subaperture))
        
                # index of valid slopes X and Y
                self.valid_slopes_maps = np.concatenate((self.valid_subapertures,self.valid_subapertures))
        
                # number of valid lenslet
                self.nValidSubaperture = int(np.sum(self.valid_subapertures))
        
                self.nSignal = 2*self.nValidSubaperture
                
                print('Re-initializing WFS...')
                self.initialize_wfs()
                print('Done!')
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WFS INTERACTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def __mul__(self,obj): 
        if obj.tag=='detector':
            obj.frame = self.camera_frame
        else:
            print('Error light propagated to the wrong type of object')
        return -1
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
    def show(self):
        attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
        print(self.tag+':')
        for a in attributes:
            if not(a[0].startswith('__') and a[0].endswith('__')):
                if not(a[0].startswith('_')):
                    if not np.shape(a[1]):
                        tmp=a[1]
                        try:
                            print('          '+str(a[0])+': '+str(tmp.tag)+' object') 
                        except:
                            print('          '+str(a[0])+': '+str(a[1])) 
                    else:
                        if np.ndim(a[1])>1:
                            print('          '+str(a[0])+': '+str(np.shape(a[1])))   
    
            
    def __repr__(self):
        self.print_properties()
        return ' '
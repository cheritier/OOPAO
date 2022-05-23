# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:52:09 2021

@author: cheritie
"""

import numpy as np
import scipy.ndimage as sp
import sys
import inspect
import time
import matplotlib.pyplot as plt
import multiprocessing
from AO_modules.Detector import Detector
from AO_modules.tools.tools import bin_ndarray
import scipy.ndimage as ndimage

try:
    from joblib import Parallel, delayed
except:
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('WARNING: The joblib module is not installed. This would speed up considerably the operations.')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

import ctypes
try : 
    mkl_rt = ctypes.CDLL('libmkl_rt.so')
    mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
    mkl_set_num_threads(6)
except:
    try:
        mkl_rt = ctypes.CDLL('./mkl_rt.dll')
        mkl_set_num_threads = mkl_rt.MKL_Set_Num_Threads
        mkl_set_num_threads(6)
    except:
        print('Could not optimize the parallelisation of the code ')



class ShackHartmann:
    def __init__(self,nSubap,telescope,lightRatio,LGS = None,threshold_cog = 0,is_geometric = False, binning_factor = 1 ):
        self.tag                    = 'shackHartmann'
        self.telescope              = telescope
        self.is_geometric           = is_geometric
        self.nSubap                 = nSubap
        self.lightRatio             = lightRatio
        self.pupil                  = telescope.pupil.astype(float)     
        self.zero_padding           = 2
        self.n_pix_subap            = self.telescope.resolution// self.nSubap 
        self.n_pix_lenslet          = self.n_pix_subap*self.zero_padding 
        self.center                 = self.n_pix_lenslet//2 
        self.threshold_cog          = threshold_cog
        self.get_camera_frame_multi = False
        self.cam                    = Detector(round(nSubap*self.n_pix_subap))                     # WFS detector object
        self.cam.photonNoise        = 0
        self.cam.readoutNoise       = 0        # single lenslet 
        self.lenslet_frame          = np.zeros([self.n_pix_subap*self.zero_padding,self.n_pix_subap*self.zero_padding], dtype =complex)
        self.photon_per_subaperture = np.zeros(self.nSubap**2)
        self.binning_factor         = binning_factor
        if LGS is None:
            self.is_LGS                 = False
        else:
            self.is_LGS                 = True
        
        self.LGS = LGS
        # joblib parameter
        self.nJobs                  = 1
        self.joblib_prefer          = 'processes'
        
        # camera frame 
        self.camera_frame           = np.zeros([self.n_pix_subap*(self.nSubap)//self.binning_factor,self.n_pix_subap*(self.nSubap)//self.binning_factor], dtype =float)
        
        # cube of lenslet zero padded
        self.cube                   = np.zeros([self.nSubap**2,self.n_pix_lenslet,self.n_pix_lenslet])
        
        self.cube_flux              = np.zeros([self.nSubap**2,self.n_pix_subap,self.n_pix_subap],dtype=(complex))

        self.index_x                = []
        self.index_y                = []

        # phasor to center spots in the center of the lenslets
        [xx,yy]                    = np.meshgrid(np.linspace(0,self.n_pix_lenslet-1,self.n_pix_lenslet),np.linspace(0,self.n_pix_lenslet-1,self.n_pix_lenslet))
        self.phasor                = np.exp(-(1j*np.pi*(self.n_pix_lenslet+1)/self.n_pix_lenslet)*(xx+yy))
        count=0
        # Get subapertures index and fluix per subaperture
        for i in range(self.nSubap):
            for j in range(self.nSubap):
                self.index_x.append(i)
                self.index_y.append(j)
                
                mask_amp_SH = np.sqrt(self.telescope.src.fluxMap[i*self.n_pix_subap:(i+1)*self.n_pix_subap,j*self.n_pix_subap:(j+1)*self.n_pix_subap]).astype(float)
                # define the cube of lenslet arrays
                self.cube[count,self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2,self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2] = mask_amp_SH
                self.cube_flux[count,:,:] = mask_amp_SH*self.phasor[self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2,self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2]

                self.photon_per_subaperture[count] = mask_amp_SH.sum()
                count+=1
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
        self.sh_measure()        
        self.reference_slopes_maps = np.copy(self.signal_2D) 
        self.isInitialized = True
        print('Done!')
        
        print('Setting slopes units..')        
        [Tip,Tilt]                         = np.meshgrid(np.linspace(0,self.telescope.resolution-1,self.telescope.resolution),np.linspace(0,self.telescope.resolution-1,self.telescope.resolution))
        Tip                                = (((Tip/Tip.max())-0.5)*2*np.pi)
        mean_slope = np.zeros(5)
        amp = 1e-9
        for i in range(5):
            self.telescope.OPD = self.telescope.pupil*Tip*(i-2)*amp
            self.telescope.OPD_no_pupil = Tip*(i-2)*amp

            self.sh_measure()        
            mean_slope[i] = np.mean(self.signal[:self.nSignal//2])
        self.p = np.polyfit(np.linspace(-2,2,5)*amp,mean_slope,deg = 1)
        self.slopes_units = self.p[0]
        print('Done!')
        self.cam.photonNoise        = readoutNoise
        self.cam.readoutNoise       = photonNoise
        self.telescope.resetOPD()

    def centroid(self,im, threshold = 0):
        im[im<threshold*im.max()]=0
        [x,y] = ndimage.center_of_mass(im.T)
        return x,y
#%% DIFFRACTIVE 

# single measurement 
    def fill_cube_LGS(self,mask,ind_x,ind_y,LGS):
        
        support         = np.copy(self.lenslet_frame)
        
        lenslet_phase   = self.telescope.src.phase[ind_x*self.n_pix_subap:(ind_x+1)*self.n_pix_subap,ind_y*self.n_pix_subap:(ind_y+1)*self.n_pix_subap]
        
        support[self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2,self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2] = mask*np.exp(1j*lenslet_phase)
        
        norma           = support.shape[0]
        
        I               = np.abs(np.fft.fftshift(np.fft.fft2(support))/norma)**2
        
        K               = np.real(np.fft.ifft2(np.fft.fft2(I)*LGS))

        return K
  
    def fill_cube(self,mask,ind_x,ind_y):
        
        support         = np.copy(self.lenslet_frame)
        
        lenslet_phase   = self.telescope.src.phase[ind_x*self.n_pix_subap:(ind_x+1)*self.n_pix_subap,ind_y*self.n_pix_subap:(ind_y+1)*self.n_pix_subap]
        
        support[self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2,self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2] = mask*np.exp(1j*lenslet_phase)
        
        return support    
# multiple measurements
    def get_phase_buffer(self,amp,ind_x,ind_y):
        support = np.copy(self.lenslet_frame)
                
        support[self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2,self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2] = np.exp(1j*self.telescope.src.phase[ind_x*self.n_pix_subap:(ind_x+1)*self.n_pix_subap,ind_y*self.n_pix_subap:(ind_y+1)*self.n_pix_subap])
        
        return support*amp*self.phasor    
    
    def get_lenslet_phase_buffer(self,phase_in):
        self.telescope.src.phase = np.squeeze(phase_in) 
        
        def joblib_get_phase_buffer():
            Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.get_phase_buffer)(i,j,k) for i,j,k in zip(self.cube[self.valid_subapertures_1D,:,:],self.index_x[self.valid_subapertures_1D],self.index_y[self.valid_subapertures_1D]))
            return Q
        
        out = np.asarray(joblib_get_phase_buffer())
        
        return out
        
    def fill_camera_frame(self,ind_x,ind_y,I,index_frame=None):
        if index_frame is None:
            self.camera_frame[ind_x*self.n_pix_subap//self.binning_factor:(ind_x+1)*self.n_pix_subap//self.binning_factor,ind_y*self.n_pix_subap//self.binning_factor:(ind_y+1)*self.n_pix_subap//self.binning_factor] = I        
        else:
            self.camera_frame[index_frame,ind_x*self.n_pix_subap//self.binning_factor:(ind_x+1)*self.n_pix_subap//self.binning_factor,ind_y*self.n_pix_subap//self.binning_factor:(ind_y+1)*self.n_pix_subap//self.binning_factor] = I

    def compute_camera_frame_multi(self,maps_intensisty):   
        self.ind_frame =np.zeros(maps_intensisty.shape[0],dtype=(int))
        self.maps_intensisty = maps_intensisty
        index_x = np.tile(self.index_x[self.valid_subapertures_1D],self.phase_buffer.shape[0])
        index_y = np.tile(self.index_y[self.valid_subapertures_1D],self.phase_buffer.shape[0])
        
        for i in range(self.phase_buffer.shape[0]):
            self.ind_frame[i*self.nValidSubaperture:(i+1)*self.nValidSubaperture]=i
        
        def joblib_fill_camera_frame():
            Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.fill_camera_frame)(i,j,k,l) for i,j,k,l in zip(index_x,index_y,self.maps_intensisty,self.ind_frame))
            return Q
        
        joblib_fill_camera_frame()
        return
        
    #%% GEOMETRIC self   
         
    def gradient_2D(self,arr):
        res_x = (np.gradient(arr,axis=1)/self.telescope.pixelSize)*self.telescope.pupil
        res_y = (np.gradient(arr,axis=0)/self.telescope.pixelSize)*self.telescope.pupil
        return res_x,res_y
        
    def lenslet_propagation_geometric(self,arr):
        
        [SLx,SLy]  = self.gradient_2D(arr)
        
        sx = (bin_ndarray(SLx, [self.nSubap,self.nSubap], operation='mean'))
        sy = (bin_ndarray(SLy, [self.nSubap,self.nSubap], operation='mean'))
        
        return np.concatenate((sx,sy))
            
            
#%% self Measurement 
    def sh_measure(self,phase_in = None):
        if phase_in is not None:
            self.telescope.src.phase = phase_in
        
        if self.current_nPhoton != self.telescope.src.nPhoton:
            print('updating the flux of the SHWFS object')
            count = 0
            # Get subapertures index and fluix per subaperture
            for i in range(self.nSubap):
                for j in range(self.nSubap):
                    mask_amp_SH = np.sqrt(self.telescope.src.fluxMap[i*self.n_pix_subap:(i+1)*self.n_pix_subap,j*self.n_pix_subap:(j+1)*self.n_pix_subap]).astype(float)
                    # define the cube of lenslet arrays
                    self.cube[count,self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2,self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2] = mask_amp_SH
                    self.cube_flux[count,:,:] = mask_amp_SH*self.phasor[self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2,self.center - self.n_pix_subap//2:self.center+self.n_pix_subap//2]
                    self.photon_per_subaperture[count] = mask_amp_SH.sum()
                    count+=1
            self.current_nPhoton = self.telescope.src.nPhoton
            self.photon_per_subaperture_2D = np.reshape(self.photon_per_subaperture,[self.nSubap,self.nSubap])     
            
        if self.is_geometric is False:
            
            if np.ndim(self.telescope.src.phase)==2:
                # reset camera frame
                self.camera_frame   = np.zeros([self.n_pix_subap*(self.nSubap)//self.binning_factor,self.n_pix_subap*(self.nSubap)//self.binning_factor], dtype =float)
                
                if self.is_LGS:
                    def fill_em_cube():
                        Q=Parallel(n_jobs=self.nJobs,prefer=self.joblib_prefer)(delayed(self.fill_cube_LGS)(i,j,k,l) for i,j,k,l in zip(self.cube_flux[self.valid_subapertures_1D,:,:],self.index_x[self.valid_subapertures_1D],self.index_y[self.valid_subapertures_1D],self.LGS.C))
                        return Q
                    
                    I = (np.asarray(fill_em_cube()))
                else:
                    def fill_em_cube():
                        Q=Parallel(n_jobs=self.nJobs,prefer=self.joblib_prefer)(delayed(self.fill_cube)(i,j,k) for i,j,k in zip(self.cube_flux[self.valid_subapertures_1D,:,:],self.index_x[self.valid_subapertures_1D],self.index_y[self.valid_subapertures_1D]))
                        return Q
                    norma = self.lenslet_frame.shape[0]
                    I = np.abs(np.fft.fft2(np.asarray(fill_em_cube()))/norma)**2                

                self.maps_intensity =  bin_ndarray(I, [I.shape[0], self.n_pix_subap//self.binning_factor,self.n_pix_subap//self.binning_factor], operation='sum')
                
                if self.cam.photonNoise!=0:
                    rs = np.random.RandomState(seed=int(time.time()))
                    self.maps_intensity  = rs.poisson(self.maps_intensity)
                        
                if self.cam.readoutNoise!=0:
                    self.maps_intensity += np.int64(np.round(np.random.randn(self.maps_intensity.shape[0],self.maps_intensity.shape[1],self.maps_intensity.shape[2])*self.cam.readoutNoise))
                
                def joblib_fill_camera_frame():
                    Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.fill_camera_frame)(i,j,k) for i,j,k in zip(self.index_x[self.valid_subapertures_1D],self.index_y[self.valid_subapertures_1D],self.maps_intensity))
                    return Q
        
                joblib_fill_camera_frame()
                
                def joblib_centroid():
                    Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.centroid)(i) for i in self.maps_intensity)
                    return Q
                
                centroid_single = np.asarray(joblib_centroid())
                val_inf = np.where(np.isinf(centroid_single))
                val_nan = np.where(np.isnan(centroid_single)) 
                if np.shape(val_inf)[1] !=0:
                    print('Warning! some subapertures are giving inf values!')
                    centroid_single[np.where(np.isinf(centroid_single))] = 0
                
                if np.shape(val_nan)[1] !=0:
                    print('Warning! some subapertures are giving nan values!')
                    centroid_single[np.where(np.isnan(centroid_single))] = 0

                self.SX[self.validLenslets_x,self.validLenslets_y] = centroid_single[:,0]
                self.SY[self.validLenslets_x,self.validLenslets_y] = centroid_single[:,1]
                
                signal_2D                           = np.concatenate((self.SX,self.SY)) - self.reference_slopes_maps
                
                signal_2D[~self.valid_slopes_maps]  = 0
                self.signal_2D                      = signal_2D/self.slopes_units
                self.signal                         = self.signal_2D[self.valid_slopes_maps]
                
                self*self.cam
                        
            else:
                # set phase buffer
                self.phase_buffer = np.moveaxis(self.telescope.src.phase_no_pupil,-1,0)
                # reset camera frame
                self.camera_frame   = np.zeros([self.phase_buffer.shape[0],self.n_pix_subap*(self.nSubap)//self.binning_factor,self.n_pix_subap*(self.nSubap)//self.binning_factor], dtype =float)

                def compute_diffractive_signals_multi():
                    Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.get_lenslet_phase_buffer)(i) for i in self.phase_buffer)
                    return Q 
                self.maps_intensity = np.reshape(np.asarray(compute_diffractive_signals_multi()),[self.phase_buffer.shape[0]*np.sum(self.valid_subapertures_1D),self.n_pix_lenslet,self.n_pix_lenslet])
                norma = self.maps_intensity.shape[1]
                
                F = np.abs(np.fft.fft2(self.maps_intensity)/norma)**2
                
                F_binned = (bin_ndarray(F, [F.shape[0],self.n_pix_subap,self.n_pix_subap], operation='sum'))

                if self.cam.photonNoise!=0:
                    rs = np.random.RandomState(seed=int(time.time()))
                    F_binned  = rs.poisson(F_binned)
                        
                if self.cam.readoutNoise!=0:
                    F_binned += np.int64(np.round(np.random.randn(F_binned.shape[0],F_binned.shape[1],F_binned.shape[2])*self.cam.readoutNoise))
                
                def joblib_centroid():
                    Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.centroid)(i) for i in F_binned)
                    return Q
                
                if self.get_camera_frame_multi is True:
                    self.compute_camera_frame_multi(F_binned)
                
                self.centroid_multi = np.asarray(joblib_centroid())
                
                self.signal_2D = np.zeros([self.phase_buffer.shape[0],self.nSubap*2,self.nSubap])

                for i in range(self.phase_buffer.shape[0]):
                    self.SX[self.validLenslets_x,self.validLenslets_y] = self.centroid_multi[i*self.nValidSubaperture:(i+1)*self.nValidSubaperture,0]
                    self.SY[self.validLenslets_x,self.validLenslets_y] = self.centroid_multi[i*self.nValidSubaperture:(i+1)*self.nValidSubaperture,1]
                    signal_2D = np.concatenate((self.SX,self.SY)) - self.reference_slopes_maps
                    signal_2D[~self.valid_slopes_maps] = 0
                    self.signal_2D[i,:,:] = signal_2D/self.slopes_units
                    
                self.signal = self.signal_2D[:,self.valid_slopes_maps].T
                self*self.cam

        else:
            if np.ndim(self.telescope.src.phase)==2:
                self.signal_2D = self.lenslet_propagation_geometric(self.telescope.src.phase_no_pupil)*self.valid_slopes_maps/self.slopes_units
                    
                self.signal = self.signal_2D[self.valid_slopes_maps]
                
            else:
                self.phase_buffer = np.moveaxis(self.telescope.src.phase_no_pupil,-1,0)

                def compute_diffractive_signals():
                    Q=Parallel(n_jobs=1,prefer='processes')(delayed(self.lenslet_propagation_geometric)(i) for i in self.phase_buffer)
                    return Q
                maps = compute_diffractive_signals()
                self.signal_2D = np.asarray(maps)/self.slopes_units
                self.signal = self.signal_2D[:,self.valid_slopes_maps].T


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
    
            
